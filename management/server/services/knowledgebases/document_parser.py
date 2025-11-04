#  Copyright 2025 zstar1003. All Rights Reserved.
#  Project source code: https://github.com/zstar1003/ragflow-plus

import gc
import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from urllib.parse import urlparse

import requests
from database import MINIO_CONFIG, get_es_client, get_minio_client
# mineru 패키지 사용
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.utils.enum_class import MakeMode

from . import logger
from .excel_parser import parse_excel_file
from .rag_tokenizer import RagTokenizer
from .korean_tokenizer import KoreanTokenizer
from .utils import _create_task_record, _update_document_progress, _update_kb_chunk_count, generate_uuid, get_bbox_from_block
from bs4 import BeautifulSoup

tknzr = RagTokenizer()
korean_tokenizer = KoreanTokenizer()

# HTML 테이블을 마크다운 테이블로 변환하는 함수
def html_table_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html  # 테이블이 아니면 원본 반환

    rows = table.find_all("tr")
    md_rows = []
    for i, row in enumerate(rows):
        cols = [col.get_text(strip=True) for col in row.find_all(["td", "th"])]
        md_rows.append("| " + " | ".join(cols) + " |")
        if i == 0:  # 헤더 다음 구분선
            md_rows.append("|" + "|".join([" --- "]*len(cols)) + "|")
    return "\n".join(md_rows)

# HTML 태그 제거 함수 (정규식 사용)
def html_to_text(html):
    # <...> 형태의 태그를 모두 제거
    return re.sub(r'<[^>]+>', ' ', html)

def tokenize_text(text):
    """토크나이저로 텍스트를 토큰화합니다."""
    logger.debug(f"[Parser-DEBUG] KoreanTokenizer 사용 전: {text[:100]}")
    # HTML 태그가 포함된 경우 태그 제거 (정규식 사용)
    if '<' in text and '>' in text:
        text = html_to_text(text)
        # text = tknzr.tokenize(text)
        # logger.info(f"[Parser-INFO] 기본Tokenizer 사용: {text[:100]}")
    tokens = korean_tokenizer.tokenize(text)
    if isinstance(tokens, list):
        text = ' '.join(tokens)
        logger.debug(f"[Parser-DEBUG] KoreanTokenizer 사용: {text[:100]}")
        return text

def merge_title_text_blocks(content_list, block_info_list, middle_json_blocks=None):
    """
    middle_json_blocks의 block_type을 사용해서 title 블록과 바로 다음 text 블록을 하나의 청크로 합치는 함수
    
    Args:
        content_list: MinerU pipeline_union_make 결과
        block_info_list: 블록 정보 리스트 (page_idx, bbox 포함)
        middle_json_blocks: middle_json에서 추출된 블록 타입 정보 리스트 [{"block_type": "title"}, {"block_type": "text"}, ...]
    
    Returns:
        merged_content_list: 병합된 콘텐츠 리스트
        merged_block_info_list: 병합된 블록 정보 리스트
    """
    merged_content_list = []
    merged_block_info_list = []
    skip_next = False
    title_count = 0
    text_count = 0
    merged_count = 0
    
    # middle_json_blocks가 있으면 block_type 정보를 사용, 없으면 기존 방식 사용
    use_middle_json = middle_json_blocks is not None and len(middle_json_blocks) == len(content_list)
    
    if use_middle_json:
        logger.info(f"[Parser-INFO] middle_json block_type 사용: {len(middle_json_blocks)} blocks")
    else:
        logger.info(f"[Parser-INFO] content_list type 사용 (middle_json 없음)")
    
    for i, chunk_data in enumerate(content_list):
        # 이전에 병합되어 건너뛸 블록인 경우
        if skip_next:
            skip_next = False
            continue
        
        # middle_json_blocks에서 block_type 정보 가져오기
        if use_middle_json and i < len(middle_json_blocks):
            current_middle_type = middle_json_blocks[i].get('block_type', '').lower()
            next_middle_type = middle_json_blocks[i + 1].get('block_type', '').lower() if i + 1 < len(middle_json_blocks) else ''
            
            # middle_json_blocks의 block_type을 직접 사용
            is_title_block = current_middle_type == 'title'
            is_next_text_block = next_middle_type == 'text' if i + 1 < len(middle_json_blocks) else False
        else:
            # 기존 방식: content_list의 type 사용
            title_types = ["title", "header", "heading", "h1", "h2", "h3", "h4", "h5", "h6"]
            text_types = ["text", "paragraph", "para"]
            
            current_type = chunk_data.get("type", "").lower()
            next_type = content_list[i + 1].get("type", "").lower() if i + 1 < len(content_list) else ""
            
            is_title_block = any(title_type in current_type for title_type in title_types)
            is_next_text_block = any(text_type in next_type for text_type in text_types) if i + 1 < len(content_list) else False
        
        # 블록 타입별 카운트
        if is_title_block:
            title_count += 1
        elif (use_middle_json and i < len(middle_json_blocks) and middle_json_blocks[i].get('block_type', '').lower() == 'text') or \
             (not use_middle_json and any(text_type in chunk_data.get("type", "").lower() for text_type in ["text", "paragraph", "para"])):
            text_count += 1
            
        # title 블록이고 다음 블록이 text인 경우 병합
        if (is_title_block and i + 1 < len(content_list) and is_next_text_block):
            
            title_chunk = chunk_data
            text_chunk = content_list[i + 1]
            
            # title과 text 내용 병합
            title_content = title_chunk.get("text", "").strip()
            text_content = text_chunk.get("text", "").strip()
            
            logger.info(f"[Parser-INFO] ✅ Title-Text 병합 발견: idx={i}")
            
            # 병합된 내용이 비어있지 않은 경우만 처리
            if title_content or text_content:
                merged_content = f"{title_content}\n{text_content}".strip()
                
                # 새로운 병합 블록 생성
                merged_chunk = {
                    "type": "text",  # 병합된 블록은 text 타입으로 설정
                    "text": merged_content
                }
                
                # bbox 정보 병합
                title_bbox = [0, 0, 0, 0]
                text_bbox = [0, 0, 0, 0]
                title_page = 0
                text_page = 0
                
                # title 블록의 정보 가져오기
                if i < len(block_info_list):
                    title_info = block_info_list[i]
                    title_page = title_info.get("page_idx", 0)
                    title_bbox = title_info.get("bbox", [0, 0, 0, 0])
                
                # text 블록의 정보 가져오기
                if i + 1 < len(block_info_list):
                    text_info = block_info_list[i + 1]
                    text_page = text_info.get("page_idx", 0)
                    text_bbox = text_info.get("bbox", [0, 0, 0, 0])
                
                # bbox 병합: 두 블록을 포함하는 최소 경계 상자 계산
                if title_bbox != [0, 0, 0, 0] and text_bbox != [0, 0, 0, 0]:
                    # 여러 페이지에 걸친 경우 첫 번째 페이지의 bbox 사용
                    if title_page == text_page:
                        # 같은 페이지: 두 bbox를 포함하는 경계 상자 계산
                        merged_bbox = [
                            min(title_bbox[0], text_bbox[0]),  # x1 최소값
                            min(title_bbox[1], text_bbox[1]),  # y1 최소값
                            max(title_bbox[2], text_bbox[2]),  # x2 최대값
                            max(title_bbox[3], text_bbox[3])   # y2 최대값
                        ]
                        merged_page = title_page
                    else:
                        # 다른 페이지: 첫 번째 블록(title)의 정보 사용
                        merged_bbox = title_bbox
                        merged_page = title_page
                else:
                    # bbox 정보가 없는 경우 title 블록 정보 사용
                    merged_bbox = title_bbox
                    merged_page = title_page
                
                # 병합된 블록 정보 생성
                merged_block_info = {
                    "page_idx": merged_page,
                    "bbox": merged_bbox
                }
                
                merged_content_list.append(merged_chunk)
                merged_block_info_list.append(merged_block_info)
                
                # 다음 텍스트 블록은 건너뛰기
                skip_next = True
                merged_count += 1
                
                logger.info(f"[Parser-INFO] ✅ Title과 Text 블록 병합 완료: page={merged_page}")
            else:
                # 내용이 비어있는 경우 원본 블록들을 개별적으로 추가
                merged_content_list.append(title_chunk)
                if i < len(block_info_list):
                    merged_block_info_list.append(block_info_list[i])
        else:
            # title이 아니거나 다음이 text가 아닌 경우 그대로 추가
            merged_content_list.append(chunk_data)
            if i < len(block_info_list):
                merged_block_info_list.append(block_info_list[i])
            else:
                # block_info_list가 부족한 경우 기본값 추가
                merged_block_info_list.append({"page_idx": 0, "bbox": [0, 0, 0, 0]})
    
    logger.info(f"[Parser-INFO] 블록 병합 통계: total={len(content_list)}, title={title_count}, text={text_count}, merged={merged_count}")
    logger.info(f"[Parser-INFO] ✅ 블록 병합 완료: {len(content_list)} -> {len(merged_content_list)} 블록")
    return merged_content_list, merged_block_info_list


def is_page_header(bbox, page_idx, block_idx_in_page, page_width=None, page_height=None):
    """
    bbox가 페이지 헤더 영역에 위치하는지 판단하는 함수
    
    Args:
        bbox: [x1, y1, x2, y2] 형식의 좌표
        page_idx: 현재 페이지 인덱스
        block_idx_in_page: 해당 페이지에서의 블록 순서 (0부터 시작)
        page_width: 페이지 너비 (기본값: 595 - A4 기준)
        page_height: 페이지 높이 (기본값: 842 - A4 기준)
    
    Returns:
        bool: 헤더 영역이면 True
    
    헤더 판단 기준:
    1. 해당 페이지의 첫 번째 블록 (block_idx_in_page == 0)
    2. x1 (좌측 시작)이 페이지 50%보다 오른쪽에 위치
    """
    if not bbox or len(bbox) != 4:
        return False
    
    x1, y1, x2, y2 = bbox
    
    # 기본 페이지 크기 (A4 기준)
    if page_width is None:
        page_width = 595
    if page_height is None:
        page_height = 842
    
    # 1. 페이지의 첫 번째 블록인지 체크
    is_first_block = (block_idx_in_page == 0)
    
    # 2. 오른쪽 편향 체크: x1이 페이지 70%보다 오른쪽
    page_right_threshold = page_width * 0.5
    is_right_aligned = (x1 > page_right_threshold)
    
    # 헤더 조건: 페이지 첫 번째 블록 AND 오른쪽 편향
    is_header = is_first_block and is_right_aligned
    
#    if is_header:
    logger.info(f"[Parser-DEBUG] 헤더 감지: page={page_idx}, block_idx={block_idx_in_page}, "
                f"bbox={bbox}, x1={x1:.1f}, page_w={page_width}")
    
    return is_header


def custom_merge_and_split(content_list, block_info_list, middle_json_blocks=None):
    """
    조항별로 블록을 병합하고 context length에 맞게 분리하는 함수
    
    1. context length보다 큰 블록 먼저 분리
    2. '제n조' 패턴으로 조항별 병합
    3. 병합 시 context length 초과하면 분리
    
    Args:
        content_list: 블록 리스트
        block_info_list: 블록 정보 리스트 (page_idx, bbox 포함)
        middle_json_blocks: middle_json에서 추출된 블록 타입 정보 리스트 [{"block_type": "title"}, {"block_type": "text"}, ...]
    
    Returns:
        tuple: (processed_content_list, processed_block_info_list)
    """
    
    # middle_json_blocks가 있으면 block_type 정보를 사용
    use_middle_json = middle_json_blocks is not None and len(middle_json_blocks) == len(content_list)
    
    if use_middle_json:
        logger.info(f"[Parser-INFO] custom_merge_and_split에서 middle_json block_type 사용: {len(middle_json_blocks)} blocks")
    else:
        logger.info(f"[Parser-INFO] custom_merge_and_split에서 middle_json block_type 미사용")
    import re
    
    CONTEXT_LENGTH = 8192
    
    # 1단계: context length보다 큰 블록 먼저 분리
    logger.info("[Parser-INFO] 1단계: context length 초과 블록 분리 시작")
    split_content_list = []
    split_block_info_list = []
    split_middle_json_blocks = []  # middle_json_blocks도 함께 관리
    split_indices = set()  # 분리된 블록의 인덱스 저장 (2, 3단계 제외용)
    
    for i, block in enumerate(content_list):
        # block_type 가져오기
        block_type = middle_json_blocks[i].get("block_type", "").lower() if use_middle_json and i < len(middle_json_blocks) else ""
        
        # '【설명】'으로 시작하는 title 블록을 text로 변환
        text = block.get("text", "")
        is_explanation_title = text.strip().startswith("【설명】")
        
        if is_explanation_title:
            # title → text 변환
            if use_middle_json and i < len(middle_json_blocks):
                logger.info(f"[Parser-INFO] '【설명】' title 블록 {i}를 text로 변환: '{text[:50]}...'")
                block_type = "text"
                # middle_json_blocks 업데이트
                original_block = middle_json_blocks[i].copy()
                original_block["block_type"] = "text"
                split_middle_json_blocks.append(original_block)
            # block 타입도 변환
            if block.get("type") == "title":
                block = block.copy()
                block["type"] = "text"
            split_content_list.append(block)
            if i < len(block_info_list):
                split_block_info_list.append(block_info_list[i])
            continue
        
        # text 타입이 아니면 그대로 추가
        if use_middle_json:
            is_text_block = (block_type == "text")
        else:
            is_text_block = (block.get("type") == "text")
        
        if not is_text_block:
            split_content_list.append(block)
            if i < len(block_info_list):
                split_block_info_list.append(block_info_list[i])
            if use_middle_json and i < len(middle_json_blocks):
                split_middle_json_blocks.append(middle_json_blocks[i])
            continue
        
        # text는 이미 위에서 선언됨
        text_length = len(text)
        
        if text_length > CONTEXT_LENGTH:
            logger.info(f"[Parser-INFO] 블록 {i} 분리 필요: 길이 {text_length} > {CONTEXT_LENGTH}")
            # 블록 분리
            num_splits = (text_length + CONTEXT_LENGTH - 1) // CONTEXT_LENGTH
            chunk_size = text_length // num_splits
            
            original_bbox = block_info_list[i]["bbox"] if i < len(block_info_list) else [0, 0, 0, 0]
            original_page_idx = block_info_list[i]["page_idx"] if i < len(block_info_list) else 0
            original_middle_block = middle_json_blocks[i] if use_middle_json and i < len(middle_json_blocks) else None
            
            for j in range(num_splits):
                start = j * chunk_size
                end = start + chunk_size if j < num_splits - 1 else text_length
                
                split_block = {
                    "type": "text",
                    "text": text[start:end]
                }
                split_content_list.append(split_block)
                split_block_info_list.append({
                    "page_idx": original_page_idx,
                    "bbox": original_bbox.copy()
                })
                if original_middle_block:
                    split_middle_json_blocks.append(original_middle_block.copy())
                split_indices.add(len(split_content_list) - 1)
            
            logger.info(f"[Parser-INFO] 블록 {i}를 {num_splits}개로 분리 완료")
        else:
            split_content_list.append(block)
            if i < len(block_info_list):
                split_block_info_list.append(block_info_list[i])
            if use_middle_json and i < len(middle_json_blocks):
                split_middle_json_blocks.append(middle_json_blocks[i])
    
    logger.info(f"[Parser-INFO] 1단계 완료: {len(content_list)} -> {len(split_content_list)} 블록")
    
    # 2단계: '제n조' 패턴 찾기 및 병합 ('제 n 조', '제n조', '제n조-n', '제n조의n' 모두 매칭)
    logger.info("[Parser-INFO] 2단계: 조항별 병합 시작")
    # 제1조, 제1조-2, 제1조의2, 제 1 조, 제 1 조 - 2, 제 1 조 의 2 등 모두 매칭
    article_pattern = re.compile(r'^제\s*(\d+)\s*조(?:\s*-\s*\d+|\s*의\s*\d+)?')
    # 숫자 번호 패턴: 
    # - 점 구분: 2.3, 2.3., 2.3.6, 2.3.6. (1개 이상의 점)
    # - 하이픈 구분: 2-3, 2-3-6, 2-3-6-1 (1개 이상의 하이픈)
    number_pattern = re.compile(r'^(\d+(?:[.-]\d+)+)\.?\s')
    chapter_pattern = re.compile(r'^제\s*(\d+)\s*장')  # '제 n 장' 패턴 추가
    
    merged_content_list = []
    merged_block_info_list = []
    merged_middle_json_blocks = []  # middle_json_blocks도 함께 관리
    
    i = 0
    while i < len(split_content_list):
        block = split_content_list[i]
        
        # block_type 가져오기
        block_type = split_middle_json_blocks[i].get("block_type", "").lower() if use_middle_json and i < len(split_middle_json_blocks) else ""
        
        # text 타입 확인
        if use_middle_json:
            is_text_block = (block_type == "text")
            is_title_block = (block_type == "title")
        else:
            is_text_block = (block.get("type") == "text")
            is_title_block = (block.get("type") == "title")
        
        text = block.get("text", "")
        match = article_pattern.match(text.strip())
        number_match = number_pattern.match(text.strip())
        
        # '제n조' 또는 숫자 번호 패턴 체크 (text 또는 title 타입)
        if (match or number_match) and (is_text_block or is_title_block):
            # 패턴 발견 - text 또는 title 타입
            if match:
                article_num = int(match.group(1))
                current_article_id = match.group(0)  # 전체 매칭 텍스트 저장
                logger.info(f"[Parser-INFO] 제{article_num}조 시작 (블록 인덱스: {i}, 타입: {block_type if use_middle_json else block.get('type')})")
            else:  # number_match
                current_article_id = number_match.group(1)  # 숫자 번호 (예: "2.3.6")
                logger.info(f"[Parser-INFO] 번호 '{current_article_id}' 시작 (블록 인덱스: {i}, 타입: {block_type if use_middle_json else block.get('type')})")
            
            # 병합 로직으로 진행
            pass  # 아래 병합 코드로 계속 진행
        elif i in split_indices:
            # 분리된 블록이면 그대로 추가
            merged_content_list.append(block)
            if i < len(split_block_info_list):
                merged_block_info_list.append(split_block_info_list[i])
            if use_middle_json and i < len(split_middle_json_blocks):
                merged_middle_json_blocks.append(split_middle_json_blocks[i])
            i += 1
            continue
        elif not is_text_block and not is_title_block:
            # text, title 타입이 아니면 그대로 추가
            merged_content_list.append(block)
            if i < len(split_block_info_list):
                merged_block_info_list.append(split_block_info_list[i])
            if use_middle_json and i < len(split_middle_json_blocks):
                merged_middle_json_blocks.append(split_middle_json_blocks[i])
            i += 1
            continue
        elif not match and not number_match:
            # '제n조' 또는 숫자 번호 패턴이 아니면 그대로 추가
            merged_content_list.append(block)
            if i < len(split_block_info_list):
                merged_block_info_list.append(split_block_info_list[i])
            if use_middle_json and i < len(split_middle_json_blocks):
                merged_middle_json_blocks.append(split_middle_json_blocks[i])
            i += 1
            continue
        
        # 패턴 발견 - 병합 시작
        if match:
            article_num = int(match.group(1))
            current_article_id = match.group(0)  # 전체 매칭 텍스트 저장 (예: "제1조", "제1조-2", "제1조의2")
            logger.info(f"[Parser-INFO] {current_article_id} 병합 시작 (블록 인덱스: {i}, 타입: {block_type if use_middle_json else block.get('type')})")
        else:  # number_match
            article_num = None  # 숫자 번호는 article_num 없음
            current_article_id = number_match.group(1)  # 숫자 번호 (예: "2.3.6")
            logger.info(f"[Parser-INFO] 번호 '{current_article_id}' 병합 시작 (블록 인덱스: {i}, 타입: {block_type if use_middle_json else block.get('type')})")
        
        # 현재 블록 정보
        merged_text = text
        merged_bbox = split_block_info_list[i]["bbox"].copy() if i < len(split_block_info_list) else [0, 0, 0, 0]
        merged_page_idx = split_block_info_list[i]["page_idx"] if i < len(split_block_info_list) else 0
        
        # bbox 업데이트용 변수
        min_x1 = merged_bbox[0]
        min_y1 = merged_bbox[1]
        max_x2 = merged_bbox[2]
        max_y2 = merged_bbox[3]
        
        # 다음 블록들 확인
        j = i + 1
        current_chunk_blocks = [i]  # 현재 청크에 포함된 블록 인덱스들
        
        while j < len(split_content_list):
            next_block = split_content_list[j]
            
            # 분리된 블록이면 중단
            if j in split_indices:
                break
            
            # block_type 가져오기
            next_block_type = split_middle_json_blocks[j].get("block_type", "").lower() if use_middle_json and j < len(split_middle_json_blocks) else ""
            
            # 타입 확인
            if use_middle_json:
                next_is_title_block = (next_block_type == "title")
                next_is_text_block = (next_block_type == "text")
            else:
                next_is_title_block = (next_block.get("type") == "title")
                next_is_text_block = (next_block.get("type") == "text")
            
            next_text = next_block.get("text", "")
            
            # 다음 블록이 title 타입이면 헤더 여부 확인
            if next_is_title_block:
                # title 내용을 로그로 표시 (최대 100자)
                title_preview = next_text[:100] if len(next_text) > 100 else next_text
                
                # bbox 정보와 페이지 정보로 페이지 헤더인지 판단
                next_bbox = split_block_info_list[j].get("bbox", None) if j < len(split_block_info_list) else None
                next_page_idx = split_block_info_list[j].get("page_idx", -1) if j < len(split_block_info_list) else -1
                
                # 디버깅: bbox 정보 로그 출력
                logger.info(f"[Parser-DEBUG] Title 블록 {j} 검사: bbox={next_bbox}, page_idx={next_page_idx}, "
                            f"block_info_exists={j < len(split_block_info_list)}")
                
                # bbox 정보가 없으면 일반 title로 처리
                if not next_bbox:
                    logger.info(f"[Parser-INFO] 블록 {j}가 title 타입 (bbox 없음), 병합 중단. Title 내용: '{title_preview}'")
                    break
                
                # 해당 페이지에서 몇 번째 블록인지 계산
                block_idx_in_page = 0
                if next_page_idx >= 0:
                    for k in range(j):
                        if k < len(split_block_info_list):
                            prev_page_idx = split_block_info_list[k].get("page_idx", -1)
                            if prev_page_idx == next_page_idx:
                                block_idx_in_page += 1
                
                logger.info(f"[Parser-DEBUG] Title 블록 {j} 위치: page={next_page_idx}, block_idx_in_page={block_idx_in_page}")
                
                if is_page_header(next_bbox, next_page_idx, block_idx_in_page):
                    # 페이지 헤더로 판단되면 병합 계속 (헤더 블록은 무시)
                    logger.info(f"[Parser-INFO] 블록 {j}가 페이지 헤더로 판단됨, 병합 계속. "
                                f"Page: {next_page_idx}, BlockIdx: {block_idx_in_page}, "
                                f"Title: '{title_preview}', bbox={next_bbox}")
                    j += 1
                    continue  # 다음 블록으로 넘어감 (헤더는 병합하지 않음)
                else:
                    # 일반 title 블록이면 병합 중단 (새로운 섹션 시작)
                    logger.info(f"[Parser-INFO] 블록 {j}가 title 타입, 병합 중단. Title 내용: '{title_preview}'")
                    break
            
            # text 타입이 아니면 중단
            if not next_is_text_block:
                break
            
            next_match = article_pattern.match(next_text.strip())
            next_number_match = number_pattern.match(next_text.strip())
            
            # '제 n 장' 패턴이면 중단
            chapter_match = chapter_pattern.match(next_text.strip())
            if chapter_match:
                chapter_num = int(chapter_match.group(1))
                logger.info(f"[Parser-INFO] 제{chapter_num}장 발견, 병합 중단")
                break
            
            # text 블록에서 다음 조항/번호 발견하면 중단
            if next_match:
                # '제n조' 패턴 체크
                next_article_id = next_match.group(0)  # 전체 매칭 텍스트
                next_article_num = int(next_match.group(1))  # 기본 조 번호
                
                # 다른 조항이면 중단
                if next_article_id != current_article_id:
                    # 조 번호가 더 크거나, 같은 번호의 다른 조항(제1조 vs 제1조-2)
                    if article_num is not None and (next_article_num > article_num or (next_article_num == article_num and next_article_id != current_article_id)):
                        logger.info(f"[Parser-INFO] {next_article_id} 발견, 병합 중단")
                        break
                    # 현재가 숫자 번호 패턴인데 '제n조' 패턴이 나오면 중단
                    elif article_num is None:
                        logger.info(f"[Parser-INFO] {next_article_id} 발견, 병합 중단")
                        break
            
            elif next_number_match:
                # 숫자 번호 패턴 체크 (예: "2.3.7", "2.3.6.1")
                next_number_id = next_number_match.group(1)  # 숫자 번호 (예: "2.3.7")
                
                # 다른 번호이면 중단
                if next_number_id != current_article_id:
                    logger.info(f"[Parser-INFO] 번호 '{next_number_id}' 발견, 병합 중단")
                    break
            
            # 3단계: 병합 시 context length 체크
            tentative_text = merged_text + "\n" + next_text
            if len(tentative_text) > CONTEXT_LENGTH:
                logger.info(f"[Parser-INFO] 블록 {j} 병합 시 길이 초과 ({len(tentative_text)} > {CONTEXT_LENGTH}), 현재까지 병합")
                # 현재까지 병합한 것을 저장
                merged_content_list.append({
                    "type": "text",
                    "text": merged_text
                })
                
                # bbox는 첫 페이지 범위 내로 제한
                if j < len(split_block_info_list):
                    next_page_idx = split_block_info_list[j]["page_idx"]
                    if next_page_idx == merged_page_idx:
                        next_bbox = split_block_info_list[j]["bbox"]
                        max_x2 = max(max_x2, next_bbox[2])
                        max_y2 = max(max_y2, next_bbox[3])
                
                merged_block_info_list.append({
                    "page_idx": merged_page_idx,
                    "bbox": [min_x1, min_y1, max_x2, max_y2]
                })
                
                # middle_json_blocks도 추가 (패턴 병합된 블록은 text 타입으로 변경)
                if use_middle_json and i < len(split_middle_json_blocks):
                    merged_middle_json_blocks.append({"block_type": "text"})
                
                # 새로운 청크 시작
                merged_text = next_text
                merged_page_idx = split_block_info_list[j]["page_idx"] if j < len(split_block_info_list) else 0
                next_bbox = split_block_info_list[j]["bbox"] if j < len(split_block_info_list) else [0, 0, 0, 0]
                min_x1 = next_bbox[0]
                min_y1 = next_bbox[1]
                max_x2 = next_bbox[2]
                max_y2 = next_bbox[3]
                current_chunk_blocks = [j]
            else:
                # 병합 가능
                merged_text = tentative_text
                current_chunk_blocks.append(j)
                
                # bbox 업데이트 (같은 페이지인 경우만)
                if j < len(split_block_info_list):
                    next_page_idx = split_block_info_list[j]["page_idx"]
                    if next_page_idx == merged_page_idx:
                        next_bbox = split_block_info_list[j]["bbox"]
                        min_x1 = min(min_x1, next_bbox[0])
                        min_y1 = min(min_y1, next_bbox[1])
                        max_x2 = max(max_x2, next_bbox[2])
                        max_y2 = max(max_y2, next_bbox[3])
            
            j += 1
        
        # 마지막 병합된 블록 추가
        merged_content_list.append({
            "type": "text",
            "text": merged_text
        })
        merged_block_info_list.append({
            "page_idx": merged_page_idx,
            "bbox": [min_x1, min_y1, max_x2, max_y2]
        })
        
        # middle_json_blocks도 추가 (병합된 블록은 text 타입으로 변경)
        if use_middle_json and i < len(split_middle_json_blocks):
            # 패턴 병합된 블록은 text로 처리
            original_type = split_middle_json_blocks[i].get("block_type", "")
            merged_middle_json_blocks.append({"block_type": "text"})
            logger.info(f"[Parser-INFO] 병합 블록 타입 변환: {original_type} → text")
        
        logger.info(f"[Parser-INFO] {current_article_id} 병합 완료: 블록 {i}~{j-1} ({len(current_chunk_blocks)}개 블록)")
        
        # 다음 처리할 블록으로 이동
        i = j
    
    logger.info(f"[Parser-INFO] 2단계 완료: {len(split_content_list)} -> {len(merged_content_list)} 블록")
    logger.info(f"[Parser-INFO] ✅ custom_merge_and_split 완료: 최종 블록 수 {len(merged_content_list)}")
    
    return merged_content_list, merged_block_info_list, merged_middle_json_blocks


def map_titles_to_text_blocks(content_list, middle_json_blocks=None):
    """
    Title 블록을 Text 블록에 매핑하는 함수
    
    각 text 블록 앞에 나온 마지막 title 블록을 찾아서 매핑합니다.
    
    Args:
        content_list: 블록 리스트 (custom_merge_and_split의 결과)
        middle_json_blocks: middle_json에서 추출된 블록 타입 정보 리스트
    
    Returns:
        dict: {text_block_index: "title_text"} 형태의 매핑
    """
    use_middle_json = middle_json_blocks is not None and len(middle_json_blocks) == len(content_list)
    
    title_mapping = {}
    current_title = ""  # 현재 섹션의 title 텍스트
    title_stack = []  # 계층 구조를 위한 title 스택 (선택사항)
    
    logger.info(f"[Parser-INFO] Title-to-Text 매핑 시작: {len(content_list)} 블록, use_middle_json={use_middle_json}")
    if middle_json_blocks:
        logger.info(f"[Parser-INFO] middle_json_blocks 길이: {len(middle_json_blocks)}")
    
    for i, block in enumerate(content_list):
        # block_type 확인
        if use_middle_json and i < len(middle_json_blocks):
            block_type = middle_json_blocks[i].get("block_type", "").lower()
            is_title_block = (block_type == "title")
            is_text_block = (block_type == "text")
            logger.debug(f"[Parser-DEBUG] 블록 {i}: middle_json block_type='{block_type}'")
        else:
            block_type = block.get("type", "")
            is_title_block = (block_type == "title")
            is_text_block = (block_type == "text")
            logger.debug(f"[Parser-DEBUG] 블록 {i}: content_list block type='{block_type}'")
        
        text = block.get("text", "").strip()
        
        if is_title_block and text:
            # Title 블록 발견 - 최대 150자로 제한
            current_title = text[:150]
            logger.info(f"[Parser-INFO] Title 블록 {i} 발견: '{current_title[:50]}...'")
        
        elif is_text_block:
            # Text 블록에 현재 title 할당
            if current_title:
                title_mapping[i] = current_title
                logger.info(f"[Parser-INFO] Text 블록 {i}에 title 할당: '{current_title[:30]}...'")
            else:
                # title이 없는 경우 빈 문자열
                title_mapping[i] = ""
                logger.debug(f"[Parser-DEBUG] Text 블록 {i}에 할당할 title 없음")
    
    logger.info(f"[Parser-INFO] Title-to-Text 매핑 완료: {len(title_mapping)} 개 text 블록에 title 할당")
    
    return title_mapping


def find_chunk_to_image(img_info, chunk_ids_list, chunk_metadata):
    """
    이미지에 가장 가까운 청크를 찾는 함수
    
    같은 페이지 내에서 상하좌우 관계를 고려하여 가장 가까운 청크 ID를 반환합니다.
    
    Args:
        img_info: 이미지 정보 {"url": str, "page_idx": int, "bbox": [x1, y1, x2, y2]}
        chunk_ids_list: 청크 ID 리스트
        chunk_metadata: 청크 메타데이터 딕셔너리 {chunk_id: {"page_idx": int, "bbox": [x1, y1, x2, y2]}}
    
    Returns:
        tuple: (nearest_chunk_id, match_type, distance) 또는 (None, None, None)
            - nearest_chunk_id: 가장 가까운 청크 ID (str or None)
            - match_type: 매칭 유형 "위쪽"|"좌측"|"우측" (str or None)
            - distance: 거리 (float or None)
    """
    img_page = img_info["page_idx"]
    img_bbox = img_info["bbox"]  # [x1, y1, x2, y2]
    
    if len(img_bbox) != 4:
        return None, None, None
    
    # 이미지의 좌표
    img_x1, img_y1, img_x2, img_y2 = img_bbox
    
    nearest_chunk_id = None
    min_distance = float('inf')
    match_type = None  # 매칭 유형 (위쪽, 좌측, 우측)
    
    # 같은 페이지의 모든 chunk와 비교
    for chunk_id in chunk_ids_list:
        chunk_meta = chunk_metadata.get(chunk_id)
        if not chunk_meta:
            continue
            
        chunk_page = chunk_meta["page_idx"]
        chunk_bbox = chunk_meta["bbox"]  # [x1, y1, x2, y2]
        
        # 같은 페이지가 아니면 건너뜀
        if chunk_page != img_page:
            continue
        
        if len(chunk_bbox) != 4:
            continue
        
        # chunk의 좌표
        chunk_x1, chunk_y1, chunk_x2, chunk_y2 = chunk_bbox
        
        distance = None
        current_match_type = None
        
        # 1. 위쪽 관계: chunk가 이미지 위쪽에 있는 경우 (chunk 하단 < 이미지 상단)
        if chunk_y2 < img_y1:
            distance = img_y1 - chunk_y2
            current_match_type = "위쪽"
        
        # 2. 좌측 관계: 이미지가 chunk 우측에 있는 경우 (이미지 좌측 > chunk 우측)
        elif img_x1 > chunk_x2:
            # Y축 겹침이 있는지 확인 (같은 라인에 있는지)
            y_overlap = not (img_y2 < chunk_y1 or img_y1 > chunk_y2)
            if y_overlap:
                distance = img_x1 - chunk_x2
                current_match_type = "좌측"
        
        # 3. 우측 관계: 이미지가 chunk 좌측에 있는 경우 (이미지 우측 < chunk 좌측)
        elif img_x2 < chunk_x1:
            # Y축 겹침이 있는지 확인 (같은 라인에 있는지)
            y_overlap = not (img_y2 < chunk_y1 or img_y1 > chunk_y2)
            if y_overlap:
                distance = chunk_x1 - img_x2
                current_match_type = "우측"
        
        # 가장 가까운 거리 업데이트
        if distance is not None and distance < min_distance:
            min_distance = distance
            nearest_chunk_id = chunk_id
            match_type = current_match_type
    
    # 결과 반환
    if nearest_chunk_id:
        return nearest_chunk_id, match_type, min_distance
    else:
        return None, None, None


@contextmanager
def capture_stdout_stderr(doc_id):
    """표준 출력과 표준 에러를 캡처하여 실시간으로 데이터베이스에 업데이트합니다."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # 문자열 버퍼 생성
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    # 사용자 정의 출력 클래스로 실시간 캡처 및 진행률 업데이트
    class ProgressCapture:
        def __init__(self, original, buffer, doc_id):
            self.original = original
            self.buffer = buffer
            self.doc_id = doc_id
            self.last_update = time.time()
            # 표준 출력 스트림과 호환되도록 필요한 속성 추가
            self.encoding = getattr(original, 'encoding', 'utf-8')
            self.errors = getattr(original, 'errors', 'strict')
            self.mode = getattr(original, 'mode', 'w')
            
        def write(self, text):
            self.original.write(text)  # 기존 출력을 유지
            self.buffer.write(text)
            
            # 진행 정보가 포함되어 있는지 확인
            if any(keyword in text for keyword in ['Predict:', '%|', 'Processing pages:', 'OCR-', 'MFD', 'MFR', 'Table', 'it/s]', 'INFO']):
                # 텍스트 정리, ANSI 이스케이프 시퀀스 및 불필요한 공백 제거
                clean_text = re.sub(r'\x1b\[[0-9;]*m', '', text.strip())
                clean_text = re.sub(r'\s+', ' ', clean_text)  # 여러 공백을 하나로 합침
                
                if clean_text and len(clean_text) > 5:  # 너무 짧은 텍스트는 필터링
                    current_time = time.time()
                    # 업데이트 빈도 제한, 너무 자주 DB 작업 방지
                    if current_time - self.last_update > 0.3:  # 0.3초마다 한 번만 업데이트
                        try:
                            # 주요 정보 추출, 우선적으로 진행률 정보 표시
                            if '%|' in clean_text and ('Predict:' in clean_text or 'Processing' in clean_text):
                                # 진행률 정보, 바로 사용
                                _update_document_progress(self.doc_id, message=clean_text[:500])
                            elif 'INFO' in clean_text and any(x in clean_text for x in ['처리', '분석', '추출']):
                                # 중요한 처리 정보
                                _update_document_progress(self.doc_id, message=clean_text[:500])
                            else:
                                # 기타 정보도 업데이트, 우선순위는 낮음
                                _update_document_progress(self.doc_id, message=clean_text[:500])
                            self.last_update = current_time
                        except Exception as e:
                            logger.error(f"[Parser-ERROR] 진행 메시지 업데이트 실패: {e}")
            
        def flush(self):
            self.original.flush()
            
        def __getattr__(self, name):
            # 다른 속성들을 원본 출력 스트림으로 프록시
            return getattr(self.original, name)
    
    try:
        # 표준 출력과 오류 출력을 대체
        sys.stdout = ProgressCapture(old_stdout, stdout_buffer, doc_id)
        sys.stderr = ProgressCapture(old_stderr, stderr_buffer, doc_id)
        yield stdout_buffer, stderr_buffer
    finally:
        # 원본 출력 복원
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def convert_pdf_safely(pdf_path, doc_id):
    """
    PDF를 안전하게 변환하고 명시적 메모리 관리를 수행합니다.
    
    Args:
        pdf_path (str): PDF 파일 경로
        doc_id (str): 문서 ID (로깅용)
    
    Returns:
        bytes: 변환된 PDF 바이트
    """
    pdf_bytes_original = None
    pdf_bytes_converted = None
    
    try:
        # 파일 읽기
        with open(pdf_path, "rb") as f:
            pdf_bytes_original = f.read()
        
        logger.info(f"[Parser-INFO] PDF 변환 시작 (Doc ID: {doc_id}, Size: {len(pdf_bytes_original)} bytes)")
        
        # PyPDFium2로 변환
        pdf_bytes_converted = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes_original)
        
        logger.info(f"[Parser-INFO] PDF 변환 완료 (Doc ID: {doc_id})")
        
        # 원본 바이트 즉시 해제
        del pdf_bytes_original
        gc.collect()
        
        return pdf_bytes_converted
        
    except Exception as e:
        logger.error(f"[Parser-ERROR] PDF 변환 실패 (Doc ID: {doc_id}): {e}")
        # 메모리 정리
        if pdf_bytes_original is not None:
            del pdf_bytes_original
        if pdf_bytes_converted is not None:
            del pdf_bytes_converted
        gc.collect()
        raise


def perform_parse(doc_id, doc_info, file_info, embedding_config, kb_info):
    """
    문서 파싱의 핵심 로직을 수행합니다.

    Args:
        doc_id (str): 문서 ID.
        doc_info (dict): 문서 정보가 담긴 딕셔너리 (name, location, type, kb_id, parser_config, created_by).
        file_info (dict): 파일 정보가 담긴 딕셔너리 (parent_id/bucket_name).
        kb_info (dict): 지식베이스 정보가 담긴 딕셔너리 (created_by).

    Returns:
        dict: 파싱 결과가 담긴 딕셔너리 (success, chunk_count).
    """
    temp_pdf_path = None
    middle_json_data = None  # middle_json 데이터 초기화
    temp_image_dir = None
    start_time = time.time()

    middle_json_content = None  # 중간 JSON 내용 초기화
    image_info_list = []  # 이미지 정보 리스트

    # 기본값 처리
    embedding_model_name = embedding_config.get("llm_name") if embedding_config and embedding_config.get("llm_name") else "bge-m3"  # 기본 모델
    
    # 모델명 처리
    if embedding_model_name:
        # "___" 접미사 제거 (예: model___suffix -> model)
        if "___" in embedding_model_name:
            embedding_model_name = embedding_model_name.split("___")[0]
        
        # "@" 접미사 제거 (예: bona/bge-m3-korean:latest@Ollama -> bona/bge-m3-korean:latest)
        if "@" in embedding_model_name:
            embedding_model_name = embedding_model_name.split("@")[0]
            logger.info(f"[Parser-INFO] 모델명에서 '@' 접미사 제거: {embedding_model_name}")

    # 실리콘플로우 플랫폼의 특수 처리를 제거하고 원래 모델명을 유지
    # 아래 코드는 주석 처리하여 사용자가 설정한 실제 모델을 사용하도록 함
    # if embedding_model_name == "netease-youdao/bce-embedding-base_v1":
    #     embedding_model_name = "BAAI/bge-m3"

    embedding_api_base = embedding_config.get("api_base") if embedding_config and embedding_config.get("api_base") else "http://localhost:11434"  # 기본 API URL

    # API 기본 주소가 빈 문자열이면 실리콘플로우 API 주소로 설정
    if embedding_api_base == "":
        embedding_api_base = "https://api.siliconflow.cn/v1/embeddings"
        logger.info(f"[Parser-INFO] API 기본 주소가 비어 있어 실리콘플로우 API 주소로 설정됨: {embedding_api_base}")

    embedding_api_key = embedding_config.get("api_key") if embedding_config else None  # None 또는 빈 문자열일 수 있음

    # Embedding API URL 완성
    embedding_url = None  # 기본값 None
    if embedding_api_base:
        # embedding_api_base에 프로토콜이 포함되어 있는지 확인 (http:// 또는 https://)
        if not embedding_api_base.startswith(("http://", "https://")):
            embedding_api_base = "http://" + embedding_api_base

        # 끝의 슬래시 제거
        normalized_base_url = embedding_api_base.rstrip("/")

        # 요청 URL에 11434 포트가 있으면 ollama 모델로 간주, ollama 전용 API 사용
        is_ollama = "11434" in normalized_base_url
        if is_ollama:
            # Ollama 전용 엔드포인트
            embedding_url = normalized_base_url + "/api/embeddings"
        elif normalized_base_url.endswith("/v1"):
            embedding_url = normalized_base_url + "/embeddings"
        elif normalized_base_url.endswith("/embeddings"):
            embedding_url = normalized_base_url
        else:
            embedding_url = normalized_base_url + "/v1/embeddings"

    logger.info(f"[Parser-INFO] Embedding 설정 사용: URL='{embedding_url}', Model='{embedding_model_name}', Key={embedding_api_key}")

    try:
        kb_id = doc_info["kb_id"]
        file_location = doc_info["location"]
        # 파일 경로에서 원래 확장자 추출
        _, file_extension = os.path.splitext(file_location)
        file_type = doc_info["type"].lower()
        
        # 실제 파일이 저장된 버킷을 찾기 위해 parent_id와 kb_id 모두 확인
        parent_id = file_info["parent_id"]
        tenant_id = kb_info["created_by"]  # 지식베이스 생성자를 tenant_id로 사용

        # 진행 상황 업데이트 콜백 (내부 업데이트 함수 직접 호출)
        def update_progress(prog=None, msg=None):
            _update_document_progress(doc_id, progress=prog, message=msg)
            logger.info(f"[Parser-PROGRESS] Doc: {doc_id}, Progress: {prog}, Message: {msg}")


        # 1. MinIO에서 파일 내용 가져오기
        minio_client = get_minio_client()
        
        # 실제 파일이 저장된 버킷 찾기 (parent_id, kb_id 순서로 확인)
        bucket_name = None
        if minio_client.bucket_exists(parent_id):
            bucket_name = parent_id
            logger.info(f"[Parser-INFO] parent_id 버킷 사용: {bucket_name}")
        elif minio_client.bucket_exists(kb_id):
            bucket_name = kb_id
            logger.info(f"[Parser-INFO] kb_id 버킷 사용: {bucket_name}")
        else:
            raise Exception(f"저장소 버킷이 존재하지 않습니다: parent_id={parent_id}, kb_id={kb_id}")

        update_progress(0.1, f"저장소에서 파일을 가져오는 중: {file_location}")
        response = minio_client.get_object(bucket_name, file_location)
        file_content = response.read()
        response.close()
        update_progress(0.2, "파일 가져오기 성공, 파싱 준비 중")


        # 2. 파일 유형에 따라 파서 선택
        content_list = []
        if file_type.endswith("pdf"):
            update_progress(0.3, "MinerU 파서 사용")

            # 임시 파일에 PDF 내용 저장
            temp_dir = tempfile.gettempdir()
            temp_pdf_path = os.path.join(temp_dir, f"{doc_id}.pdf")
            with open(temp_pdf_path, "wb") as f:
                f.write(file_content)

            # file_content 즉시 해제 (대용량 메모리 확보)
            del file_content
            gc.collect()

            # PDF 바이트를 mineru 호환 형식으로 안전하게 변환
            pdf_bytes = convert_pdf_safely(temp_pdf_path, doc_id)
            
            # 임시 출력 디렉토리 설정
            temp_image_dir = os.path.join(temp_dir, f"images_{doc_id}")
            os.makedirs(temp_image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(temp_image_dir)

            # MinerU로 처리, 상세 출력 캡처
            with capture_stdout_stderr(doc_id):
                # 언어 및 파싱 방법 설정 (기본값 사용)
                lang = "korean"  # 한국어 기본값, 필요에 따라 변경
                parse_method = "auto"  # 자동 감지, 필요에 따라 "txt" 또는 "ocr"로 변경 가능
        
                update_progress(0.4, f"{parse_method}로 PDF 처리 중, 구체적인 진행 상황은 컨테이너 로그 참조")
                # pipeline 백엔드로 문서 분석 수행
                # pipeline_doc_analyze는 여러 문서를 일괄 처리할 수 있으므로 리스트로 전달
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    [pdf_bytes],  # 문서 바이트 리스트
                    [lang],       # 언어 리스트
                    parse_method=parse_method,
                    formula_enable=False,
                    table_enable=True
                )
            
                update_progress(0.6, f"{parse_method} 결과 처리 중")
                # 첫 번째(유일한) 문서의 결과 가져오기
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]

                # 중간 JSON 생성
                middle_json = pipeline_result_to_middle_json(
                    model_list, 
                    images_list, 
                    pdf_doc, 
                    image_writer, 
                    _lang, 
                    _ocr_enable, 
                )
                middle_json_data = middle_json  # 병합 함수에서 사용할 데이터 할당
            
                update_progress(0.8, "내용 추출 중")
                # PDF 정보 접근
                pdf_info = middle_json["pdf_info"]
                # content_list 생성
                image_dir = os.path.basename(temp_image_dir)
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                # table 블록을 마크다운으로 변환
                for chunk in content_list:
                    if chunk.get("type", "").lower() == "table":
                        # table 블록의 구조 로그 (디버깅용)
                        has_text = "text" in chunk
                        has_table_body = "table_body" in chunk
                        text_has_html = "<table" in chunk.get("text", "")
                        table_body_has_html = "<table" in chunk.get("table_body", "")
                        logger.info(f"[Parser-DEBUG] Table 블록 구조: has_text={has_text}, has_table_body={has_table_body}, "
                                    f"text_has_html={text_has_html}, table_body_has_html={table_body_has_html}")
                        
                        # text 필드에 HTML table이 있으면 마크다운 변환
                        if text_has_html:
                            logger.info(f"[Parser-INFO] Table의 text 필드를 마크다운으로 변환")
                            chunk["text"] = html_table_to_markdown(chunk["text"])
                        
                        # table_body 필드에 HTML table이 있으면 마크다운 변환
                        if table_body_has_html:
                            logger.info(f"[Parser-INFO] Table의 table_body 필드를 마크다운으로 변환")
                            chunk["table_body"] = html_table_to_markdown(chunk["table_body"])
                # 중간 JSON 문자열 직접 가져오기
                middle_json_content = middle_json
                # 로깅
                logger.info(f"[Parser-INFO] 문서 처리 완료, 청크 수: {len(content_list)}")
                
                # 첫 몇 개 블록의 타입과 내용 샘플 로그
                for idx, chunk in enumerate(content_list[:10]):  # 처음 10개만 로그
                    chunk_type = chunk.get("type", "unknown")
                    chunk_text = chunk.get("text", "")[:100] if chunk.get("text") else ""
                    logger.info(f"[Parser-INFO] Block[{idx}]: type='{chunk_type}', text='{chunk_text}...'")
            
            # MinerU 처리 완료 후 대용량 객체 명시적 해제
            logger.info(f"[Parser-INFO] 메모리 정리 시작 (Doc ID: {doc_id})")
            try:
                if 'pdf_bytes' in locals():
                    del pdf_bytes
                if 'infer_results' in locals():
                    del infer_results
                if 'all_image_lists' in locals():
                    del all_image_lists
                if 'all_pdf_docs' in locals():
                    del all_pdf_docs
                if 'model_list' in locals():
                    del model_list
                if 'images_list' in locals():
                    del images_list
                if 'pdf_doc' in locals():
                    del pdf_doc
                # 가비지 컬렉션 강제 실행
                gc.collect()
                logger.info(f"[Parser-INFO] 메모리 정리 완료 (Doc ID: {doc_id})")
            except Exception as mem_clean_err:
                logger.warning(f"[Parser-WARNING] 메모리 정리 중 경고 (Doc ID: {doc_id}): {mem_clean_err}")
        
        elif file_type.endswith("word") or file_type.endswith("ppt") or file_type.endswith("txt") or file_type.endswith("md") or file_type.endswith("html"):
            update_progress(0.3, f"지원하지 않는 파일 유형: {file_type}")
            raise NotImplementedError(f"파일 유형 '{file_type}'에 대한 파서가 아직 구현되지 않았습니다. MinerU 2.1.0부터는 따로 PDF로 변환 후 처리필요")
        # 엑셀 파일은 별도로 처리
        elif file_type.endswith("excel"):
            update_progress(0.3, "MinerU 파서 사용")
            # 임시 파일에 내용 저장
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"{doc_id}{file_extension}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            logger.info(f"[Parser-INFO] 임시 파일 경로: {temp_file_path}")

            update_progress(0.8, "내용 추출 중")
            # 내용 리스트 처리
            content_list = parse_excel_file(temp_file_path)

        elif file_type.endswith("visual"):
            update_progress(0.3, "MinerU 파서 사용")

            # 임시 파일에 내용 저장
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"{doc_id}{file_extension}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"[Parser-INFO] 임시 파일 경로: {temp_file_path}")

            # 이미지 바이트 읽기
            image_bytes = read_fn(temp_file_path)
            
            # 임시 출력 디렉토리 설정
            temp_image_dir = os.path.join(temp_dir, f"images_{doc_id}")
            os.makedirs(temp_image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(temp_image_dir)

            # MinerU로 처리, 상세 출력 캡처
            with capture_stdout_stderr(doc_id):
                # 언어 설정 (기본값)
                lang = "korean"  # 필요에 따라 변경
                
                update_progress(0.4, "이미지 분석 중 (OCR 처리)")
                # pipeline 백엔드로 이미지 분석 - OCR 처리 활성화
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                    [image_bytes],  # 이미지 바이트 리스트
                    [lang],        # 언어 리스트
                    parse_method="ocr",  # 이미지 파일은 OCR 모드 사용
                    formula_enable=True,
                    table_enable=True
                )
                
                update_progress(0.6, "결과 처리 중")
                # 첫 번째(유일한) 이미지의 결과 가져오기
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]
            
                # 중간 JSON 생성
                middle_json = pipeline_result_to_middle_json(
                    model_list, 
                    images_list, 
                    pdf_doc, 
                    image_writer, 
                    _lang, 
                    _ocr_enable
                )
                middle_json_data = middle_json  # 병합 함수에서 사용할 데이터 할당
                
                update_progress(0.8, "내용 추출 중")
                # PDF 정보 접근
                pdf_info = middle_json["pdf_info"]
                # content_list 생성
                image_dir = os.path.basename(temp_image_dir)
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                # 중간 JSON 직접 가져오기
                middle_json_content = middle_json
                
                # 로깅
                logger.info(f"[Parser-INFO] 이미지 처리 완료, 청크 수: {len(content_list)}")
        else:
            update_progress(0.3, f"지원하지 않는 파일 유형: {file_type}")
            raise NotImplementedError(f"파일 유형 '{file_type}'의 파서는 아직 구현되지 않았습니다.")


        # middle_json_content를 파싱하여 블록 정보 추출
        block_info_list = []
        middle_json_blocks = []  # merge_title_text_blocks 함수에서 사용할 블록 리스트
        if middle_json_content:
            if isinstance(middle_json_content, dict):
                middle_data = middle_json_content  # 바로 할당
            else:
                middle_data = None
                logger.warning(f"[Parser-WARNING] middle_json_content가 예상한 딕셔너리 형식이 아닙니다. 실제 타입: {type(middle_json_content)}.")
            try:
                # middle_json의 블록 타입 분석
                middle_block_types = {}
                total_middle_blocks = 0
                
                # 정보 추출
                for page_idx, page_data in enumerate(middle_data.get("pdf_info", [])):
                    page_blocks = page_data.get("preproc_blocks", [])
                    logger.info(f"[Parser-INFO] Page {page_idx}: {len(page_blocks)} blocks in middle_json")
                    
                    for block_idx, block in enumerate(page_blocks):
                        total_middle_blocks += 1
                        block_type = block.get("type", "unknown")
                        middle_block_types[block_type] = middle_block_types.get(block_type, 0) + 1
                        
                        # merge_title_text_blocks 함수에서 사용할 블록 정보 추가
                        middle_json_blocks.append({"block_type": block_type})
                        
                        block_bbox = get_bbox_from_block(block)
                        # 텍스트가 있고 bbox가 있는 블록만 추출
                        if block_bbox != [0, 0, 0, 0]:
                            block_info_list.append({"page_idx": page_idx, "bbox": block_bbox})
                        else:
                            logger.warning("[Parser-WARNING] 블록의 bbox 형식이 유효하지 않아 건너뜀.")

                    logger.info(f"[Parser-INFO] middle_data에서 {len(block_info_list)}개의 블록 정보를 추출함.")
                
                logger.info(f"[Parser-INFO] MiddleJSON 블록 타입 분포 (총 {total_middle_blocks}개): {middle_block_types}")

            except json.JSONDecodeError:
                logger.error("[Parser-ERROR] middle_json_content 파싱 실패.")
                raise Exception("[Parser-ERROR] middle_json_content 파싱 실패.")
            except Exception as e:
                logger.error(f"[Parser-ERROR] middle_json_content 처리 중 오류: {e}")
                raise Exception(f"[Parser-ERROR] middle_json_content 처리 중 오류: {e}")

            # Title과 Text 블록 병합 처리 (SKIP - custom_merge_and_split만 사용)
            if content_list and block_info_list:
                # try:
                #     logger.info(f"[Parser-INFO] Title-Text 블록 병합 시작: {len(content_list)}개 블록")
                #     # middle_json_blocks 사용 (middle_json_content에서 추출된 블록 타입 정보)
                #     middle_blocks_for_merge = middle_json_blocks if 'middle_json_blocks' in locals() and middle_json_blocks else None
                #     content_list, block_info_list = merge_title_text_blocks(content_list, block_info_list, middle_blocks_for_merge)
                #     logger.info(f"[Parser-INFO] Title-Text 블록 병합 완료: {len(content_list)}개 블록")
                # except Exception as e:
                #     logger.warning(f"[Parser-WARNING] Title-Text 블록 병합 중 오류, 원본 사용: {e}")
                
                # 조항별 병합 및 context length 기반 분리 처리
                try:
                    logger.info(f"[Parser-INFO] 조항별 병합 및 분리 시작: {len(content_list)}개 블록")
                    # middle_json_blocks 전달
                    middle_blocks_for_custom = middle_json_blocks if 'middle_json_blocks' in locals() and middle_json_blocks else None
                    content_list, block_info_list, middle_json_blocks = custom_merge_and_split(content_list, block_info_list, middle_blocks_for_custom)
                    logger.info(f"[Parser-INFO] 조항별 병합 및 분리 완료: {len(content_list)}개 블록")
                except Exception as e:
                    logger.warning(f"[Parser-WARNING] 조항별 병합 및 분리 중 오류, 원본 사용: {e}")
                    middle_json_blocks = None  # 에러 시 middle_json_blocks 초기화

                # Title-to-Text 매핑 생성
                try:
                    logger.info(f"[Parser-INFO] Title-to-Text 매핑 시작")
                    middle_blocks_for_mapping = middle_json_blocks if middle_json_blocks else None
                    title_mapping = map_titles_to_text_blocks(content_list, middle_blocks_for_mapping)
                    logger.info(f"[Parser-INFO] Title-to-Text 매핑 완료: {len(title_mapping)}개 매핑")
                except Exception as e:
                    logger.warning(f"[Parser-WARNING] Title-to-Text 매핑 중 오류, 빈 매핑 사용: {e}")
                    title_mapping = {}

        # 3. 파싱 결과 처리 (MinIO 업로드, ES 저장)
        update_progress(0.95, "파싱 결과 저장 중")
        es_client = get_es_client()
        # 참고: MinIO 버킷은 파일의 parent_id가 아니라 kb_id(지식베이스 ID)여야 함
        output_bucket = kb_id
        if not minio_client.bucket_exists(output_bucket):
            minio_client.make_bucket(output_bucket)
            logger.info(f"[Parser-INFO] MinIO 버킷 생성: {output_bucket}")

        # 임베딩 벡터 차원 구하기
        embedding_dim = None
        try:
            # 테스트 텍스트로 벡터 차원 먼저 구하기
            test_content = "test"
            headers = {"Content-Type": "application/json"}
            if embedding_api_key:
                headers["Authorization"] = f"Bearer {embedding_api_key}"

            is_ollama = "11434" in embedding_url if embedding_url else False
            if is_ollama:
                test_resp = requests.post(
                    embedding_url,
                    headers=headers,
                    json={"model": embedding_model_name, "prompt": test_content},
                    timeout=15,
                )
            else:
                test_resp = requests.post(
                    embedding_url,
                    headers=headers,
                    json={"model": embedding_model_name, "input": test_content},
                    timeout=15,
                )
            
            test_resp.raise_for_status()
            test_data = test_resp.json()
            
            if is_ollama:
                test_vec = test_data.get("embedding")
            else:
                test_vec = test_data["data"][0]["embedding"]
            
            embedding_dim = len(test_vec)
            logger.info(f"[Parser-INFO] 임베딩 차원 감지: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"[Parser-ERROR] 임베딩 차원 구하기 실패: {e}")
            raise Exception(f"[Parser-ERROR] 임베딩 차원 구하기 실패: {e}")

        index_name = f"ragflow_{tenant_id}"
        vector_field_name = f"q_{embedding_dim}_vec"
        
        if not es_client.indices.exists(index=index_name):
            # 동적 차원으로 인덱스 생성
            es_client.indices.create(
                index=index_name,
                body={
                    "settings": {"number_of_replicas": 0},
                    "mappings": {
                        "properties": {
                            "doc_id": {"type": "keyword"}, 
                            "kb_id": {"type": "keyword"}, 
                            "content_with_weight": {"type": "text"}, 
                            vector_field_name: {"type": "dense_vector", "dims": embedding_dim}
                        }
                    },
                },
            )
            logger.info(f"[Parser-INFO] Elasticsearch 인덱스 생성: {index_name}, 벡터 차원: {embedding_dim}")
        else:
            # 기존 인덱스에 현재 차원의 벡터 필드가 있는지 확인
            try:
                mapping = es_client.indices.get_mapping(index=index_name)
                existing_properties = mapping[index_name]["mappings"]["properties"]
                
                if vector_field_name not in existing_properties:
                    # 새 벡터 필드 추가
                    es_client.indices.put_mapping(
                        index=index_name,
                        body={
                            "properties": {
                                vector_field_name: {"type": "dense_vector", "dims": embedding_dim}
                            }
                        }
                    )
                    logger.info(f"[Parser-INFO] 인덱스 {index_name}에 새 벡터 필드 추가: {vector_field_name}, 차원: {embedding_dim}")
            except Exception as e:
                logger.error(f"[Parser-ERROR] 인덱스 매핑 업데이트 실패: {e}")
                raise Exception(f"[Parser-ERROR] 인덱스 매핑 업데이트 실패: {e}")

        chunk_count = 0
        chunk_ids_list = []
        chunk_metadata = {}  # chunk_id -> {"page_idx": int, "bbox": [x1, y1, x2, y2]} 매핑


        for chunk_idx, chunk_data in enumerate(content_list):
            page_idx = 0  # 기본 페이지 인덱스
            bbox = [0, 0, 0, 0]  # 기본 bbox

            # chunk_idx로 block_info_list에서 해당 블록 정보 직접 가져오기 시도
            if chunk_idx < len(block_info_list):
                block_info = block_info_list[chunk_idx]
                page_idx = block_info.get("page_idx", 0)
                bbox = block_info.get("bbox", [0, 0, 0, 0])
                # bbox가 유효하지 않으면 기본값으로 재설정 (필요시)
                if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox)):
                    logger.info(f"[Parser-WARNING] Chunk {chunk_idx}의 bbox 형식이 유효하지 않아 기본값 사용: {bbox}")
                    bbox = [0, 0, 0, 0]
            else:
                # block_info_list 길이가 content_list보다 짧으면 경고 출력 (한 번만)
                if chunk_idx == len(block_info_list):
                    logger.warning(f"[Parser-WARNING] block_info_list 길이({len(block_info_list)})가 content_list 길이({len(content_list)})보다 짧음. 이후 블록은 기본 page_idx와 bbox 사용.")


            if chunk_data["type"] == "text" or chunk_data["type"] == "table" or chunk_data["type"] == "equation" or chunk_data["type"] == "title":
                if chunk_data["type"] == "text" or chunk_data["type"] == "title":
                    content = chunk_data["text"]
                    if not content or not content.strip():
                        continue
                    # 마크다운 특수문자 필터링
                    content = re.sub(r"[!#\\$/]", "", content)
                elif chunk_data["type"] == "equation":
                    content = chunk_data["text"]
                    if not content or not content.strip():
                        continue
                elif chunk_data["type"] == "table":
                    # MinerU는 table 내용을 table_body 또는 text 필드에 저장할 수 있음
                    caption_list = chunk_data.get("table_caption", [])  # 리스트, 기본값 빈 리스트
                    table_body = chunk_data.get("table_body", "")  # 표 본문 (우선순위 1)
                    
                    # table_body가 비어있으면 text 필드 사용 (fallback)
                    if not table_body.strip():
                        table_body = chunk_data.get("text", "")
                        logger.info(f"[Parser-INFO] Table 블록: table_body 비어있음, text 필드 사용")

                    # 표 본문이 완전히 비어 있으면 건너뜀
                    if not table_body.strip():
                        logger.warning(f"[Parser-WARNING] Table 블록 {chunk_idx}: table_body와 text 모두 비어있어 건너뜀")
                        continue

                    # caption_list가 문자열 리스트인지 확인
                    if isinstance(caption_list, list) and all(isinstance(item, str) for item in caption_list):
                        # 리스트의 모든 문자열을 공백으로 연결
                        caption_str = " ".join(caption_list)
                    elif isinstance(caption_list, str):
                        # caption이 문자열이면 바로 사용
                        caption_str = caption_list
                    else:
                        # 기타(빈 리스트, None 등)면 빈 문자열 사용
                        caption_str = ""
                    # 처리된 캡션과 표 본문 연결
                    content = caption_str + table_body


                embedding_vec = []  # 빈 리스트로 초기화
                # 임베딩 벡터 구하기
                try:
                    headers = {"Content-Type": "application/json"}
                    if embedding_api_key:
                        headers["Authorization"] = f"Bearer {embedding_api_key}"

                    if is_ollama:
                        embedding_resp = requests.post(
                            embedding_url,  # 동적으로 생성된 URL 사용
                            headers=headers,  # headers 추가 (API Key 포함 가능)
                            json={
                                "model": embedding_model_name,  # 동적으로 가져오거나 기본 모델명 사용
                                "prompt": content,
                            },
                            timeout=15,  # 타임아웃 약간 증가
                        )
                    else:
                        embedding_resp = requests.post(
                            embedding_url,  # 동적으로 생성된 URL 사용
                            headers=headers,  # headers 추가 (API Key 포함 가능)
                            json={
                                "model": embedding_model_name,  # 동적으로 가져오거나 기본 모델명 사용
                                "input": content,
                            },
                            timeout=15,  # 타임아웃 약간 증가
                        )

                    embedding_resp.raise_for_status()
                    embedding_data = embedding_resp.json()

                    # ollama 임베딩 모델의 반환값은 별도 처리
                    if is_ollama:
                        embedding_vec = embedding_data.get("embedding")
                    else:
                        embedding_vec = embedding_data["data"][0]["embedding"]

                    # 벡터 차원이 예상과 다르면 오류
                    if len(embedding_vec) != embedding_dim:
                        error_msg = f"[Parser-ERROR] 임베딩 벡터 차원이 일치하지 않음, 예상: {embedding_dim}, 실제: {len(embedding_vec)}"
                        logger.error(error_msg)
                        update_progress(-5, error_msg)
                        raise ValueError(error_msg)
                    logger.info(f"[Parser-INFO] 임베딩 성공, 차원: {len(embedding_vec)}")
                except Exception as e:
                    logger.error(f"[Parser-ERROR] 임베딩 실패: {e}")
                    raise Exception(f"[Parser-ERROR] 임베딩 실패: {e}")

                chunk_id = generate_uuid()


                try:
                    # ES 문서 준비
                    current_time_es = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_timestamp_es = datetime.now().timestamp()

                    # 좌표 포맷 변환
                    x1, y1, x2, y2 = bbox
                    bbox_reordered = [x1, x2, y1, y2]

                    # Title 매핑에서 연관된 title 가져오기
                    section_title = title_mapping.get(chunk_idx, "")
                    
                    # 문서 이름과 섹션 title 결합
                    if section_title:
                        combined_title = f"{doc_info['name']} {section_title}".strip()
                        logger.debug(f"[Parser-INFO] Chunk {chunk_idx} title: '{combined_title[:80]}...'")
                    else:
                        combined_title = doc_info["name"]
                    
                    # 토큰화 결과를 한 번만 계산하여 재사용 (중복 로그 방지)
                    title_tokens = tokenize_text(combined_title)
                    content_tokens = tokenize_text(content)

                    es_doc = {
                        "doc_id": doc_id,
                        "kb_id": kb_id,
                        "docnm_kwd": doc_info["name"],
                        "title_tks": title_tokens,
                        "title_sm_tks": title_tokens,
                        "content_with_weight": content,
                        "content_ltks": content_tokens,
                        "content_sm_ltks": content_tokens,
                        "page_num_int": [page_idx + 1],
                        "position_int": [[page_idx + 1] + bbox_reordered],  # 포맷: [[page, x1, x2, y1, y2]]
                        "top_int": [1],
                        "create_time": current_time_es,
                        "create_timestamp_flt": current_timestamp_es,
                        "img_id": "",
                        vector_field_name: embedding_vec,
                    }

                    # Elasticsearch에 저장
                    es_client.index(index=index_name, id=chunk_id, document=es_doc)  # document 파라미터 사용

                    chunk_count += 1
                    chunk_ids_list.append(chunk_id)
                    
                    # chunk 메타데이터 저장 (이미지 연관을 위해)
                    chunk_metadata[chunk_id] = {
                        "page_idx": page_idx,
                        "bbox": bbox  # [x1, y1, x2, y2]
                    }

                except Exception as e:
                    logger.error(f"[Parser-ERROR] 텍스트 블록 {chunk_idx} (page: {page_idx}, bbox: {bbox}) 처리 실패: {e}")
                    raise Exception(f"[Parser-ERROR] 텍스트 블록 {chunk_idx} (page: {page_idx}, bbox: {bbox}) 처리 실패: {e}")


            elif chunk_data["type"] == "image":
                img_path_relative = chunk_data.get("img_path")
                if not img_path_relative or not temp_image_dir:
                    continue

                img_path_abs = os.path.join(temp_image_dir, os.path.basename(img_path_relative))
                if not os.path.exists(img_path_abs):
                    logger.warning(f"[Parser-WARNING] 이미지 파일이 존재하지 않음: {img_path_abs}")
                    continue

                # 이미지의 페이지 및 bbox 정보 가져오기
                img_page_idx = 0
                img_bbox = [0, 0, 0, 0]
                if chunk_idx < len(block_info_list):
                    img_block_info = block_info_list[chunk_idx]
                    img_page_idx = img_block_info.get("page_idx", 0)
                    img_bbox = img_block_info.get("bbox", [0, 0, 0, 0])

                img_id = generate_uuid()
                img_ext = os.path.splitext(img_path_abs)[1]
                img_key = f"images/{img_id}{img_ext}"  # MinIO 내 오브젝트명
                content_type = f"image/{img_ext[1:].lower()}"
                if content_type == "image/jpg":
                    content_type = "image/jpeg"

                try:
                    # MinIO에 이미지 업로드 (버킷은 kb_id)
                    minio_client.fput_object(bucket_name=output_bucket, object_name=img_key, file_path=img_path_abs, content_type=content_type)

                    # 이미지 공개 접근 권한 설정
                    policy = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"AWS": "*"}, "Action": ["s3:GetObject"], "Resource": [f"arn:aws:s3:::{kb_id}/images/*"]}]}
                    minio_client.set_bucket_policy(kb_id, json.dumps(policy))

                    logger.info(f"이미지 업로드 성공: {img_key}")
                    minio_endpoint = MINIO_CONFIG["endpoint"]
                    use_ssl = MINIO_CONFIG.get("secure", False)
                    protocol = "https" if use_ssl else "http"
                    img_url = f"{protocol}://{minio_endpoint}/{output_bucket}/{img_key}"

                    # 이미지 정보 기록 (URL, 페이지, bbox)
                    image_info = {
                        "url": img_url,
                        "page_idx": img_page_idx,
                        "bbox": img_bbox,  # [x1, y1, x2, y2]
                    }
                    image_info_list.append(image_info)

                    logger.info(f"이미지 접근 링크: {img_url}")

                except Exception as e:
                    logger.error(f"[Parser-ERROR] 이미지 업로드 실패 {img_path_abs}: {e}")
                    raise Exception(f"[Parser-ERROR] 이미지 업로드 실패 {img_path_abs}: {e}")

        # 처리 요약 정보 출력
        logger.info(f"[Parser-INFO] 총 {chunk_count}개의 텍스트 블록 처리함.")

        # 4. 텍스트 블록의 이미지 정보 업데이트
        if image_info_list and chunk_ids_list:
            linked_count = 0
            failed_count = 0
            
            try:
                for img_info in image_info_list:
                    img_page = img_info["page_idx"]
                    
                    # 가장 가까운 청크 찾기
                    nearest_chunk_id, match_type, distance = find_chunk_to_image(
                        img_info=img_info,
                        chunk_ids_list=chunk_ids_list,
                        chunk_metadata=chunk_metadata
                    )
                    
                    # 가장 가까운 chunk가 있으면 이미지 연결
                    if nearest_chunk_id:
                        # 상대 경로 부분 저장
                        parsed_url = urlparse(img_info["url"])
                        relative_path = parsed_url.path.lstrip("/")  # 앞의 슬래시 제거
                        # ES 문서 업데이트
                        direct_update = {"doc": {"img_id": relative_path}}
                        es_client.update(index=index_name, id=nearest_chunk_id, body=direct_update, refresh=True)
                        logger.info(f"[Parser-INFO] 페이지 {img_page}의 텍스트 블록 {nearest_chunk_id}에 이미지 연결 ({match_type}, 거리: {distance:.2f}): {relative_path}")
                        linked_count += 1
                    else:
                        logger.warning(f"[Parser-WARNING] 페이지 {img_page}의 이미지에 대해 연결할 chunk를 찾지 못함")
                        failed_count += 1
                
                logger.info(f"[Parser-INFO] 이미지-청크 연결 완료: 성공 {linked_count}개, 실패 {failed_count}개")
                
            except Exception as e:
                logger.error(f"[Parser-ERROR] 텍스트 블록 이미지 연결 업데이트 실패: {e}")
                raise Exception(f"[Parser-ERROR] 텍스트 블록 이미지 연결 업데이트 실패: {e}")


        # 5. 최종 상태 업데이트
        process_duration = time.time() - start_time
        _update_document_progress(doc_id, progress=1.0, message="파싱 완료", status="1", run="3", chunk_count=chunk_count, process_duration=process_duration)
        _update_kb_chunk_count(kb_id, chunk_count)  # 지식베이스 전체 블록 수 업데이트
        _create_task_record(doc_id, chunk_ids_list)  # task 기록 생성

        update_progress(1.0, "파싱 완료")
        logger.info(f"[Parser-INFO] 파싱 완료, 문서ID: {doc_id}, 소요시간: {process_duration:.2f}s, 블록 수: {chunk_count}")

        return {"success": True, "chunk_count": chunk_count}

    except Exception as e:
        process_duration = time.time() - start_time
        logger.error(f"[Parser-ERROR] 문서 {doc_id} 파싱 실패: {e}")
        error_message = f"파싱 실패: {e}"
        # 문서 상태를 실패로 업데이트
        _update_document_progress(doc_id, status="1", run="0", message=error_message, process_duration=process_duration)  # status=1은 완료, run=0은 실패
        return {"success": False, "error": error_message}

    finally:
        # 명시적 메모리 해제 (임시 파일 정리 전)
        logger.info(f"[Parser-INFO] Finally 블록에서 최종 메모리 정리 시작 (Doc ID: {doc_id})")
        try:
            # 대용량 객체들 명시적 해제
            cleanup_vars = [
                'file_content', 'pdf_bytes', 'content_list', 
                'middle_json_data', 'middle_json_content', 'middle_json',
                'infer_results', 'all_image_lists', 'all_pdf_docs',
                'model_list', 'images_list', 'pdf_doc', 'pdf_info',
                'image_info_list', 'block_info_list', 'chunk_ids_list', 'chunk_metadata'
            ]
            
            for var_name in cleanup_vars:
                if var_name in locals():
                    try:
                        del locals()[var_name]
                    except:
                        pass  # 이미 삭제된 경우 무시
            
            # 가비지 컬렉션 강제 실행
            gc.collect()
            logger.info(f"[Parser-INFO] 메모리 해제 완료 (Doc ID: {doc_id})")
        except Exception as mem_err:
            logger.warning(f"[Parser-WARNING] 메모리 해제 중 경고 (Doc ID: {doc_id}): {mem_err}")
        
        # 임시 파일 정리
        try:
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                logger.info(f"[Parser-INFO] 임시 PDF 파일 삭제: {temp_pdf_path}")
            if temp_image_dir and os.path.exists(temp_image_dir):
                shutil.rmtree(temp_image_dir, ignore_errors=True)
                logger.info(f"[Parser-INFO] 임시 이미지 디렉토리 삭제: {temp_image_dir}")
        except Exception as clean_e:
            logger.error(f"[Parser-WARNING] 임시 파일 정리 실패: {clean_e}")
        
        # 최종 가비지 컬렉션
        gc.collect()
        logger.info(f"[Parser-INFO] 최종 정리 완료 (Doc ID: {doc_id})")
