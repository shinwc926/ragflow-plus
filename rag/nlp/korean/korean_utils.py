#!/usr/bin/env python3
"""
RAGflow-plus 한국어 공통 유틸리티
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# =============================================================================
# 한국어 감지 및 기본 유틸리티 함수들
# =============================================================================

def is_korean_char(char: str) -> bool:
    """단일 문자가 한국어인지 확인"""
    if not char:
        return False
    code = ord(char)
    return (0xAC00 <= code <= 0xD7AF) or \
           (0x3130 <= code <= 0x318F) or \
           (0x1100 <= code <= 0x11FF)

def contains_korean(text: str) -> bool:
    """문자열에 한국어가 포함되어 있는지 확인"""
    if not text:
        return False
    return any(is_korean_char(char) for char in text)

def get_korean_ratio(text: str) -> float:
    """텍스트에서 한국어 문자 비율 계산"""
    if not text:
        return 0.0
    
    korean_chars = sum(1 for char in text if is_korean_char(char))
    total_chars = len([char for char in text if not char.isspace()])
    
    return korean_chars / total_chars if total_chars > 0 else 0.0

def is_primarily_korean(text: str, threshold: float = 0.5) -> bool:
    """텍스트가 주로 한국어인지 확인"""
    return get_korean_ratio(text) >= threshold

# =============================================================================
# 사전 및 설정 관련 함수들
# =============================================================================

def get_custom_dict_path() -> Optional[str]:
    """커스텀 사전 경로 다 넣음"""
    possible_paths = [
        "/ragflow/KOR_MORPHEME/insurance_custom_dict.dict",
        "/home/snucsnl/shin_ragflow/ragflow-plus/KOR_MORPHEME/insurance_custom_dict.dict",
        "/home/snucsnl/shin_ragflow/ragflow-plus-kor/KOR_MORPHEME/insurance_custom_dict.dict",
        os.path.join(Path(__file__).parent.parent.parent, "KOR_MORPHEME", "insurance_custom_dict.dict"),
        os.path.join(Path(__file__).parent, "insurance_custom_dict.dict"),
        "./KOR_MORPHEME/insurance_custom_dict.dict",
        "insurance_custom_dict.dict"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def load_dict_scores(dict_path: str) -> Dict[str, float]:
    """사용자 사전에서 점수 로드"""
    scores = {}
    if not dict_path or not os.path.exists(dict_path):
        return scores
    
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        word, pos, score = parts[0], parts[1], parts[2]
                        try:
                            scores[word] = float(score)
                        except ValueError:
                            pass
    except Exception as e:
        pass
    
    return scores

def get_pos_scores() -> Dict[str, float]:
    """POS 태그별 점수 - 명사 중심, 조사/어미는 매우 낮게"""
    return {
        # 명사류 (높은 점수)
        'NNG': 2.0,    # 일반명사
        'NNP': 2.5,    # 고유명사
        'NNB': 1.8,    # 의존명사
        'NR': 2.0,     # 수사
        'NP': 1.7,     # 대명사
        
        # 동사/형용사류 (중간 점수)
        'VV': 0.9,     # 동사
        'VA': 0.8,     # 형용사
        'VX': 0.6,     # 보조용언
        'VCP': 0.3,    # 긍정지정사
        'VCN': 0.4,    # 부정지정사
        
        # 수식언류
        'MM': 0.7,     # 관형사
        'MAG': 0.6,    # 일반부사
        'MAJ': 0.7,    # 접속부사
        'IC': 0.3,     # 감탄사
        
        # 조사류 (매우 낮은 점수)
        'JKS': 0.05,   # 주격조사
        'JKC': 0.05,   # 보격조사
        'JKG': 0.05,   # 관형격조사
        'JKO': 0.05,   # 목적격조사
        'JKB': 0.05,   # 부사격조사
        'JKV': 0.05,   # 호격조사
        'JKQ': 0.05,   # 인용격조사
        'JX': 0.05,    # 보조사
        'JC': 0.05,    # 접속조사
        
        # 어미류 (매우 낮은 점수)
        'EP': 0.03,    # 선어말어미
        'EF': 0.03,    # 종결어미
        'EC': 0.03,    # 연결어미
        'ETN': 0.05,   # 명사형전성어미
        'ETM': 0.05,   # 관형사형전성어미
        
        # 접사류
        'XPN': 0.4,    # 체언접두사
        'XSN': 0.4,    # 명사파생접미사
        'XSV': 0.3,    # 동사파생접미사
        'XSA': 0.3,    # 형용사파생접미사
        'XSM': 0.3,    # 부사파생접미사
        'XR': 0.5,     # 어근
        
        # 기호류 (매우 낮은 점수)
        'SF': 0.01,    # 마침표,물음표,느낌표
        'SP': 0.01,    # 쉼표,가운뎃점,콜론,빗금
        'SS': 0.01,    # 따옴표,괄호표,줄표
        'SE': 0.01,    # 줄임표
        'SO': 0.01,    # 붙임표
        'SW': 0.02,    # 기타기호
        'SL': 0.3,     # 외국어
        'SH': 0.8,     # 한자
        'SN': 0.5,     # 숫자
        'SB': 0.1,     # 기타 문자
        
        # 분석 불능
        'UN': 0.0,     # 미등록어
        
        # 웹 관련
        'W_URL': 0.7,
        'W_EMAIL': 0.7,
        'W_HASHTAG': 0.6,
        'W_MENTION': 0.6,
        'W_SERIAL': 0.8,
        'W_EMOJI': 0.2,
        
        # 특수
        'Z_CODA': 0.01,
        'Z_SIOT': 0.01
    }

def get_excluded_pos_tags() -> set:
    """제거할 POS 태그들 정의"""
    return {
        # 조사류
        'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC',
        # 어미류
        'EP', 'EF', 'EC', 'ETN', 'ETM',
        # 기호류
        'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB',
        # 기타
        'UN', 'W_EMOJI', 'Z_CODA', 'Z_SIOT'
    }

def should_exclude_token(token: str, pos: str = "", min_length: int = 1) -> bool:
    """토큰을 제외할지 판단"""
    if not token or len(token) < min_length:
        return True
    
    # 제외할 POS 태그 확인
    if pos in get_excluded_pos_tags():
        return True
    
    # 단순 기호나 공백만 있는 경우
    if token.strip() == "" or token in ".,!?;:":
        return True
    
    # 너무 짧은 조사나 어미
    if len(token) == 1 and pos.startswith(('J', 'E')):
        return True
    
    return False

# =============================================================================
# 로깅 유틸리티
# =============================================================================

def setup_korean_logger(name: str = 'korean_search') -> logging.Logger:
    """한국어 검색 전용 로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 핸들러가 이미 있으면 추가하지 않음
    if logger.handlers:
        return logger
    
    # 로그 디렉토리 생성
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}.log'), 
        encoding='utf-8'
    )
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# =============================================================================
# 모듈 초기화
# =============================================================================

# 기본 로거 설정
korean_logger = setup_korean_logger('korean_search')

# 커스텀 사전 경로 설정
CUSTOM_DICT_PATH = get_custom_dict_path()
if CUSTOM_DICT_PATH:
    korean_logger.info(f"Custom dictionary found: {CUSTOM_DICT_PATH}")
else:
    korean_logger.warning("Custom dictionary not found")

# POS 점수 및 제외 태그 로드
POS_SCORES = get_pos_scores()
EXCLUDED_POS_TAGS = get_excluded_pos_tags()

korean_logger.info("Korean utils module initialized successfully")
