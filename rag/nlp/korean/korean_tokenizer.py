#!/usr/bin/env python3
"""
RAGflow-plus 한국어 토크나이저
"""
import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent))

# 공통 유틸리티 import
from .korean_utils import (
    contains_korean, is_korean_char, get_custom_dict_path, 
    load_dict_scores, get_pos_scores, get_excluded_pos_tags, 
    should_exclude_token, korean_logger, CUSTOM_DICT_PATH, 
    POS_SCORES, EXCLUDED_POS_TAGS
)

# Kiwi import
try:
    from kiwipiepy import Kiwi
    kiwi_available = True
except ImportError:
    kiwi_available = False
    korean_logger.warning("kiwipiepy not available")

# ragflow-plus 모듈 import
try:
    from rag.utils.doc_store_conn import MatchTextExpr
    ragflow_available = True
except ImportError:
    ragflow_available = False
    korean_logger.warning("RagFlow modules not available")

# =============================================================================
# Kiwi 초기화 및 관리
# =============================================================================

class KiwiManager:
    """Kiwi 인스턴스 관리 클래스"""
    
    def __init__(self):
        self._kiwi_instance = None
        self._dict_scores = {}
        self._initialized = False
    
    def initialize(self):
        """Kiwi 초기화"""
        if not kiwi_available:
            korean_logger.error("Kiwi not available")
            return False
        
        if self._initialized:
            return True
        
        try:
            # Kiwi 인스턴스 생성
            self._kiwi_instance = Kiwi(
                num_workers=16,
                model_path=None,
                load_default_dict=True,
                integrate_allomorph=True,
                model_type='knlm',
                typos=None,
                typo_cost_threshold=2.5
            )
            
            # 커스텀 사전 로드
            if CUSTOM_DICT_PATH and os.path.exists(CUSTOM_DICT_PATH):
                self._kiwi_instance.load_user_dictionary(CUSTOM_DICT_PATH)
                self._dict_scores = load_dict_scores(CUSTOM_DICT_PATH)
                korean_logger.info(f"Custom dictionary loaded: {CUSTOM_DICT_PATH}")
            else:
                korean_logger.warning("Custom dictionary not found")
            
            self._initialized = True
            korean_logger.info("Kiwi tokenizer initialized successfully")
            return True
            
        except Exception as e:
            korean_logger.error(f"Kiwi initialization failed: {e}")
            return False
    
    @property
    def kiwi(self):
        """Kiwi 인스턴스 반환"""
        if not self._initialized:
            self.initialize()
        return self._kiwi_instance
    
    @property
    def dict_scores(self):
        """사전 점수 반환"""
        if not self._initialized:
            self.initialize()
        return self._dict_scores
    
    def is_available(self):
        """Kiwi 사용 가능 여부"""
        return kiwi_available and self._initialized

# 전역 Kiwi 매니저
_kiwi_manager = KiwiManager()

def get_kiwi():
    """Kiwi 인스턴스 반환"""
    return _kiwi_manager.kiwi

def get_dict_scores():
    """사전 점수 반환"""
    return _kiwi_manager.dict_scores

# =============================================================================
# 토큰화 함수들
# =============================================================================

def tokenize_to_weighted_tokens(text: str) -> List[Dict[str, Any]]:
    """텍스트를 가중치 토큰으로 변환"""
    if not text or not _kiwi_manager.is_available():
        return []
    
    try:
        tokens = _kiwi_manager.kiwi.tokenize(text)
        weighted_tokens = []
        dict_scores = _kiwi_manager.dict_scores
        
        for token in tokens:
            # POS 점수 계산
            pos_score = POS_SCORES.get(token.tag, 0.0)
            
            # 사전 점수 계산
            custom_score = dict_scores.get(token.form, 0.0)
            
            # 최종 점수
            final_score = pos_score + custom_score
            
            weighted_token = {
                "token": token.form,
                "boost": final_score,
                "pos": token.tag
            }
            weighted_tokens.append(weighted_token)
        
        return weighted_tokens
        
    except Exception as e:
        korean_logger.error(f"Tokenization failed: {e}")
        return []
# 
def get_filtered_tokens_with_compounds(text: str, min_length: int = 1, min_boost: float = 0.0) -> List[Dict[str, Any]]:
    """복합명사를 보존하면서 필터링된 토큰 리스트 반환"""
    if not text or not _kiwi_manager.is_available():
        return []
    
    try:
        # 1. 먼저 복합명사 추출
        compound_tokens = extract_compound_tokens_by_spacing(text, _kiwi_manager.kiwi)
        
        filtered_tokens = []
        dict_scores = _kiwi_manager.dict_scores
        
        if compound_tokens:
            # 2. 복합어 처리
            for compound in compound_tokens:
                compound_form = compound['form']
                
                # 제외할 토큰인지 확인
                if should_exclude_token(compound_form, 'NNG', min_length):
                    continue
                
                # 복합명사의 점수 계산 (구성 요소들의 최대값 사용)
                component_scores = []
                for component in compound['components']:
                    pos_score = POS_SCORES.get(component['tag'], 0.0)
                    custom_score = dict_scores.get(component['form'], 0.0)
                    component_scores.append(pos_score + custom_score)
                
                final_score = max(component_scores) if component_scores else POS_SCORES.get('NNG', 0.0)
                
                # 최소 부스트 점수 확인
                if final_score < min_boost:
                    continue
                
                weighted_token = {
                    "token": compound_form,
                    "boost": final_score,
                    "pos": "NNG",
                    "is_compound": True,
                    "components": compound['components']
                }
                filtered_tokens.append(weighted_token)
            
            # 3. 복합어 부분을 제외한 나머지 텍스트 처리
            remaining_text = text
            for compound in sorted(compound_tokens, key=lambda x: x['start'], reverse=True):
                remaining_text = remaining_text[:compound['start']] + remaining_text[compound['end']:]
            
            # 남은 텍스트가 있으면 토큰화
            if remaining_text.strip():
                tokens = _kiwi_manager.kiwi.tokenize(remaining_text.strip())
                for token in tokens:
                    if should_exclude_token(token.form, token.tag, min_length):
                        continue
                    
                    pos_score = POS_SCORES.get(token.tag, 0.0)
                    custom_score = dict_scores.get(token.form, 0.0)
                    final_score = pos_score + custom_score
                    
                    if final_score < min_boost:
                        continue
                    
                    weighted_token = {
                        "token": token.form,
                        "boost": final_score,
                        "pos": token.tag,
                        "is_compound": False
                    }
                    filtered_tokens.append(weighted_token)
        else:
            # 복합어가 없으면 전체 텍스트를 일반적으로 토큰화
            tokens = _kiwi_manager.kiwi.tokenize(text)
            for token in tokens:
                if should_exclude_token(token.form, token.tag, min_length):
                    continue
                
                pos_score = POS_SCORES.get(token.tag, 0.0)
                custom_score = dict_scores.get(token.form, 0.0)
                final_score = pos_score + custom_score
                
                if final_score < min_boost:
                    continue
                
                weighted_token = {
                    "token": token.form,
                    "boost": final_score,
                    "pos": token.tag,
                    "is_compound": False
                }
                filtered_tokens.append(weighted_token)
        
        return filtered_tokens
        
    except Exception as e:
        korean_logger.error(f"Token filtering with compounds failed: {e}")
        return []


def extract_compound_tokens_by_spacing(text: str, kiwi) -> List[Dict[str, Any]]:
    """원문 띄어쓰기 기준으로 복합어 추출 (첫 번째 코드에서 추출)"""
    try:
        analysis = kiwi.analyze(text)
        if not analysis or not analysis[0]:
            return []
        
        # 형태소 분석으로 내용어와 위치 정보 추출
        tokens = []
        for token in analysis[0][0]:
            if token.tag in ('NNG', 'NNP', 'NNB', 'SL', 'SN'):
                tokens.append({
                    'form': token.form,
                    'start': token.start,
                    'end': token.start + token.len,
                    'tag': token.tag
                })
        
        if not tokens:
            return []
        
        compound_results = []
        i = 0
        
        while i < len(tokens):
            current_group = [tokens[i]]
            current_end = tokens[i]['end']
            
            # 다음 토큰들과 연속성 체크
            j = i + 1
            while j < len(tokens):
                next_token = tokens[j]
                between_text = text[current_end:next_token['start']]
                has_space = ' ' in between_text or '\t' in between_text or '\n' in between_text
                
                if not has_space and (next_token['start'] - current_end <= 1):
                    current_group.append(next_token)
                    current_end = next_token['end']
                    j += 1
                else:
                    break
            
            # 복합어인 경우만 저장
            if len(current_group) > 1:
                compound_form = ''.join([token['form'] for token in current_group])
                original_text = text[current_group[0]['start']:current_group[-1]['end']]
                
                compound_results.append({
                    'form': compound_form,
                    'original': original_text,
                    'components': current_group,
                    'start': current_group[0]['start'],
                    'end': current_group[-1]['end']
                })
            
            i = j if j > i else i + 1
        
        return compound_results
        
    except Exception as e:
        print(f"복합어 추출 오류: {e}")
        return []
#
    
# NOTE : 복합명사 너무 부서짐.. 띄어쓰기 기반으로 살리기(신 전무님 조언 참고)
# def get_filtered_tokens(text: str, min_length: int = 1, min_boost: float = 0.0) -> List[Dict[str, Any]]:
#     """필터링된 토큰 리스트 반환"""
#     if not text or not _kiwi_manager.is_available():
#         return []
    
#     try:
#         tokens = _kiwi_manager.kiwi.tokenize(text)
#         filtered_tokens = []
#         dict_scores = _kiwi_manager.dict_scores
        
#         for token in tokens:
#             # 제외할 토큰인지 확인
#             if should_exclude_token(token.form, token.tag, min_length):
#                 continue
            
#             # 점수 계산
#             pos_score = POS_SCORES.get(token.tag, 0.0)
#             custom_score = dict_scores.get(token.form, 0.0)
#             final_score = pos_score + custom_score
            
#             # 최소 부스트 점수 확인
#             if final_score < min_boost:
#                 continue
            
#             weighted_token = {
#                 "token": token.form,
#                 "boost": final_score,
#                 "pos": token.tag
#             }
#             filtered_tokens.append(weighted_token)
        
#         return filtered_tokens
        
#     except Exception as e:
#         korean_logger.error(f"Token filtering failed: {e}")
#         return []



def extract_meaningful_words(text: str, min_boost: float = 0.5) -> List[str]:
    """의미있는 단어만 추출"""
    tokens = get_filtered_tokens_with_compounds(text, min_boost=min_boost)
    return [token["token"] for token in tokens]

def korean_tokenize_for_search(text: str) -> List[Dict[str, Any]]:
    """검색용 한국어 토큰화"""
    if not contains_korean(text):
        return []
    return tokenize_to_weighted_tokens(text)

def korean_extract_keywords(text: str, min_boost: float = 0.5) -> List[str]:
    """한국어 키워드 추출"""
    tokens = korean_tokenize_for_search(text)
    return [token["token"] for token in tokens if token["boost"] >= min_boost]

def korean_weighted_query(text: str) -> str:
    """한국어용 가중치 쿼리 생성"""
    tokens = korean_tokenize_for_search(text)
    if not tokens:
        return text
    
    # 가중치가 높은 토큰들로 쿼리 구성
    weighted_terms = []
    for token in tokens:
        if token["boost"] > 0.3:
            weighted_terms.append(f"{token['token']}^{token['boost']:.2f}")
    
    return " ".join(weighted_terms) if weighted_terms else text

# =============================================================================
# RagFlow 호환 클래스들
# =============================================================================

class KoreanRagTokenizer:
    """RagFlow 호환 한국어 토크나이저"""
    
    def __init__(self):
        korean_logger.info("Korean RagTokenizer initialized")
    
    def tokenize(self, text: str) -> List[str]:
        """기본 토큰화"""
        tokens = get_filtered_tokens_with_compounds(text, min_boost=0.1)
        return [token["token"] for token in tokens]
    
    def fine_grained_tokenize(self, text: str) -> List[str]:
        """세분화된 토큰화"""
        tokens = get_filtered_tokens_with_compounds(text, min_length=1, min_boost=0.0)
        return [token["token"] for token in tokens]
    
    def tag(self, token: str) -> str:
        """POS 태깅"""
        if not _kiwi_manager.is_available():
            return "UN"
        
        try:
            tokens = _kiwi_manager.kiwi.tokenize(token)
            if tokens:
                return tokens[0].tag
        except:
            pass
        return "UN"
    
    def freq(self, token: str) -> float:
        """토큰 빈도"""
        dict_scores = _kiwi_manager.dict_scores
        return dict_scores.get(token, 0.0)

# =============================================================================
# 전역 인스턴스 관리
# =============================================================================

_korean_rag_tokenizer = None

def get_korean_rag_tokenizer():
    """한국어 RagTokenizer 전역 인스턴스 반환"""
    global _korean_rag_tokenizer
    if _korean_rag_tokenizer is None:
        _korean_rag_tokenizer = KoreanRagTokenizer()
    return _korean_rag_tokenizer

# =============================================================================
# 테스트 함수
# =============================================================================

def test_korean_tokenizer():
    """한국어 토크나이저 테스트"""
    korean_logger.info("한국어 토크나이저 테스트 시작")
    
    test_texts = [
        "자동차보험 가입방법을 알려주세요",
        "의료비 보장 범위는 어떻게 되나요?",
        "보험료 계산 방식이 궁금합니다"
    ]
    
    for i, text in enumerate(test_texts, 1):
        korean_logger.info(f"\n=== 테스트 {i}: {text} ===")
        
        # 가중치 토큰화
        weighted_tokens = tokenize_to_weighted_tokens(text)
        korean_logger.info(f"가중치 토큰: {weighted_tokens[:5]}")  # 처음 5개만
        
        # 키워드 추출
        keywords = korean_extract_keywords(text)
        korean_logger.info(f"키워드: {keywords}")
        
        # 가중치 쿼리
        query = korean_weighted_query(text)
        korean_logger.info(f"쿼리: {query}")

# =============================================================================
# 모듈 초기화 및 export
# =============================================================================

__all__ = ['KiwiManager', 'KoreanRagTokenizer', 'get_filtered_tokens_with_compounds', 'get_boosted_terms']

# 자동 초기화
if kiwi_available:
    korean_logger.info("Initializing Kiwi tokenizer...")
    _kiwi_manager.initialize()
else:
    korean_logger.warning("Kiwi not available - Korean tokenizer disabled")

korean_logger.info("Korean tokenizer module loaded")

if __name__ == "__main__":
    test_korean_tokenizer()
