#!/usr/bin/env python3
"""
RAGflow-plus 한국어 검색 엔진
"""
import os
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 공통 유틸리티 import
from .korean_utils import (
    contains_korean, korean_logger, CUSTOM_DICT_PATH,
    load_dict_scores, POS_SCORES, EXCLUDED_POS_TAGS
)

# ragflow-plus 모듈 import
try:
    from rag.nlp.search import Dealer
    from rag.utils.doc_store_conn import DocStoreConnection, MatchTextExpr, OrderByExpr, FusionExpr, MatchDenseExpr
    from rag.settings import TAG_FLD, PAGERANK_FLD
    ragflow_available = True
except ImportError as e:
    korean_logger.warning(f"RagFlow modules not available: {e}")
    ragflow_available = False

# korean_tokenizer 모듈에서 함수들 import
try:
    from .korean_tokenizer import (
        tokenize_to_weighted_tokens,
        get_filtered_tokens_with_compounds,
        korean_extract_keywords,
        korean_weighted_query,
        KoreanRagTokenizer,
        get_korean_rag_tokenizer
    )
    korean_tokenizer_available = True
except ImportError as e:
    korean_logger.warning(f"Korean tokenizer not available: {e}")
    korean_tokenizer_available = False

# =============================================================================
# 한국어 검색 딜러
# =============================================================================

class KoreanSearchDealer(Dealer if ragflow_available else object):
    """
    한국어 지원 검색 처리기
    기존 RagFlow Dealer를 상속하여 한국어 처리 기능 추가
    """
    
    def __init__(self, dataStore: Optional[Any] = None):
        if ragflow_available:
            super().__init__(dataStore)
        else:
            self.dataStore = dataStore
        
        korean_logger.info("Korean Search Dealer initialized")
    
    def is_korean_query(self, query: str) -> bool:
        """쿼리가 한국어인지 확인"""
        return contains_korean(query)
    
    def enhance_korean_query(self, query: str) -> tuple:
        """한국어 쿼리 강화"""
        if not korean_tokenizer_available:
            return query, {}
        
        if not self.is_korean_query(query):
            return query, {}
        
        korean_logger.info(f"🔍 Processing Korean query: {query}")
        
        try:
            # 한국어 토큰화 및 키워드 추출
            tokens = get_filtered_tokens_with_compounds(query, min_length=1, min_boost=0.3)
            keywords = [token["token"] for token in tokens if token["boost"] > 0.5]
            
            if not keywords:
                return query, {}
            
            # 강화된 쿼리 생성
            enhanced_query = " ".join(keywords)
            
            korean_logger.info(f"Keywords extracted: {keywords}")
            korean_logger.info(f"Enhanced query: {enhanced_query}")
            
            return enhanced_query, {"original_query": query, "keywords": keywords}
            
        except Exception as e:
            korean_logger.error(f"Korean query enhancement failed: {e}")
            return query, {}
    
    def search(self, req, idx_names, kb_ids, emb_mdl=None, highlight=False, rank_feature=None):
        """검색 수행 (한국어 강화 포함)"""
        if not ragflow_available:
            korean_logger.error("RagFlow not available")
            return []
        
        # 쿼리 텍스트 추출
        query_text = self._extract_query_text(req)
        
        # 한국어 쿼리인지 확인하고 강화
        if self.is_korean_query(query_text):
            return self._korean_search(req, idx_names, kb_ids, emb_mdl, highlight, rank_feature)
        else:
            # 일반 검색
            return super().search(req, idx_names, kb_ids, emb_mdl, highlight, rank_feature)
    
    def _extract_query_text(self, req) -> str:
        """요청에서 쿼리 텍스트 추출"""
        if hasattr(req, 'question'):
            return req.question
        elif hasattr(req, 'query'):
            return req.query
        elif isinstance(req, str):
            return req
        return ""
    
    def _korean_search(self, req, idx_names, kb_ids, emb_mdl=None, highlight=False, rank_feature=None):
        """한국어 특화 검색"""
        korean_logger.info("Starting Korean-specific search")
        
        try:
            # 쿼리 텍스트 추출
            query_text = self._extract_query_text(req)
            
            # 한국어 쿼리 강화
            enhanced_query, metadata = self.enhance_korean_query(query_text)
            
            # 요청 객체 수정
            original_query = self._modify_request_query(req, enhanced_query)
            
            # 기본 검색 수행
            results = super().search(req, idx_names, kb_ids, emb_mdl, highlight, rank_feature)
            
            # 원래 쿼리 복원
            self._restore_request_query(req, original_query)
            
            korean_logger.info(f"Korean search completed: {len(results)} results")
            return results
            
        except Exception as e:
            korean_logger.error(f"Korean search failed: {e}")
            # 기본 검색으로 폴백
            return super().search(req, idx_names, kb_ids, emb_mdl, highlight, rank_feature)
    
    def _modify_request_query(self, req, enhanced_query: str) -> str:
        """요청 객체의 쿼리 수정"""
        original_query = ""
        if hasattr(req, 'question'):
            original_query = req.question
            req.question = enhanced_query
        elif hasattr(req, 'query'):
            original_query = req.query
            req.query = enhanced_query
        return original_query
    
    def _restore_request_query(self, req, original_query: str):
        """요청 객체의 쿼리 복원"""
        if hasattr(req, 'question'):
            req.question = original_query
        elif hasattr(req, 'query'):
            req.query = original_query

# =============================================================================
# 한국어 강화 쿼리어
# =============================================================================

class KoreanEnhancedQueryer:
    """한국어 강화 쿼리 생성기"""
    
    def __init__(self):
        korean_logger.info("Korean Enhanced Queryer initialized")
    
    def question(self, txt: str, min_match: float = 0.3):
        """한국어 질문 처리"""
        if not korean_tokenizer_available:
            return self._create_empty_match_text()
        
        if not contains_korean(txt):
            korean_logger.info(f"Non-Korean query: {txt}")
            return self._create_empty_match_text()
        
        korean_logger.info(f"Processing Korean question: {txt}")
        
        try:
            # 한국어 토큰화
            tokens = get_filtered_tokens_with_compounds(txt, min_length=1, min_boost=min_match)
            
            if not tokens:
                korean_logger.warning(f"No meaningful tokens found for: {txt}")
                return self._enhance_korean_fallback(txt)
            
            # MatchTextExpr 생성
            if ragflow_available:
                match_expr = MatchTextExpr()
                match_expr.field = "content"
                
                # 토큰들을 쿼리로 변환
                query_parts = [token["token"] for token in tokens if token["boost"] > min_match]
                match_expr.value = " ".join(query_parts)
                match_expr.boost = 1.0
                
                korean_logger.info(f"Created MatchTextExpr: {match_expr.value}")
                return match_expr
            else:
                return self._create_mock_match_text(" ".join([t["token"] for t in tokens]))
                
        except Exception as e:
            korean_logger.error(f"Korean question processing failed: {e}")
            return self._enhance_korean_fallback(txt)
    
    def _enhance_korean_fallback(self, txt: str):
        """한국어 처리 실패시 폴백"""
        korean_logger.info(f"Using fallback for: {txt}")
        
        # 간단한 공백 기반 분할
        words = [word.strip() for word in txt.split() if word.strip()]
        meaningful_words = [word for word in words if len(word) > 1]
        
        if not meaningful_words:
            return self._create_empty_match_text()
        
        if ragflow_available:
            match_expr = MatchTextExpr()
            match_expr.field = "content"
            match_expr.value = " ".join(meaningful_words)
            match_expr.boost = 0.8
            return match_expr
        else:
            return self._create_mock_match_text(" ".join(meaningful_words))
    
    def _create_empty_match_text(self):
        """빈 MatchTextExpr 생성"""
        if ragflow_available:
            match_expr = MatchTextExpr()
            match_expr.field = "content"
            match_expr.value = ""
            match_expr.boost = 0.0
            return match_expr
        else:
            return self._create_mock_match_text("")
    
    def _create_mock_match_text(self, text):
        """Mock MatchTextExpr 생성 (RagFlow 없을 때)"""
        class MockMatchText:
            def __init__(self, value):
                self.field = "content"
                self.value = value
                self.boost = 1.0
        
        return MockMatchText(text)

# =============================================================================
# 편의 함수들
# =============================================================================

def korean_search_query(query: str, min_boost: float = 0.5) -> Dict[str, Any]:
    """한국어 검색 쿼리 분석 및 강화"""
    if not korean_tokenizer_available:
        korean_logger.warning("Korean tokenizer not available")
        return {"original_query": query, "enhanced_query": query, "keywords": []}
    
    is_korean = contains_korean(query)
    
    if is_korean:
        try:
            tokens = get_filtered_tokens_with_compounds(query, min_boost=min_boost)
            keywords = [token["token"] for token in tokens]
            enhanced_query = " ".join(keywords) if keywords else query
            
            return {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "keywords": keywords,
                "is_korean": True,
                "token_count": len(tokens)
            }
        except Exception as e:
            korean_logger.error(f"Korean query analysis failed: {e}")
            return {"original_query": query, "enhanced_query": query, "keywords": [], "is_korean": True}
    else:
        return {"original_query": query, "enhanced_query": query, "keywords": [], "is_korean": False}

# =============================================================================
# 전역 인스턴스 관리
# =============================================================================

_korean_search_dealer = None
_korean_queryer = None

def get_korean_search_dealer_instance(dataStore=None):
    """한국어 검색 딜러 전역 인스턴스 반환"""
    global _korean_search_dealer
    if _korean_search_dealer is None:
        _korean_search_dealer = KoreanSearchDealer(dataStore)
    elif dataStore is not None and hasattr(_korean_search_dealer, 'dataStore'):
        _korean_search_dealer.dataStore = dataStore
    return _korean_search_dealer

def get_korean_enhanced_queryer():
    """한국어 쿼리어 전역 인스턴스 반환"""
    global _korean_queryer
    if _korean_queryer is None:
        _korean_queryer = KoreanEnhancedQueryer()
    return _korean_queryer

# =============================================================================
# 테스트 함수들
# =============================================================================

def test_korean_search():
    """한국어 검색 기능 테스트"""
    korean_logger.info("한국어 검색 기능 테스트 시작")
    
    test_queries = [
        "자동차보험 가입방법을 알려주세요",
        "의료비 보장 범위는 어떻게 되나요?",
        "보험료 계산 방식이 궁금합니다",
        "How to apply for car insurance?",  # 영어 테스트
    ]
    
    for i, query in enumerate(test_queries, 1):
        korean_logger.info(f"\n=== 테스트 {i}: {query} ===")
        result = korean_search_query(query)
        korean_logger.info(f"결과: {result}")

def test_korean_queryer():
    """한국어 쿼리어 테스트"""
    print("한국어 쿼리어 테스트")
    
    queryer = get_korean_enhanced_queryer()
    
    test_questions = [
        "보험료는 어떻게 계산하나요?",
        "What is the insurance premium calculation method?",
        "",  # 빈 쿼리
    ]
    
    for question in test_questions:
        print(f"\n질문: {question}")
        result = queryer.question(question)
        if hasattr(result, 'value'):
            print(f"결과: {result.value} (boost: {getattr(result, 'boost', 'N/A')})")
        else:
            print(f"결과: {result}")

# =============================================================================
# 모듈 초기화
# =============================================================================

if korean_tokenizer_available:
    korean_logger.info("Korean tokenizer module loaded successfully")
else:
    korean_logger.warning("Korean tokenizer module not available")

if ragflow_available:
    korean_logger.info("RagFlow modules loaded successfully")
else:
    korean_logger.warning("RagFlow modules not available - running in standalone mode")

korean_logger.info("Korean search module initialization complete")

if __name__ == "__main__":
    test_korean_search()
    test_korean_queryer()
