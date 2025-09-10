#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Korean Query Processor for RAGFlow"""

import logging
import re
from .korean_utils import (
    contains_korean, is_korean_char, get_custom_dict_path, 
    load_dict_scores, get_pos_scores, get_excluded_pos_tags, 
    should_exclude_token, korean_logger, CUSTOM_DICT_PATH, 
    POS_SCORES, EXCLUDED_POS_TAGS
)

from .korean_tokenizer import KiwiManager, get_filtered_tokens_with_compounds


class KoreanQueryProcessor:
    """한국어 전용 쿼리 처리기 - 공통 유틸리티 사용"""
    
    def __init__(self,stop_words=None): # stop_words: 금지어 삭제
        # KiwiManager 싱글톤 사용
        self.kiwi_manager = KiwiManager()
        self.kiwi = self.kiwi_manager.kiwi
        
        # 공통 유틸리티에서 가져온 설정들 사용
        self.dict_scores = {}
        self.pos_scores = get_pos_scores()
        self.excluded_pos = get_excluded_pos_tags()
        self.special_words = stop_words or set() # 금지어 주어지지 않을 시 공백 유지
        self.special_char_patterns = [
            r"[!@#$%%^&*()_+\-=\[\]{}|\\:\";'<>?,./\`~]" # 단순 키보드만을 이용해 칠 수 있는 특수기호
        ]
        self.replacement_rules =[] # 치환
        
        if self.kiwi:
            # 커스텀 사전 점수 로드
            dict_path = get_custom_dict_path()
            if dict_path:
                self.dict_scores = load_dict_scores(dict_path)
                korean_logger.info(f"[Query] Korean dictionary scores loaded: {len(self.dict_scores)} entries")
            
            korean_logger.info("[Query] Korean query processor initialized with shared KiwiManager")
        else:
            korean_logger.warning("[Query] Korean query processor initialized without Kiwi")
    
    @staticmethod
    def is_korean_text(text):
        """텍스트가 한국어인지 감지 - 공통 유틸리티 사용"""
        return contains_korean(text)
    
    def clean_special_chars(self,text):
        """특수기호를 \s로 바꿈 ex. 마음@에이아이 -> 마음 에이아이"""
        if not text:
            return text
        cleaned_text = text
        for pattern in self.special_char_patterns:
            cleaned_text = re.sub(pattern, " ", cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text
    
    def preprocess_text(self,txt, include_numbers=False, filter_stopwords=True):
        if not txt:
            return ""
        for pattern, replacement in self.replacement_rules:
            txt = re.sub(pattern, replacement,txt) # 1. 치환 수행
        cleaned_text = self.clean_special_chars(txt)
        
        if not cleaned_text.strip():
            return "" 
        
        if filter_stopwords or not include_numbers:
            tokens = cleaned_text.split()
            filtered_tokens = []
            
            for token in tokens:
                if filter_stopwords and token in self.stop_words:
                    continue # 조건 부여시 금지어 안 모음
                if re.match(r"^[0-9]+$", token) and not include_numbers:
                    continue # 조건 부여시 숫자 안 모음
                if token.strip():
                    filtered_tokens.append(token) # 나머지는 모음
            return " ".join(filtered_tokens)

    def should_exclude_token(self, token, min_length=1):
        """토큰을 제외해야 하는지 판단 - 공통 유틸리티 사용"""
        return should_exclude_token(token, min_length)
    
    def tokenize_korean(self, text):
        """한국어 형태소 분석 및 토크나이징 - 공통 함수 사용"""
        if not text:
            return [(text, 0.3)]
        preprocessed_text = self.preprocess_text(text, include_numbers=False, filter_stopwords=False)
        
        if not preprocessed_text.strip():
            return [(text,0.3)]
    
        tokens = get_filtered_tokens_with_compounds(text, min_length=1, min_boost=0.0)
        
        filtered_tokens=[]
        for token_info in tokens:
            token = token_info["token"]
            boost = token_info["boost"]
            
            if len(token.strip())>0:
                cleaned_token = self.clean_special_chars(token)
                if cleaned_token.strip():
                    filtered_tokens.append((cleaned_token.strip(), boost))
        return filtered_tokens if filtered_tokens else [(preprocessed_text, 0.3)]
            
    def _basic_korean_tokenize(self, text):
        """기본 한국어 토크나이징 - KiwiManager fallback 사용"""
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text.strip():
            preprocessed_text = text
        return self.kiwi_manager.basic_tokenize(text)
    
    def build_korean_query(self, text, min_match=0.3): # 0.6
        """한국어 텍스트를 검색 쿼리로 변환 - 가중치 기반"""
        weighted_tokens = self.tokenize_korean(text)
        
        if not weighted_tokens:
            clean_text = self.clean_special_chars(text)
            
            return clean_text if clean_text.strip() else text, [clean_text] if clean_text.strip() else [text]
        
        # 가중치 순으로 정렬 (높은 가중치가 먼저)
        weighted_tokens.sort(key=lambda x: x[1], reverse=True)
        
        query_parts = []
        keywords = []
        
        korean_logger.debug(f"[TOKENIZER] Raw weighted tokens: {weighted_tokens}")
        
        for token, weight in weighted_tokens:
            if len(token.strip()) < 2:
                continue

            keywords.append(token)
            
            # 가중치에 따른 쿼리 생성 (더 세밀한 구분)
            if weight >= 2.5:
                # 매우 높은 가중치 (사용자 사전 + 높은 점수): 강력한 매치
                token_query = f'("{token}"^{weight:.2f} OR {token}^{weight*0.9:.2f})'
                # 높은 가중치 단어는 여러 번 포함
                query_parts.extend([token_query] * 2)
            elif weight >= 1.5:
                # 높은 가중치: 완전매치 + 부분매치
                token_query = f'("{token}"^{weight:.2f} OR {token}^{weight*0.8:.2f})'
                query_parts.append(token_query)
            elif weight >= 1.0:
                # 중간 가중치: 기본 매치
                token_query = f'{token}^{weight:.2f}'
                query_parts.append(token_query)
            else:
                # 낮은 가중치: 약한 매치 (조사, 어미 등)
                token_query = f'{token}^{weight:.2f}'
                query_parts.append(token_query)
        
        if not query_parts:
            clean_text = self.clean_special_chars(text)
            return clean_text if clean_text.strip() else text, [clean_text] if clean_text.strip() else [text]

        # 쿼리 조합 (OR 연결)
        query = " OR ".join(query_parts)
        
        # 전체 구문 매치 추가 (사용자 사전 단어가 많을 때 더 높은 가중치)
        dict_word_count = sum(1 for _, w in weighted_tokens if w >= 1.5)
        if len(weighted_tokens) > 1:
            phrase_weight = sum(w for _, w in weighted_tokens if w >= 1.0) * (0.3 + dict_word_count * 0.1)
            query = f'("{text}"^{phrase_weight:.2f}) OR ({query})'
        
        korean_logger.info(f" [QUERY] Final query: {query}")
        korean_logger.info(f" [QUERY] Keywords: {keywords}")
        korean_logger.debug(f" [QUERY] Dict word count: {dict_word_count}, phrase weight: {phrase_weight:.2f}")
        
        return query, keywords
