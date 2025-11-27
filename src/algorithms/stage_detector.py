"""
로스팅 단계 감지 알고리즘
온도, RoR, 습도 등을 기반으로 현재 로스팅 단계를 감지합니다.
"""

import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime

from src.utils.constants import (
    RoastingStage,
    RoastLevel,
    BeanColor,
    TEMPERATURE_THRESHOLDS,
    ROR_THRESHOLDS,
    HUMIDITY_THRESHOLDS,
    GREEN_BEAN_THRESHOLDS,
    BEAN_COLOR_TEMPERATURES,
)

# 모델 기반 분류기 (선택적)
try:
    from src.models.sensor_classifier import SensorDataClassifier
    from src.models.image_classifier import ImageClassifierPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    SensorDataClassifier = None
    ImageClassifierPredictor = None


class RoastingStageDetector:
    """로스팅 단계 감지 클래스"""
    
    def __init__(self, use_ml_model: bool = False, sensor_model_path: Optional[str] = None):
        """
        Args:
            use_ml_model: 머신러닝 모델 사용 여부
            sensor_model_path: 센서 데이터 분류 모델 경로
        """
        self.stage_history: List[Dict] = []
        self.first_crack_detected = False
        self.second_crack_detected = False
        
        # 머신러닝 모델 (선택적)
        self.use_ml_model = use_ml_model and ML_AVAILABLE
        self.sensor_classifier = None
        
        if self.use_ml_model and sensor_model_path:
            try:
                self.sensor_classifier = SensorDataClassifier()
                self.sensor_classifier.load_model(sensor_model_path)
                print(f"머신러닝 모델 로드 완료: {sensor_model_path}")
            except Exception as e:
                print(f"머신러닝 모델 로드 실패: {e}")
                self.use_ml_model = False
    
    def detect_roast_level(
        self,
        bean_temp: float,
        bean_color: Optional[str] = None,
        sensor_data: Optional[Dict] = None
    ) -> Tuple[RoastLevel, Dict]:
        """
        배전도 레벨 감지 (생원두 포함)
        머신러닝 모델이 있으면 모델 사용, 없으면 규칙 기반 사용
        
        Args:
            bean_temp: 원두 온도
            bean_color: 원두 색상 (선택사항)
            sensor_data: 전체 센서 데이터 (머신러닝 모델 사용 시 필요)
            
        Returns:
            (배전도 레벨, 예측 정보 딕셔너리)
        """
        # 머신러닝 모델 사용
        if self.use_ml_model and self.sensor_classifier and sensor_data:
            try:
                prediction = self.sensor_classifier.predict(sensor_data)
                return prediction["predicted_level"], {
                    "method": "ml_model",
                    "confidence": prediction["confidence"],
                    "all_probabilities": prediction["all_probabilities"]
                }
            except Exception as e:
                print(f"머신러닝 모델 예측 실패: {e}, 규칙 기반으로 전환")
        
        # 규칙 기반 감지 (기본)
        # 생원두 감지
        if bean_temp <= GREEN_BEAN_THRESHOLDS["max_temp"]:
            return RoastLevel.GREEN, {"method": "rule_based"}
        
        # 온도 기반 배전도 감지
        if bean_temp < 195:
            level = RoastLevel.GREEN  # 아직 로스팅 시작 전
        elif bean_temp < 205:
            level = RoastLevel.LIGHT
        elif bean_temp < 215:
            level = RoastLevel.MEDIUM
        elif bean_temp < 225:
            level = RoastLevel.MEDIUM_DARK
        else:
            level = RoastLevel.DARK
        
        return level, {"method": "rule_based"}
    
    def detect_bean_color(self, bean_temp: float) -> BeanColor:
        """
        원두 색상 감지 (온도 기반)
        
        Args:
            bean_temp: 원두 온도
            
        Returns:
            원두 색상
        """
        if bean_temp < 50:
            return BeanColor.GREEN
        elif bean_temp < 100:
            return BeanColor.YELLOW
        elif bean_temp < 150:
            return BeanColor.LIGHT_BROWN
        elif bean_temp < 190:
            return BeanColor.BROWN
        elif bean_temp < 220:
            return BeanColor.DARK_BROWN
        else:
            return BeanColor.VERY_DARK
    
    def detect_stage(
        self,
        bean_temp: float,
        drum_temp: float,
        humidity: float,
        ror: float,
        elapsed_time: float,
        heating_power: float
    ) -> RoastingStage:
        """
        현재 로스팅 단계 감지
        
        Args:
            bean_temp: 원두 온도
            drum_temp: 드럼 온도
            humidity: 습도
            ror: Rate of Rise
            elapsed_time: 경과 시간 (초)
            heating_power: 가열량
            
        Returns:
            감지된 로스팅 단계
        """
        # 생원두 단계 (온도가 매우 낮은 경우)
        if bean_temp < GREEN_BEAN_THRESHOLDS["max_temp"]:
            stage = RoastingStage.DRYING  # 건조 단계로 분류하되, 배전도는 GREEN으로 표시
        
        # 2차 크랙 감지 (가장 높은 우선순위)
        elif self._detect_second_crack(bean_temp, ror):
            stage = RoastingStage.SECOND_CRACK
            self.second_crack_detected = True
        
        # 1차 크랙 감지
        elif self._detect_first_crack(bean_temp, ror):
            stage = RoastingStage.FIRST_CRACK
            self.first_crack_detected = True
        
        # 발열/발화 구간 (1차 크랙 이후 ~ 2차 크랙 이전)
        elif self.first_crack_detected and not self.second_crack_detected:
            stage = RoastingStage.DEVELOPMENT
        
        # 갈변 단계
        elif bean_temp >= TEMPERATURE_THRESHOLDS["maillard_start"]:
            stage = RoastingStage.MAILLARD
        
        # 건조 단계
        elif bean_temp >= TEMPERATURE_THRESHOLDS["drying_start"]:
            stage = RoastingStage.DRYING
        
        # 초기 단계
        else:
            stage = RoastingStage.DRYING
        
        # 단계 히스토리 기록
        self.stage_history.append({
            "timestamp": datetime.now(),
            "stage": stage,
            "bean_temp": bean_temp,
            "ror": ror,
            "elapsed_time": elapsed_time,
        })
        
        return stage
    
    def _detect_first_crack(
        self,
        bean_temp: float,
        ror: float
    ) -> bool:
        """
        1차 크랙 감지 로직
        
        Args:
            bean_temp: 원두 온도
            ror: Rate of Rise
            
        Returns:
            1차 크랙 감지 여부
        """
        # 이미 감지된 경우 재감지하지 않음
        if self.first_crack_detected:
            return False
        
        # 온도 범위 체크
        temp_min = TEMPERATURE_THRESHOLDS["first_crack_min"]
        temp_max = TEMPERATURE_THRESHOLDS["first_crack_max"]
        
        if not (temp_min <= bean_temp <= temp_max):
            return False
        
        # RoR 급감 패턴 체크 (최근 히스토리 확인)
        if len(self.stage_history) >= 3:
            recent_rors = [
                point["ror"] for point in self.stage_history[-3:]
                if "ror" in point
            ]
            
            if len(recent_rors) >= 2:
                # RoR이 급격히 감소하는 패턴 감지
                ror_drop = recent_rors[0] - recent_rors[-1]
                if ror_drop >= ROR_THRESHOLDS["first_crack_drop"]:
                    return True
        
        return False
    
    def _detect_second_crack(
        self,
        bean_temp: float,
        ror: float
    ) -> bool:
        """
        2차 크랙 감지 로직
        
        Args:
            bean_temp: 원두 온도
            ror: Rate of Rise
            
        Returns:
            2차 크랙 감지 여부
        """
        # 1차 크랙이 먼저 감지되어야 함
        if not self.first_crack_detected:
            return False
        
        # 이미 감지된 경우 재감지하지 않음
        if self.second_crack_detected:
            return False
        
        # 온도 범위 체크
        temp_min = TEMPERATURE_THRESHOLDS["second_crack_min"]
        temp_max = TEMPERATURE_THRESHOLDS["second_crack_max"]
        
        if not (temp_min <= bean_temp <= temp_max):
            return False
        
        # RoR 급감 패턴 체크
        if len(self.stage_history) >= 3:
            recent_rors = [
                point["ror"] for point in self.stage_history[-3:]
                if "ror" in point
            ]
            
            if len(recent_rors) >= 2:
                ror_drop = recent_rors[0] - recent_rors[-1]
                if ror_drop >= ROR_THRESHOLDS["second_crack_drop"]:
                    return True
        
        return False
    
    def get_current_stage_info(self) -> Optional[Dict]:
        """
        현재 단계 정보 반환
        
        Returns:
            최근 단계 정보 딕셔너리
        """
        if not self.stage_history:
            return None
        
        return self.stage_history[-1]
    
    def reset(self):
        """감지 상태 초기화"""
        self.stage_history = []
        self.first_crack_detected = False
        self.second_crack_detected = False

