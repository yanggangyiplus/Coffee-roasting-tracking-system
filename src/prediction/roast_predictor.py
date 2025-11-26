"""
목표 배전도 도달 시간 예측 모듈
현재 상태를 기반으로 목표 배전도에 도달하는 시간을 예측합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

from src.utils.constants import RoastLevel, ROAST_LEVEL_TEMPERATURES


class RoastLevelPredictor:
    """배전도 도달 시간 예측 클래스"""
    
    def __init__(self):
        """초기화"""
        self.temperature_history: list = []
        self.time_history: list = []
    
    def predict_time_to_target(
        self,
        current_temp: float,
        current_ror: float,
        target_level: RoastLevel,
        elapsed_time: float
    ) -> Dict:
        """
        목표 배전도 도달까지 예상 시간 계산
        
        Args:
            current_temp: 현재 원두 온도
            current_ror: 현재 RoR (도/분)
            target_level: 목표 배전도 레벨
            elapsed_time: 현재까지 경과 시간 (초)
            
        Returns:
            예측 결과 딕셔너리
        """
        # 목표 온도 범위 가져오기
        target_temp_min, target_temp_max = ROAST_LEVEL_TEMPERATURES[target_level]
        target_temp = (target_temp_min + target_temp_max) / 2  # 평균값 사용
        
        # 현재 온도가 이미 목표 온도 이상인 경우
        if current_temp >= target_temp_max:
            return {
                "target_reached": True,
                "estimated_time_seconds": 0,
                "estimated_time_minutes": 0,
                "progress_percent": 100,
                "target_temp": target_temp,
                "current_temp": current_temp,
            }
        
        # 온도 차이 계산
        temp_diff = target_temp - current_temp
        
        # RoR이 0이거나 너무 작은 경우 기본값 사용
        if current_ror <= 0:
            current_ror = 1.0  # 기본 RoR (도/분)
        
        # 간단한 선형 예측: 남은 온도 / 현재 RoR
        estimated_minutes = temp_diff / current_ror
        
        # RoR 감소를 고려한 보정 (시간이 지날수록 RoR이 감소함)
        # 지수 감쇠 모델 적용
        if len(self.temperature_history) >= 3:
            # 최근 RoR 추세 분석
            recent_rors = [
                (self.temperature_history[i] - self.temperature_history[i-1]) / 
                ((self.time_history[i] - self.time_history[i-1]) / 60.0)
                for i in range(1, len(self.temperature_history))
            ]
            
            if recent_rors:
                avg_ror = np.mean(recent_rors)
                if avg_ror < current_ror:
                    # RoR이 감소 추세이면 예상 시간을 늘림
                    estimated_minutes *= 1.2
        
        estimated_seconds = estimated_minutes * 60
        
        # 진행률 계산 (현재 온도 기준)
        temp_range = target_temp_max - current_temp
        if temp_range > 0:
            progress = min(100, max(0, (current_temp - (target_temp - temp_range)) / temp_range * 100))
        else:
            progress = 100
        
        # 히스토리 업데이트
        self.temperature_history.append(current_temp)
        self.time_history.append(elapsed_time)
        
        # 히스토리 크기 제한 (최근 50개만 유지)
        if len(self.temperature_history) > 50:
            self.temperature_history = self.temperature_history[-50:]
            self.time_history = self.time_history[-50:]
        
        return {
            "target_reached": False,
            "estimated_time_seconds": round(estimated_seconds, 1),
            "estimated_time_minutes": round(estimated_minutes, 2),
            "progress_percent": round(progress, 1),
            "target_temp": target_temp,
            "current_temp": current_temp,
            "temp_diff": round(temp_diff, 1),
            "current_ror": current_ror,
        }
    
    def check_target_reached(
        self,
        current_temp: float,
        target_level: RoastLevel
    ) -> bool:
        """
        목표 배전도 도달 여부 확인
        
        Args:
            current_temp: 현재 원두 온도
            target_level: 목표 배전도 레벨
            
        Returns:
            목표 도달 여부
        """
        target_temp_min, target_temp_max = ROAST_LEVEL_TEMPERATURES[target_level]
        return current_temp >= target_temp_min
    
    def reset(self):
        """예측 상태 초기화"""
        self.temperature_history = []
        self.time_history = []

