"""
센서 데이터 처리 모듈
온도, 습도, 가열량 데이터를 처리하고 RoR을 계산합니다.
"""

"""
센서 데이터 처리 모듈
온도, 습도, 가열량 데이터를 처리하고 RoR을 계산합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class SensorDataProcessor:
    """센서 데이터 처리 클래스"""
    
    def __init__(self, time_window: int = 30):
        """
        Args:
            time_window: RoR 계산을 위한 시간 윈도우 (초)
        """
        self.time_window = time_window
        self.data_history: List[Dict] = []
    
    def add_data_point(
        self,
        bean_temp: float,
        drum_temp: float,
        humidity: float,
        heating_power: float,
        timestamp: Optional[datetime] = None,
        ambient_temp: Optional[float] = None,
        ambient_humidity: Optional[float] = None,
        bean_color: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        새로운 센서 데이터 포인트 추가 및 RoR 계산
        
        Args:
            bean_temp: 원두 온도 (섭씨)
            drum_temp: 드럼 온도 (섭씨)
            humidity: 습도 (%)
            heating_power: 가열량 (%)
            timestamp: 타임스탬프 (없으면 현재 시간)
            ambient_temp: 주변 온도 (섭씨, 선택사항)
            ambient_humidity: 주변 습도 (%, 선택사항)
            bean_color: 원두 색상 (선택사항)
            **kwargs: 추가 필드들
            
        Returns:
            처리된 데이터 딕셔너리
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 이전 데이터와 비교하여 RoR 계산
        ror = self._calculate_ror(bean_temp)
        
        data_point = {
            "timestamp": timestamp,
            "bean_temp": bean_temp,
            "drum_temp": drum_temp,
            "humidity": humidity,
            "heating_power": heating_power,
            "ror": ror,
            "elapsed_time": self._calculate_elapsed_time(timestamp),
        }
        
        # 추가 필드들 추가
        if ambient_temp is not None:
            data_point["ambient_temp"] = ambient_temp
        if ambient_humidity is not None:
            data_point["ambient_humidity"] = ambient_humidity
        if bean_color is not None:
            data_point["bean_color"] = bean_color
        
        # 기타 추가 필드들
        for key, value in kwargs.items():
            data_point[key] = value
        
        self.data_history.append(data_point)
        return data_point
    
    def _calculate_ror(self, current_temp: float) -> float:
        """
        Rate of Rise (RoR) 계산
        현재 온도와 이전 온도의 차이를 시간 차이로 나눈 값
        
        Args:
            current_temp: 현재 원두 온도
            
        Returns:
            RoR (도/분)
        """
        if len(self.data_history) < 2:
            return 0.0
        
        # 최근 두 포인트 사용
        prev_point = self.data_history[-1]
        prev_temp = prev_point["bean_temp"]
        prev_time = prev_point["timestamp"]
        
        current_time = datetime.now()
        time_diff_seconds = (current_time - prev_time).total_seconds()
        
        if time_diff_seconds == 0:
            return 0.0
        
        temp_diff = current_temp - prev_temp
        time_diff_minutes = time_diff_seconds / 60.0
        
        # RoR 계산 (도/분)
        ror = temp_diff / time_diff_minutes if time_diff_minutes > 0 else 0.0
        
        return round(ror, 2)
    
    def _calculate_elapsed_time(self, timestamp: datetime) -> float:
        """
        경과 시간 계산 (초)
        
        Args:
            timestamp: 현재 타임스탬프
            
        Returns:
            경과 시간 (초)
        """
        if len(self.data_history) == 0:
            return 0.0
        
        start_time = self.data_history[0]["timestamp"]
        elapsed = (timestamp - start_time).total_seconds()
        return round(elapsed, 1)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        현재까지 수집된 데이터를 DataFrame으로 반환
        
        Returns:
            센서 데이터 DataFrame
        """
        if not self.data_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.data_history)
        return df
    
    def reset(self):
        """데이터 히스토리 초기화"""
        self.data_history = []
    
    def load_from_dataframe(self, df: pd.DataFrame):
        """
        DataFrame에서 데이터 로드
        
        Args:
            df: 센서 데이터가 포함된 DataFrame
        """
        self.reset()
        
        required_columns = ["bean_temp", "drum_temp", "humidity", "heating_power"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame에 필수 컬럼이 없습니다: {required_columns}")
        
        # timestamp 컬럼이 없으면 생성
        if "timestamp" not in df.columns:
            if "time" in df.columns:
                # time 컬럼이 있으면 datetime으로 변환 시도
                try:
                    df["timestamp"] = pd.to_datetime(df["time"])
                except:
                    df["timestamp"] = pd.date_range(
                        start=datetime.now() - timedelta(seconds=len(df)),
                        periods=len(df),
                        freq="S"
                    )
            else:
                df["timestamp"] = pd.date_range(
                    start=datetime.now() - timedelta(seconds=len(df)),
                    periods=len(df),
                    freq="S"
                )
        
        # RoR 계산을 위해 순차적으로 추가
        for idx, row in df.iterrows():
            # 추가 필드 추출
            extra_fields = {}
            if "ambient_temp" in df.columns:
                extra_fields["ambient_temp"] = row.get("ambient_temp")
            if "ambient_humidity" in df.columns:
                extra_fields["ambient_humidity"] = row.get("ambient_humidity")
            if "bean_color" in df.columns:
                extra_fields["bean_color"] = row.get("bean_color")
            
            # 기타 모든 컬럼을 추가 필드로 포함
            for col in df.columns:
                if col not in ["bean_temp", "drum_temp", "humidity", "heating_power", "timestamp", "time"]:
                    if col not in extra_fields:
                        extra_fields[col] = row.get(col)
            
            self.add_data_point(
                bean_temp=row["bean_temp"],
                drum_temp=row["drum_temp"],
                humidity=row["humidity"],
                heating_power=row["heating_power"],
                timestamp=row["timestamp"],
                **extra_fields
            )

