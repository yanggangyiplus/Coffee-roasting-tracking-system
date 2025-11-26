"""
파일 로더 모듈
CSV, 엑셀 파일을 읽어서 DataFrame으로 변환합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


class FileLoader:
    """파일 로더 클래스"""
    
    @staticmethod
    def load_csv(file_path: str, encoding: str = "utf-8") -> pd.DataFrame:
        """
        CSV 파일 로드
        
        Args:
            file_path: CSV 파일 경로
            encoding: 파일 인코딩 (기본값: utf-8)
            
        Returns:
            로드된 DataFrame
        """
        try:
            # UTF-8로 시도
            df = pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            # UTF-8 실패 시 다른 인코딩 시도
            try:
                df = pd.read_csv(file_path, encoding="cp949")
            except:
                df = pd.read_csv(file_path, encoding="latin-1")
        
        return df
    
    @staticmethod
    def load_excel(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        엑셀 파일 로드
        
        Args:
            file_path: 엑셀 파일 경로
            sheet_name: 시트 이름 (없으면 첫 번째 시트)
            
        Returns:
            로드된 DataFrame
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    
    @staticmethod
    def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        컬럼 이름을 표준화합니다.
        다양한 이름을 표준 이름으로 매핑합니다.
        
        Args:
            df: 원본 DataFrame
            
        Returns:
            컬럼 이름이 표준화된 DataFrame
        """
        df = df.copy()
        
        # 컬럼 이름 매핑 (대소문자 무시)
        column_mapping = {
            # 온도 관련
            "bean_temp": ["bean_temp", "bean temperature", "원두온도", "원두 온도", "bt", "bean temp"],
            "drum_temp": ["drum_temp", "drum temperature", "드럼온도", "드럼 온도", "et", "drum temp", "env_temp", "environment temp"],
            "ambient_temp": ["ambient_temp", "ambient temperature", "주변온도", "주변 온도", "날씨온도", "날씨 온도", "outside_temp"],
            
            # 습도 관련
            "humidity": ["humidity", "습도", "rh", "relative humidity"],
            "ambient_humidity": ["ambient_humidity", "ambient humidity", "주변습도", "주변 습도", "날씨습도", "날씨 습도"],
            
            # 가열량
            "heating_power": ["heating_power", "heating power", "가열량", "power", "heat"],
            
            # 시간 관련
            "timestamp": ["timestamp", "time", "시간", "datetime", "date"],
            "elapsed_time": ["elapsed_time", "elapsed time", "경과시간", "경과 시간", "elapsed"],
            
            # 원두 색상
            "bean_color": ["bean_color", "bean color", "원두색상", "원두 색상", "color"],
        }
        
        # 소문자로 변환하여 매칭
        df.columns = df.columns.str.lower().str.strip()
        
        # 매핑 적용
        new_columns = {}
        for standard_name, variations in column_mapping.items():
            for col in df.columns:
                if col in variations or any(variation.lower() in col.lower() for variation in variations):
                    new_columns[col] = standard_name
                    break
        
        df.rename(columns=new_columns, inplace=True)
        
        return df
    
    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame 검증 및 정리
        
        Args:
            df: 원본 DataFrame
            
        Returns:
            정리된 DataFrame
        """
        df = df.copy()
        
        # 필수 컬럼 확인
        required_columns = ["bean_temp", "drum_temp", "humidity", "heating_power"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        # 숫자형 컬럼 변환
        numeric_columns = ["bean_temp", "drum_temp", "humidity", "heating_power"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 결측값 처리 (전방 채우기)
        df[numeric_columns] = df[numeric_columns].fillna(method="ffill").fillna(method="bfill")
        
        # 타임스탬프 처리
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"], errors="coerce")
        
        return df
    
    @classmethod
    def load_file(cls, file_path: str) -> pd.DataFrame:
        """
        파일을 자동으로 감지하여 로드
        
        Args:
            file_path: 파일 경로
            
        Returns:
            로드 및 정리된 DataFrame
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # 파일 확장자에 따라 로드
        suffix = path.suffix.lower()
        
        if suffix == ".csv":
            df = cls.load_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            df = cls.load_excel(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")
        
        # 컬럼 이름 표준화
        df = cls.normalize_column_names(df)
        
        # 검증 및 정리
        df = cls.validate_and_clean(df)
        
        return df

