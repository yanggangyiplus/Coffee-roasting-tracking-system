"""
로스팅 프로파일 저장 및 관리 모듈
"""

import pandas as pd
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from src.utils.constants import RoastLevel, RoastingStage

# DTW 알고리즘 (scipy 사용)
try:
    from scipy.spatial.distance import euclidean
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ProfileManager:
    """로스팅 프로파일 관리 클래스"""
    
    def __init__(self, db_path: str = "data/processed/roasting_profiles.db"):
        """
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 프로파일 메타데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_name TEXT NOT NULL,
                bean_type TEXT,
                target_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_time_seconds REAL,
                final_temp REAL,
                notes TEXT
            )
        """)
        
        # 센서 데이터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profile_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER,
                timestamp TIMESTAMP,
                bean_temp REAL,
                drum_temp REAL,
                humidity REAL,
                heating_power REAL,
                ror REAL,
                stage TEXT,
                elapsed_time REAL,
                FOREIGN KEY (profile_id) REFERENCES profiles(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_profile(
        self,
        profile_name: str,
        data_df: pd.DataFrame,
        bean_type: Optional[str] = None,
        target_level: Optional[RoastLevel] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        로스팅 프로파일 저장
        
        Args:
            profile_name: 프로파일 이름
            data_df: 센서 데이터 DataFrame
            bean_type: 원두 종류
            target_level: 목표 배전도
            notes: 메모
            
        Returns:
            저장된 프로파일 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 최종 온도 및 총 시간 계산
        final_temp = data_df["bean_temp"].iloc[-1] if len(data_df) > 0 else 0
        total_time = data_df["elapsed_time"].iloc[-1] if len(data_df) > 0 else 0
        
        # 프로파일 메타데이터 저장
        cursor.execute("""
            INSERT INTO profiles 
            (profile_name, bean_type, target_level, total_time_seconds, final_temp, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            profile_name,
            bean_type,
            target_level.value if target_level else None,
            total_time,
            final_temp,
            notes
        ))
        
        profile_id = cursor.lastrowid
        
        # 센서 데이터 저장
        for _, row in data_df.iterrows():
            cursor.execute("""
                INSERT INTO profile_data
                (profile_id, timestamp, bean_temp, drum_temp, humidity, 
                 heating_power, ror, stage, elapsed_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile_id,
                row.get("timestamp", datetime.now()),
                row.get("bean_temp", 0),
                row.get("drum_temp", 0),
                row.get("humidity", 0),
                row.get("heating_power", 0),
                row.get("ror", 0),
                row.get("stage", ""),
                row.get("elapsed_time", 0),
            ))
        
        conn.commit()
        conn.close()
        
        return profile_id
    
    def load_profile(self, profile_id: int) -> Dict:
        """
        프로파일 로드
        
        Args:
            profile_id: 프로파일 ID
            
        Returns:
            프로파일 딕셔너리 (메타데이터 + 데이터)
        """
        conn = sqlite3.connect(self.db_path)
        
        # 메타데이터 로드
        meta_df = pd.read_sql_query(
            "SELECT * FROM profiles WHERE id = ?",
            conn,
            params=(profile_id,)
        )
        
        if meta_df.empty:
            conn.close()
            return None
        
        # 센서 데이터 로드
        data_df = pd.read_sql_query(
            "SELECT * FROM profile_data WHERE profile_id = ? ORDER BY elapsed_time",
            conn,
            params=(profile_id,)
        )
        
        conn.close()
        
        return {
            "metadata": meta_df.iloc[0].to_dict(),
            "data": data_df
        }
    
    def list_profiles(
        self,
        bean_type: Optional[str] = None,
        target_level: Optional[RoastLevel] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        프로파일 목록 조회 (필터링 지원)
        
        Args:
            bean_type: 원두 종류 필터
            target_level: 배전도 필터
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            프로파일 목록 DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM profiles WHERE 1=1"
        params = []
        
        if bean_type:
            query += " AND bean_type = ?"
            params.append(bean_type)
        
        if target_level:
            query += " AND target_level = ?"
            params.append(target_level.value)
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)
        
        query += " ORDER BY created_at DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def delete_profile(self, profile_id: int) -> bool:
        """
        프로파일 삭제
        
        Args:
            profile_id: 프로파일 ID
            
        Returns:
            삭제 성공 여부
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 센서 데이터 먼저 삭제
        cursor.execute("DELETE FROM profile_data WHERE profile_id = ?", (profile_id,))
        
        # 프로파일 메타데이터 삭제
        cursor.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        
        return deleted
    
    def calculate_similarity(self, profile1: Dict, profile2: Dict) -> Dict:
        """
        두 프로파일 간 유사도 계산 (DTW 기반)
        
        Args:
            profile1: 첫 번째 프로파일 딕셔너리
            profile2: 두 번째 프로파일 딕셔너리
            
        Returns:
            유사도 정보 딕셔너리
        """
        data1 = profile1["data"]
        data2 = profile2["data"]
        
        # 시간 정규화 (0-1 범위로)
        time1 = data1["elapsed_time"].values
        time2 = data2["elapsed_time"].values
        
        if len(time1) == 0 or len(time2) == 0:
            return {"similarity": 0.0, "dtw_distance": float('inf')}
        
        # 정규화
        time1_norm = time1 / time1.max() if time1.max() > 0 else time1
        time2_norm = time2 / time2.max() if time2.max() > 0 else time2
        
        # 온도 곡선 정규화
        temp1 = data1["bean_temp"].values
        temp2 = data2["bean_temp"].values
        
        temp1_norm = (temp1 - temp1.min()) / (temp1.max() - temp1.min() + 1e-10)
        temp2_norm = (temp2 - temp2.min()) / (temp2.max() - temp2.min() + 1e-10)
        
        # 동일한 시간 간격으로 보간
        common_time = np.linspace(0, 1, max(len(time1), len(time2)))
        
        if SCIPY_AVAILABLE:
            try:
                f1 = interp1d(time1_norm, temp1_norm, kind='linear', fill_value='extrapolate')
                f2 = interp1d(time2_norm, temp2_norm, kind='linear', fill_value='extrapolate')
                
                temp1_interp = f1(common_time)
                temp2_interp = f2(common_time)
                
                # DTW 거리 계산 (간단한 버전)
                dtw_distance = self._dtw_distance(temp1_interp, temp2_interp)
                
                # 유사도 (0-1 범위, 1이 가장 유사)
                max_distance = np.sqrt(len(common_time)) * (temp1_interp.max() - temp1_interp.min())
                similarity = 1.0 / (1.0 + dtw_distance / (max_distance + 1e-10))
                
            except Exception as e:
                # 보간 실패 시 유클리드 거리 사용
                dtw_distance = np.linalg.norm(temp1_norm[:len(temp2_norm)] - temp2_norm[:len(temp1_norm)])
                similarity = 1.0 / (1.0 + dtw_distance)
        else:
            # scipy가 없으면 간단한 유클리드 거리
            min_len = min(len(temp1_norm), len(temp2_norm))
            dtw_distance = np.linalg.norm(temp1_norm[:min_len] - temp2_norm[:min_len])
            similarity = 1.0 / (1.0 + dtw_distance)
        
        return {
            "similarity": float(similarity),
            "dtw_distance": float(dtw_distance),
        }
    
    def _dtw_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Dynamic Time Warping 거리 계산 (간단한 구현)
        
        Args:
            x: 첫 번째 시계열
            y: 두 번째 시계열
            
        Returns:
            DTW 거리
        """
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(x[i-1] - y[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )
        
        return dtw_matrix[n, m]
    
    def calculate_statistics(self, profile: Dict) -> Dict:
        """
        프로파일 통계 정보 계산
        
        Args:
            profile: 프로파일 딕셔너리
            
        Returns:
            통계 정보 딕셔너리
        """
        data = profile["data"]
        
        if len(data) == 0:
            return {}
        
        stats = {
            "total_time_minutes": data["elapsed_time"].max() / 60.0,
            "initial_temp": data["bean_temp"].iloc[0],
            "final_temp": data["bean_temp"].iloc[-1],
            "max_temp": data["bean_temp"].max(),
            "avg_temp": data["bean_temp"].mean(),
            "avg_ror": data["ror"].mean(),
            "max_ror": data["ror"].max(),
            "min_ror": data["ror"].min(),
            "avg_humidity": data["humidity"].mean() if "humidity" in data.columns else None,
            "avg_heating_power": data["heating_power"].mean() if "heating_power" in data.columns else None,
            "temp_rise_rate": (data["bean_temp"].iloc[-1] - data["bean_temp"].iloc[0]) / (data["elapsed_time"].max() / 60.0) if data["elapsed_time"].max() > 0 else 0,
        }
        
        return stats
    
    def compare_profiles(self, profile_ids: List[int]) -> Dict:
        """
        여러 프로파일 비교 (강화된 버전)
        
        Args:
            profile_ids: 비교할 프로파일 ID 리스트
            
        Returns:
            비교 결과 딕셔너리
        """
        profiles = []
        for pid in profile_ids:
            profile = self.load_profile(pid)
            if profile:
                profiles.append(profile)
        
        if len(profiles) < 2:
            return {"error": "비교하려면 최소 2개의 프로파일이 필요합니다."}
        
        comparison = {
            "profiles": [],
            "temperature_curves": [],
            "ror_curves": [],
            "statistics": [],
            "similarity_matrix": [],
        }
        
        # 프로파일 정보 수집
        for profile in profiles:
            metadata = profile["metadata"]
            data = profile["data"]
            
            # 기본 정보
            profile_info = {
                "id": metadata["id"],
                "name": metadata["profile_name"],
                "bean_type": metadata["bean_type"],
                "target_level": metadata["target_level"],
                "total_time": metadata["total_time_seconds"],
                "final_temp": metadata["final_temp"],
            }
            comparison["profiles"].append(profile_info)
            
            # 통계 정보
            stats = self.calculate_statistics(profile)
            comparison["statistics"].append(stats)
            
            # 곡선 데이터
            comparison["temperature_curves"].append({
                "name": metadata["profile_name"],
                "time": data["elapsed_time"].tolist(),
                "temp": data["bean_temp"].tolist(),
            })
            
            comparison["ror_curves"].append({
                "name": metadata["profile_name"],
                "time": data["elapsed_time"].tolist(),
                "ror": data["ror"].tolist(),
            })
        
        # 유사도 행렬 계산
        similarity_matrix = []
        for i, profile1 in enumerate(profiles):
            row = []
            for j, profile2 in enumerate(profiles):
                if i == j:
                    similarity = 1.0
                else:
                    sim_info = self.calculate_similarity(profile1, profile2)
                    similarity = sim_info["similarity"]
                row.append(similarity)
            similarity_matrix.append(row)
        
        comparison["similarity_matrix"] = similarity_matrix
        
        return comparison

