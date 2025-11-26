"""
로스팅 프로파일 저장 및 관리 모듈
"""

import pandas as pd
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.utils.constants import RoastLevel, RoastingStage


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
    
    def compare_profiles(self, profile_ids: List[int]) -> Dict:
        """
        여러 프로파일 비교
        
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
        }
        
        for profile in profiles:
            metadata = profile["metadata"]
            data = profile["data"]
            
            comparison["profiles"].append({
                "id": metadata["id"],
                "name": metadata["profile_name"],
                "bean_type": metadata["bean_type"],
                "target_level": metadata["target_level"],
                "total_time": metadata["total_time_seconds"],
                "final_temp": metadata["final_temp"],
            })
            
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
        
        return comparison

