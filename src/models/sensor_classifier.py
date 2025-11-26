"""
센서 데이터 기반 원두 상태 분류 모델
온도, 습도, RoR 등을 입력받아 배전도 상태를 분류합니다.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.constants import RoastLevel


class SensorDataClassifier:
    """센서 데이터 기반 원두 상태 분류 모델"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Args:
            model_type: 모델 타입 ("random_forest" 또는 "gradient_boosting")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            "bean_temp",
            "drum_temp",
            "humidity",
            "heating_power",
            "ror",
            "elapsed_time"
        ]
        
        # 배전도 레벨 매핑
        self.roast_level_mapping = {
            RoastLevel.GREEN: 0,
            RoastLevel.LIGHT: 1,
            RoastLevel.MEDIUM: 2,
            RoastLevel.MEDIUM_DARK: 3,
            RoastLevel.DARK: 4,
        }
        self.idx_to_level = {v: k for k, v in self.roast_level_mapping.items()}
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        특징 추출
        
        Args:
            df: 센서 데이터 DataFrame
            
        Returns:
            특징 배열
        """
        features = []
        
        for _, row in df.iterrows():
            feature_vector = [
                row.get("bean_temp", 0),
                row.get("drum_temp", 0),
                row.get("humidity", 0),
                row.get("heating_power", 0),
                row.get("ror", 0),
                row.get("elapsed_time", 0),
            ]
            
            # 추가 특징: 온도 차이
            feature_vector.append(row.get("drum_temp", 0) - row.get("bean_temp", 0))
            
            # 추가 특징: 평균 온도
            feature_vector.append((row.get("bean_temp", 0) + row.get("drum_temp", 0)) / 2)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_labels_from_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        데이터에서 레이블 생성 (온도 기반)
        
        Args:
            df: 센서 데이터 DataFrame
            
        Returns:
            레이블 배열
        """
        labels = []
        
        for _, row in df.iterrows():
            bean_temp = row.get("bean_temp", 0)
            
            # 온도 기반 배전도 분류
            if bean_temp < 50:
                level = RoastLevel.GREEN
            elif bean_temp < 195:
                level = RoastLevel.GREEN
            elif bean_temp < 205:
                level = RoastLevel.LIGHT
            elif bean_temp < 215:
                level = RoastLevel.MEDIUM
            elif bean_temp < 225:
                level = RoastLevel.MEDIUM_DARK
            else:
                level = RoastLevel.DARK
            
            labels.append(self.roast_level_mapping[level])
        
        return np.array(labels)
    
    def train(
        self,
        data_df: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        n_estimators: int = 100
    ) -> Dict:
        """
        모델 학습
        
        Args:
            data_df: 학습 데이터 DataFrame
            labels: 레이블 배열 (없으면 자동 생성)
            test_size: 테스트 데이터 비율
            n_estimators: 트리 개수
            
        Returns:
            학습 결과 딕셔너리
        """
        # 특징 추출
        X = self.prepare_features(data_df)
        
        # 레이블 생성
        if labels is None:
            y = self.create_labels_from_data(data_df)
        else:
            y = labels
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 특징 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 선택 및 학습
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # 평가
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # 분류 리포트
        report = classification_report(
            y_test, y_test_pred,
            target_names=[self.idx_to_level[i].value for i in sorted(self.idx_to_level.keys())],
            output_dict=True
        )
        
        print(f"학습 완료 ({self.model_type})")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\n분류 리포트:")
        print(classification_report(
            y_test, y_test_pred,
            target_names=[self.idx_to_level[i].value for i in sorted(self.idx_to_level.keys())]
        ))
        
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "classification_report": report,
            "model_type": self.model_type,
        }
    
    def predict(self, data: Dict) -> Dict:
        """
        단일 데이터 포인트 예측
        
        Args:
            data: 센서 데이터 딕셔너리
            
        Returns:
            예측 결과 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        # 특징 추출
        feature_vector = np.array([[
            data.get("bean_temp", 0),
            data.get("drum_temp", 0),
            data.get("humidity", 0),
            data.get("heating_power", 0),
            data.get("ror", 0),
            data.get("elapsed_time", 0),
            data.get("drum_temp", 0) - data.get("bean_temp", 0),
            (data.get("bean_temp", 0) + data.get("drum_temp", 0)) / 2,
        ]])
        
        # 스케일링
        feature_scaled = self.scaler.transform(feature_vector)
        
        # 예측
        predicted_idx = self.model.predict(feature_scaled)[0]
        probabilities = self.model.predict_proba(feature_scaled)[0]
        
        predicted_level = self.idx_to_level[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # 모든 클래스의 확률
        all_probs = {
            self.idx_to_level[i]: probabilities[i]
            for i in range(len(self.idx_to_level))
        }
        
        return {
            "predicted_level": predicted_level,
            "confidence": confidence,
            "all_probabilities": all_probs,
        }
    
    def save_model(self, filepath: str):
        """모델 저장"""
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "model_type": self.model_type,
                "roast_level_mapping": self.roast_level_mapping,
            }, f)
        
        print(f"모델 저장 완료: {model_path}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data["model_type"]
        self.roast_level_mapping = data["roast_level_mapping"]
        self.idx_to_level = {v: k for k, v in self.roast_level_mapping.items()}
        
        print(f"모델 로드 완료: {model_path}")

