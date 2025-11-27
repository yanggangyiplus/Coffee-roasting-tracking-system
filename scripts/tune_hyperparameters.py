"""
하이퍼파라미터 튜닝 스크립트
GridSearch 및 RandomSearch를 사용한 모델 최적화
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.sensor_classifier import SensorDataClassifier
from src.data.profile_manager import ProfileManager


def prepare_features(df: pd.DataFrame) -> tuple:
    """특징 및 레이블 준비"""
    classifier = SensorDataClassifier()
    
    X = classifier.prepare_features(df)
    y = classifier.create_labels_from_data(df)
    
    return X, y


def grid_search_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest Grid Search"""
    print("=" * 60)
    print("Random Forest Grid Search")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print("Grid Search 실행 중...")
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"\n최적 파라미터: {grid_search.best_params_}")
    print(f"최고 교차 검증 점수: {grid_search.best_score_:.4f}")
    
    # 테스트 세트 평가
    y_pred = grid_search.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"테스트 정확도: {test_acc:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def random_search_gradient_boosting(X_train, y_train, X_test, y_test):
    """Gradient Boosting Random Search"""
    print("\n" + "=" * 60)
    print("Gradient Boosting Random Search")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_distributions = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    
    gb = GradientBoostingClassifier(random_state=42)
    
    print("Random Search 실행 중...")
    random_search = RandomizedSearchCV(
        gb,
        param_distributions,
        n_iter=20,
        cv=5,
        scoring="accuracy",
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X_train_scaled, y_train)
    
    print(f"\n최적 파라미터: {random_search.best_params_}")
    print(f"최고 교차 검증 점수: {random_search.best_score_:.4f}")
    
    # 테스트 세트 평가
    y_pred = random_search.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"테스트 정확도: {test_acc:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_


def main():
    """메인 함수"""
    print("=" * 60)
    print("하이퍼파라미터 튜닝")
    print("=" * 60)
    
    # 데이터 로드
    profile_manager = ProfileManager()
    profiles_df = profile_manager.list_profiles()
    
    if len(profiles_df) == 0:
        print("저장된 프로파일이 없습니다.")
        print("먼저 프로파일을 저장하거나 샘플 데이터를 생성하세요.")
        return
    
    # 모든 프로파일 데이터 수집
    all_data = []
    for _, profile_meta in profiles_df.iterrows():
        profile = profile_manager.load_profile(profile_meta["id"])
        if profile:
            data_df = profile["data"]
            all_data.append(data_df)
    
    if len(all_data) == 0:
        print("프로파일 데이터를 찾을 수 없습니다.")
        return
    
    # 데이터 통합
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"총 데이터 포인트: {len(combined_df)}개")
    
    # 특징 및 레이블 준비
    X, y = prepare_features(combined_df)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"학습 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")
    
    # Random Forest 튜닝
    rf_best_model, rf_best_params = grid_search_random_forest(
        X_train, y_train, X_test, y_test
    )
    
    # Gradient Boosting 튜닝
    gb_best_model, gb_best_params = random_search_gradient_boosting(
        X_train, y_train, X_test, y_test
    )
    
    # 결과 저장
    print("\n" + "=" * 60)
    print("최적 파라미터 요약")
    print("=" * 60)
    print("\nRandom Forest:")
    for key, value in rf_best_params.items():
        print(f"  {key}: {value}")
    
    print("\nGradient Boosting:")
    for key, value in gb_best_params.items():
        print(f"  {key}: {value}")
    
    print("\n✅ 하이퍼파라미터 튜닝 완료!")
    print("\n최적 파라미터를 사용하여 모델을 다시 학습하세요.")


if __name__ == "__main__":
    main()

