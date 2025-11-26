"""
센서 데이터 기반 원두 상태 분류 모델 학습 스크립트
"""

import sys
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.sensor_classifier import SensorDataClassifier
from src.data.profile_manager import ProfileManager


def main():
    """메인 함수"""
    print("=" * 60)
    print("센서 데이터 기반 원두 상태 분류 모델 학습")
    print("=" * 60)
    
    # 프로파일 매니저에서 데이터 로드
    profile_manager = ProfileManager()
    profiles_df = profile_manager.list_profiles()
    
    if len(profiles_df) == 0:
        print("저장된 프로파일이 없습니다.")
        print("먼저 샘플 데이터를 생성하거나 프로파일을 저장하세요.")
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
    
    # 모델 학습
    classifier = SensorDataClassifier(model_type="random_forest")
    
    results = classifier.train(
        data_df=combined_df,
        test_size=0.2,
        n_estimators=100
    )
    
    # 모델 저장
    classifier.save_model("models/sensor_classifier/model.pkl")
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"테스트 정확도: {results['test_accuracy']:.4f}")
    print(f"모델 저장 위치: models/sensor_classifier/model.pkl")


if __name__ == "__main__":
    main()

