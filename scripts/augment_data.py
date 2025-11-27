"""
데이터 증강 스크립트
이미지 및 센서 데이터 증강 실행
"""

import sys
import pandas as pd
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_augmentation import (
    ImageAugmenter,
    SensorDataAugmenter,
    augment_image_dataset,
    augment_sensor_data
)
from src.data.profile_manager import ProfileManager


def augment_images():
    """이미지 데이터셋 증강"""
    print("=" * 60)
    print("이미지 데이터셋 증강")
    print("=" * 60)
    
    input_dir = "data/raw/data1/train"
    output_dir = "data/augmented/images"
    
    if not Path(input_dir).exists():
        print(f"입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return
    
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"각 이미지당 3개 증강 생성")
    
    try:
        augment_image_dataset(input_dir, output_dir, num_augmentations=3)
        print("\n✅ 이미지 증강 완료!")
    except ImportError as e:
        print(f"❌ 오류: {e}")
        print("albumentations 라이브러리를 설치하세요: pip install albumentations")


def augment_sensor_profiles():
    """센서 데이터 프로파일 증강"""
    print("=" * 60)
    print("센서 데이터 프로파일 증강")
    print("=" * 60)
    
    profile_manager = ProfileManager()
    profiles_df = profile_manager.list_profiles()
    
    if len(profiles_df) == 0:
        print("저장된 프로파일이 없습니다.")
        print("먼저 프로파일을 저장하거나 샘플 데이터를 생성하세요.")
        return
    
    print(f"총 {len(profiles_df)}개 프로파일 발견")
    
    augmenter = SensorDataAugmenter()
    augmented_count = 0
    
    for _, profile_meta in profiles_df.iterrows():
        profile = profile_manager.load_profile(profile_meta["id"])
        if profile:
            data_df = profile["data"]
            
            # 증강 생성
            augmented_profiles = augmenter.augment_profile(
                data_df,
                num_augmentations=2,
                noise_level=0.02
            )
            
            # 증강된 프로파일 저장
            for i, aug_df in enumerate(augmented_profiles):
                profile_name = f"{profile_meta['profile_name']}_aug_{i}"
                profile_manager.save_profile(
                    profile_name=profile_name,
                    data_df=aug_df,
                    bean_type=profile_meta["bean_type"],
                    target_level=None,  # 원본과 동일
                    notes=f"증강 데이터 (원본 ID: {profile_meta['id']})"
                )
                augmented_count += 1
    
    print(f"\n✅ {augmented_count}개 증강 프로파일 생성 완료!")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="데이터 증강 스크립트")
    parser.add_argument(
        "--type",
        choices=["image", "sensor", "all"],
        default="all",
        help="증강할 데이터 타입"
    )
    
    args = parser.parse_args()
    
    if args.type in ["image", "all"]:
        augment_images()
        print()
    
    if args.type in ["sensor", "all"]:
        augment_sensor_profiles()
        print()


if __name__ == "__main__":
    main()

