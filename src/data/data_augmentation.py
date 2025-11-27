"""
데이터 증강 모듈
이미지 및 센서 데이터 증강 기능 제공
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image

# 이미지 증강
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    ToTensorV2 = None

# 시계열 증강
try:
    import tsaug
    TSAUG_AVAILABLE = True
except ImportError:
    TSAUG_AVAILABLE = False
    tsaug = None


class ImageAugmenter:
    """이미지 데이터 증강 클래스"""
    
    def __init__(self):
        """초기화"""
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError("albumentations 라이브러리가 필요합니다: pip install albumentations")
        
        # 증강 파이프라인 정의
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.val_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def augment_image(self, image_path: str, save_path: Optional[str] = None) -> Image.Image:
        """
        이미지 증강
        
        Args:
            image_path: 원본 이미지 경로
            save_path: 저장 경로 (선택사항)
            
        Returns:
            증강된 이미지
        """
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)
        
        # 증강 적용
        augmented = self.train_transform(image=image_array)
        augmented_image = Image.fromarray(augmented["image"])
        
        if save_path:
            augmented_image.save(save_path)
        
        return augmented_image
    
    def augment_dataset(
        self,
        input_dir: str,
        output_dir: str,
        num_augmentations: int = 3
    ):
        """
        데이터셋 전체 증강
        
        Args:
            input_dir: 입력 디렉토리
            output_dir: 출력 디렉토리
            num_augmentations: 각 이미지당 증강 개수
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 클래스별 디렉토리 순회
        for class_dir in input_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # 이미지 파일 처리
            image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            
            for img_file in image_files:
                # 원본 복사
                import shutil
                shutil.copy(img_file, output_class_dir / img_file.name)
                
                # 증강 생성
                for i in range(num_augmentations):
                    aug_img = self.augment_image(str(img_file))
                    aug_filename = f"{img_file.stem}_aug_{i}{img_file.suffix}"
                    aug_img.save(output_class_dir / aug_filename)
            
            print(f"{class_name}: {len(image_files)}개 이미지 → {len(image_files) * (num_augmentations + 1)}개 생성")


class SensorDataAugmenter:
    """센서 데이터 증강 클래스"""
    
    def __init__(self):
        """초기화"""
        pass
    
    def add_noise(
        self,
        df: pd.DataFrame,
        noise_level: float = 0.02,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        노이즈 추가
        
        Args:
            df: 원본 데이터프레임
            noise_level: 노이즈 레벨 (표준편차 비율)
            columns: 노이즈를 추가할 컬럼 리스트 (없으면 모든 숫자 컬럼)
            
        Returns:
            증강된 데이터프레임
        """
        df_aug = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # timestamp, elapsed_time 제외
            columns = [c for c in columns if c not in ["timestamp", "elapsed_time"]]
        
        for col in columns:
            if col in df_aug.columns:
                std = df_aug[col].std()
                noise = np.random.normal(0, std * noise_level, len(df_aug))
                df_aug[col] = df_aug[col] + noise
        
        return df_aug
    
    def time_scale(
        self,
        df: pd.DataFrame,
        scale_factor: float = 1.1
    ) -> pd.DataFrame:
        """
        시간 스케일 변형 (시간 축 늘리기/줄이기)
        
        Args:
            df: 원본 데이터프레임
            scale_factor: 스케일 팩터 (>1이면 늘어남, <1이면 줄어듦)
            
        Returns:
            증강된 데이터프레임
        """
        df_aug = df.copy()
        
        if "elapsed_time" in df_aug.columns:
            df_aug["elapsed_time"] = df_aug["elapsed_time"] * scale_factor
        
        return df_aug
    
    def interpolate(
        self,
        df: pd.DataFrame,
        target_length: Optional[int] = None,
        factor: float = 1.5
    ) -> pd.DataFrame:
        """
        보간을 통한 데이터 포인트 증가
        
        Args:
            df: 원본 데이터프레임
            target_length: 목표 길이 (없으면 factor 기반)
            factor: 길이 증가 팩터
            
        Returns:
            증강된 데이터프레임
        """
        if target_length is None:
            target_length = int(len(df) * factor)
        
        # 시간 컬럼 확인
        if "elapsed_time" not in df.columns:
            df_aug = df.copy()
            df_aug["elapsed_time"] = np.arange(len(df))
        
        # 새로운 시간 축 생성
        original_time = df["elapsed_time"].values
        new_time = np.linspace(original_time.min(), original_time.max(), target_length)
        
        # 보간
        df_aug = pd.DataFrame({"elapsed_time": new_time})
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_columns:
            if col != "elapsed_time":
                from scipy.interpolate import interp1d
                f = interp1d(original_time, df[col].values, kind='linear', fill_value='extrapolate')
                df_aug[col] = f(new_time)
        
        # 비숫자 컬럼은 전방 채우기
        for col in df.columns:
            if col not in numeric_columns and col != "elapsed_time":
                df_aug[col] = df[col].iloc[0]
        
        return df_aug
    
    def augment_profile(
        self,
        df: pd.DataFrame,
        num_augmentations: int = 3,
        noise_level: float = 0.02
    ) -> List[pd.DataFrame]:
        """
        프로파일 데이터 증강 (여러 변형 생성)
        
        Args:
            df: 원본 데이터프레임
            num_augmentations: 생성할 증강 개수
            noise_level: 노이즈 레벨
            
        Returns:
            증강된 데이터프레임 리스트
        """
        augmented_profiles = []
        
        for i in range(num_augmentations):
            df_aug = df.copy()
            
            # 랜덤하게 증강 기법 선택
            aug_type = np.random.choice(["noise", "time_scale", "both"])
            
            if aug_type in ["noise", "both"]:
                df_aug = self.add_noise(df_aug, noise_level=noise_level)
            
            if aug_type in ["time_scale", "both"]:
                scale_factor = np.random.uniform(0.9, 1.1)
                df_aug = self.time_scale(df_aug, scale_factor=scale_factor)
            
            augmented_profiles.append(df_aug)
        
        return augmented_profiles


def augment_image_dataset(
    input_dir: str,
    output_dir: str,
    num_augmentations: int = 3
):
    """
    이미지 데이터셋 증강 헬퍼 함수
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        num_augmentations: 각 이미지당 증강 개수
    """
    augmenter = ImageAugmenter()
    augmenter.augment_dataset(input_dir, output_dir, num_augmentations)


def augment_sensor_data(
    df: pd.DataFrame,
    num_augmentations: int = 3,
    noise_level: float = 0.02
) -> List[pd.DataFrame]:
    """
    센서 데이터 증강 헬퍼 함수
    
    Args:
        df: 원본 데이터프레임
        num_augmentations: 생성할 증강 개수
        noise_level: 노이즈 레벨
        
    Returns:
        증강된 데이터프레임 리스트
    """
    augmenter = SensorDataAugmenter()
    return augmenter.augment_profile(df, num_augmentations, noise_level)

