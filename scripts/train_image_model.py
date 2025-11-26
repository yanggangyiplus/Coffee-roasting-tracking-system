"""
이미지 기반 원두 상태 분류 모델 학습 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.image_classifier import ImageClassifierTrainer


def main():
    """메인 함수"""
    print("=" * 60)
    print("이미지 기반 원두 상태 분류 모델 학습")
    print("=" * 60)
    
    # 데이터 디렉토리 경로
    data_dir = "data/raw/data1"
    
    # 학습기 초기화
    trainer = ImageClassifierTrainer(model_dir="models/image_classifier")
    
    # 모델 학습
    history = trainer.train(
        train_data_dir=data_dir,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        save_best=True
    )
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"최종 검증 정확도: {history['val_acc'][-1]:.2f}%")
    print(f"모델 저장 위치: models/image_classifier/")


if __name__ == "__main__":
    main()

