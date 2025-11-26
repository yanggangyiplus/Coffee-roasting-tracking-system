# 머신러닝 모델 가이드

## 개요

이 시스템은 두 가지 머신러닝 모델을 제공합니다:

1. **이미지 기반 분류 모델 (CNN)**: 원두 이미지를 입력받아 배전도 상태를 분류
2. **센서 데이터 기반 분류 모델**: 온도, 습도, RoR 등을 입력받아 배전도 상태를 분류

## 이미지 기반 분류 모델

### 모델 아키텍처
- **백본**: ResNet18 (사전 학습된 가중치 사용)
- **분류기**: 4개 클래스 (Green, Light, Medium, Dark)
- **입력 크기**: 224x224 RGB 이미지

### 데이터셋 구조
```
data/raw/data1/
├── train/
│   ├── Green/    (300개 이미지)
│   ├── Light/    (300개 이미지)
│   ├── Medium/   (300개 이미지)
│   └── Dark/     (300개 이미지)
└── test/
    ├── Green/    (100개 이미지)
    ├── Light/    (100개 이미지)
    ├── Medium/   (100개 이미지)
    └── Dark/     (100개 이미지)
```

### 모델 학습

```bash
python scripts/train_image_model.py
```

**학습 파라미터**:
- Epochs: 20
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam
- Scheduler: StepLR (7 epoch마다 학습률 감소)

**데이터 증강**:
- Random Horizontal Flip
- Random Rotation (±10도)
- Color Jitter (밝기, 대비 조정)

### 모델 사용

```python
from src.models.image_classifier import ImageClassifierPredictor

# 모델 로드
predictor = ImageClassifierPredictor("models/image_classifier/best_model.pth")

# 예측
result = predictor.predict("path/to/bean_image.png")
print(f"예측 배전도: {result['roast_level'].value}")
print(f"신뢰도: {result['confidence']*100:.1f}%")
```

## 센서 데이터 기반 분류 모델

### 모델 타입
- **Random Forest**: 기본 모델 (추천)
- **Gradient Boosting**: 대안 모델

### 입력 특징
1. 원두 온도 (bean_temp)
2. 드럼 온도 (drum_temp)
3. 습도 (humidity)
4. 가열량 (heating_power)
5. RoR (Rate of Rise)
6. 경과 시간 (elapsed_time)
7. 온도 차이 (drum_temp - bean_temp)
8. 평균 온도 ((bean_temp + drum_temp) / 2)

### 출력 클래스
- Green (생원두)
- Light (약배전)
- Medium (중배전)
- Medium Dark (중강배전)
- Dark (강배전)

### 모델 학습

**방법 1: 저장된 프로파일 데이터 사용**
```bash
python scripts/train_sensor_model.py
```

**방법 2: 프로그래밍 방식**
```python
from src.models.sensor_classifier import SensorDataClassifier
import pandas as pd

# 데이터 로드
df = pd.read_csv("roasting_data.csv")

# 모델 학습
classifier = SensorDataClassifier(model_type="random_forest")
results = classifier.train(df, test_size=0.2, n_estimators=100)

# 모델 저장
classifier.save_model("models/sensor_classifier/model.pkl")
```

### 모델 사용

```python
from src.models.sensor_classifier import SensorDataClassifier

# 모델 로드
classifier = SensorDataClassifier()
classifier.load_model("models/sensor_classifier/model.pkl")

# 예측
sensor_data = {
    "bean_temp": 200.0,
    "drum_temp": 210.0,
    "humidity": 45.0,
    "heating_power": 60.0,
    "ror": 5.0,
    "elapsed_time": 600.0,
}

result = classifier.predict(sensor_data)
print(f"예측 배전도: {result['predicted_level'].value}")
print(f"신뢰도: {result['confidence']*100:.1f}%")
```

## 대시보드에서 모델 사용

1. **머신러닝 모델 활성화**:
   - 사이드바에서 "머신러닝 모델 사용" 체크박스 선택
   - 모델 파일이 있으면 자동으로 로드

2. **예측 결과 확인**:
   - 배전도 메트릭에 "(ML)" 표시 및 신뢰도 표시
   - 머신러닝 모델이 사용되면 더 정확한 예측 제공

## 모델 성능 향상 팁

### 이미지 모델
1. **더 많은 데이터**: 각 클래스당 최소 500개 이상 권장
2. **데이터 품질**: 일관된 조명 조건, 배경 제거
3. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 조정
4. **전이 학습**: 더 큰 모델 (ResNet50, EfficientNet) 사용

### 센서 데이터 모델
1. **더 많은 프로파일**: 다양한 원두 종류, 환경 조건 포함
2. **특징 엔지니어링**: 추가 특징 (온도 변화율, 습도 변화율 등)
3. **앙상블**: 여러 모델 조합
4. **하이퍼파라미터 튜닝**: 트리 개수, 깊이 조정

## 모델 비교

| 모델 | 장점 | 단점 | 사용 시기 |
|------|------|------|----------|
| **규칙 기반** | 빠름, 해석 가능 | 정확도 낮음 | 초기 단계, 빠른 프로토타이핑 |
| **센서 데이터 ML** | 정확도 높음, 실시간 예측 | 학습 데이터 필요 | 충분한 데이터 수집 후 |
| **이미지 ML** | 시각적 검증 가능 | 이미지 필요, 처리 시간 | 원두 색상 확인 가능할 때 |

## 모델 저장 위치

- 이미지 모델: `models/image_classifier/best_model.pth`
- 센서 데이터 모델: `models/sensor_classifier/model.pkl`

## 문제 해결

### 모델이 로드되지 않는 경우
1. 모델 파일 경로 확인
2. 모델 파일이 존재하는지 확인
3. 필요한 라이브러리 설치 확인 (`torch`, `torchvision`)

### 예측 정확도가 낮은 경우
1. 더 많은 학습 데이터 수집
2. 데이터 품질 확인
3. 하이퍼파라미터 튜닝
4. 모델 아키텍처 변경 고려

