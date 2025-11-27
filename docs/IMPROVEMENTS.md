# 개선 사항 및 향후 계획

## ✅ 구현 완료된 개선 사항

### 1. 다중 프로파일 비교 분석 강화

**구현 내용**:
- 2개 이상 프로파일 동시 선택 UI
- 겹쳐진 그래프 시각화 (온도 곡선, RoR 곡선)
- 통계적 비교 (평균 RoR, 총 시간, 온도 상승률 등)
- DTW 기반 유사도 계산 및 히트맵 시각화

**사용 방법**:
1. 대시보드에서 "프로파일 관리" → "프로파일 비교" 탭 선택
2. 비교할 프로파일 2개 이상 선택
3. 통계 비교 테이블, 유사도 행렬, 곡선 비교 그래프 확인

**파일 위치**:
- `src/data/profile_manager.py`: `compare_profiles()`, `calculate_similarity()`, `calculate_statistics()`
- `app/main.py`: `show_profile_comparison()`

### 2. 데이터 증강 기능

**이미지 데이터 증강**:
- 회전, 플립, 밝기/대비 조정
- 색상 조정 (Hue, Saturation, Value)
- 가우시안 노이즈, 블러 추가
- Albumentations 라이브러리 사용

**센서 데이터 증강**:
- 노이즈 추가 (가우시안 노이즈)
- 시간 스케일 변형 (시간 축 늘리기/줄이기)
- 보간을 통한 데이터 포인트 증가
- 여러 증강 기법 조합

**사용 방법**:
```bash
# 이미지 데이터 증강
python scripts/augment_data.py --type image

# 센서 데이터 증강
python scripts/augment_data.py --type sensor

# 모두 증강
python scripts/augment_data.py --type all
```

**파일 위치**:
- `src/data/data_augmentation.py`: `ImageAugmenter`, `SensorDataAugmenter`
- `scripts/augment_data.py`: 증강 실행 스크립트

### 3. 하이퍼파라미터 튜닝

**구현 내용**:
- Random Forest Grid Search
- Gradient Boosting Random Search
- 5-fold Cross Validation
- 최적 파라미터 자동 탐색

**사용 방법**:
```bash
python scripts/tune_hyperparameters.py
```

**튜닝 파라미터**:
- Random Forest: n_estimators, max_depth, min_samples_split, min_samples_leaf
- Gradient Boosting: n_estimators, max_depth, learning_rate, min_samples_split, min_samples_leaf

**파일 위치**:
- `scripts/tune_hyperparameters.py`: 하이퍼파라미터 튜닝 스크립트

## 📋 향후 계획

### 1. 실제 센서 하드웨어 연동 (Arduino/Raspberry Pi)

**필요한 하드웨어**:
- Arduino Uno/Mega 또는 Raspberry Pi 3/4
- MAX6675 K-Type 열전대 모듈 x2 (원두/드럼 온도)
- DHT22 습도 센서
- USB 케이블, 점퍼 케이블, 브레드보드

**필요한 라이브러리**:
- `pyserial`: Arduino 시리얼 통신
- `board`, `busio`, `adafruit_dht`: Raspberry Pi GPIO 제어

**구현 예정**:
- `src/data/sensor_stream.py`의 `RealSensorStream` 클래스 구현
- Arduino 스케치 코드 (.ino 파일)
- Python 시리얼 리더 클래스

### 2. LSTM/Transformer 시계열 예측 모델

**필요한 데이터**:
- 대량의 로스팅 프로파일 데이터 (최소 100개 이상)
- 시간별 온도/습도/RoR 시계열 데이터
- 레이블 데이터: 각 시점의 배전도 레벨, 크랙 발생 시점

**필요한 라이브러리**:
- `torch>=2.0.0`
- `transformers>=4.30.0` (Hugging Face Transformers)

**구현 예정 파일**:
- `src/models/time_series_lstm.py`: LSTM 모델
- `src/models/time_series_transformer.py`: Transformer 모델
- `scripts/train_time_series_model.py`: 학습 스크립트

### 3. 모델 성능 개선 (더 많은 학습 데이터)

**데이터 수집 전략**:
- 실제 로스팅 데이터 수집
- 사용자가 직접 로스팅하며 프로파일 저장
- 카페/로스터리와 협업
- 공개 데이터셋 활용

**데이터 증강 강화**:
- 더 다양한 증강 기법 추가
- 현실적인 온도 곡선, RoR 패턴 생성
- 도메인 지식 기반 증강

## 🎯 우선순위

1. ✅ **즉시 시작 가능** (완료):
   - 다중 프로파일 비교 분석
   - 데이터 증강 기능
   - 하이퍼파라미터 튜닝

2. **중기 계획** (데이터 수집 필요):
   - LSTM/Transformer 모델 (프로파일 100개+ 수집 후)

3. **장기 계획** (하드웨어 필요):
   - 실제 센서 연동 (하드웨어 구매 및 테스트 필요)

## 📊 현재 시스템 성능

### 센서 데이터 분류 모델
- **기본 정확도**: 규칙 기반 약 70-80%
- **ML 모델 정확도**: 학습 데이터에 따라 85-95% (예상)
- **개선 방향**: 더 많은 데이터 수집 및 하이퍼파라미터 튜닝

### 이미지 분류 모델
- **기본 정확도**: ResNet18 사전 학습 모델 기준
- **예상 정확도**: 충분한 데이터로 90% 이상 가능
- **개선 방향**: 데이터 증강 및 전이 학습 최적화

## 💡 사용 팁

1. **데이터 수집**: 가능한 한 많은 프로파일을 저장하세요
2. **데이터 증강**: 작은 데이터셋으로 시작할 때 증강 기능 활용
3. **모델 비교**: 규칙 기반과 ML 모델 결과를 비교하여 선택
4. **프로파일 비교**: 유사한 조건의 프로파일을 비교하여 패턴 파악

