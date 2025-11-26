# 사용 예제

## 기본 사용법

### 1. 프로그래밍 방식으로 데이터 처리

```python
from src.data.processor import SensorDataProcessor
from src.algorithms.stage_detector import RoastingStageDetector
from src.prediction.roast_predictor import RoastLevelPredictor
from src.utils.constants import RoastLevel
from datetime import datetime

# 데이터 프로세서 초기화
processor = SensorDataProcessor()
stage_detector = RoastingStageDetector()
predictor = RoastLevelPredictor()

# 센서 데이터 추가
data_point = processor.add_data_point(
    bean_temp=150.0,
    drum_temp=165.0,
    humidity=45.0,
    heating_power=70.0
)

# 로스팅 단계 감지
current_stage = stage_detector.detect_stage(
    bean_temp=data_point["bean_temp"],
    drum_temp=data_point["drum_temp"],
    humidity=data_point["humidity"],
    ror=data_point["ror"],
    elapsed_time=data_point["elapsed_time"],
    heating_power=data_point["heating_power"]
)

print(f"현재 단계: {current_stage.value}")
print(f"RoR: {data_point['ror']:.2f}°C/분")

# 목표 배전도 예측
prediction = predictor.predict_time_to_target(
    current_temp=data_point["bean_temp"],
    current_ror=data_point["ror"],
    target_level=RoastLevel.MEDIUM,
    elapsed_time=data_point["elapsed_time"]
)

print(f"예상 시간: {prediction['estimated_time_minutes']:.1f}분")
print(f"진행률: {prediction['progress_percent']:.1f}%")
```

### 2. 프로파일 저장 및 로드

```python
from src.data.profile_manager import ProfileManager
import pandas as pd

# 프로파일 매니저 초기화
profile_manager = ProfileManager()

# 데이터프레임 준비 (센서 데이터)
df = processor.get_dataframe()

# 프로파일 저장
profile_id = profile_manager.save_profile(
    profile_name="에티오피아_약배전_20240101",
    data_df=df,
    bean_type="에티오피아 예가체프",
    target_level=RoastLevel.LIGHT,
    notes="첫 번째 로스팅 시도"
)

print(f"프로파일 저장 완료: ID {profile_id}")

# 프로파일 로드
profile = profile_manager.load_profile(profile_id)
print(f"프로파일 이름: {profile['metadata']['profile_name']}")
print(f"총 시간: {profile['metadata']['total_time_seconds']/60:.1f}분")

# 프로파일 목록 조회
profiles_df = profile_manager.list_profiles(
    bean_type="에티오피아 예가체프",
    target_level=RoastLevel.LIGHT
)
print(profiles_df)
```

### 3. CSV 파일에서 데이터 로드

```python
import pandas as pd
from src.data.processor import SensorDataProcessor

# CSV 파일 읽기
df = pd.read_csv("data/raw/sample_roasting.csv")

# 프로세서에 로드
processor = SensorDataProcessor()
processor.load_from_dataframe(df)

# 처리된 데이터 가져오기
processed_df = processor.get_dataframe()
print(processed_df.head())
```

### 4. 여러 프로파일 비교

```python
from src.data.profile_manager import ProfileManager

profile_manager = ProfileManager()

# 비교할 프로파일 ID 리스트
profile_ids = [1, 2, 3]

# 프로파일 비교
comparison = profile_manager.compare_profiles(profile_ids)

# 온도 곡선 비교
for curve in comparison["temperature_curves"]:
    print(f"{curve['name']}: 최종 온도 {max(curve['temp']):.1f}°C")
```

## Streamlit 대시보드 사용법

### 대시보드 실행

```bash
streamlit run app/main.py
```

### 로스팅 프로세스

1. **시작 설정**
   - 프로파일 이름 입력
   - 원두 종류 입력
   - 목표 배전도 선택

2. **데이터 입력**
   - 센서 데이터 입력 폼에 값 입력
   - "데이터 추가" 버튼 클릭
   - 실시간으로 그래프 업데이트 확인

3. **모니터링**
   - 현재 로스팅 단계 확인
   - 온도 및 RoR 그래프 모니터링
   - 목표 배전도 도달 예상 시간 확인

4. **저장**
   - 로스팅 완료 후 "프로파일 저장" 클릭
   - 프로파일이 데이터베이스에 저장됨

## 샘플 데이터 생성

```bash
python scripts/generate_sample_data.py
```

이 스크립트는 다음 샘플 프로파일을 생성합니다:
- 에티오피아 예가체프 (약배전, 11분)
- 콜롬비아 수프리모 (중배전, 12.5분)
- 브라질 산토스 (중강배전, 13분)
- 인도네시아 만델링 (강배전, 14분)

생성된 데이터는 다음 위치에 저장됩니다:
- CSV 파일: `data/raw/`
- 데이터베이스: `data/processed/roasting_profiles.db`

## 고급 사용법

### 커스텀 온도 임계값 설정

```python
from src.utils.constants import TEMPERATURE_THRESHOLDS

# 임계값 수정
TEMPERATURE_THRESHOLDS["first_crack_min"] = 195
TEMPERATURE_THRESHOLDS["first_crack_max"] = 210
```

### RoR 계산 윈도우 조정

```python
# 더 긴 시간 윈도우 사용 (더 부드러운 RoR)
processor = SensorDataProcessor(time_window=60)

# 더 짧은 시간 윈도우 사용 (더 빠른 반응)
processor = SensorDataProcessor(time_window=15)
```

### 배치 데이터 처리

```python
import pandas as pd
from src.data.processor import SensorDataProcessor
from src.algorithms.stage_detector import RoastingStageDetector

# 배치 데이터 로드
df = pd.read_csv("batch_roasting_data.csv")

# 프로세서 초기화
processor = SensorDataProcessor()
processor.load_from_dataframe(df)

# 전체 데이터에 대해 단계 감지
stage_detector = RoastingStageDetector()
stages = []

for _, row in processor.get_dataframe().iterrows():
    stage = stage_detector.detect_stage(
        bean_temp=row["bean_temp"],
        drum_temp=row["drum_temp"],
        humidity=row["humidity"],
        ror=row["ror"],
        elapsed_time=row["elapsed_time"],
        heating_power=row["heating_power"]
    )
    stages.append(stage.value)

# 결과 확인
print(f"감지된 단계: {set(stages)}")
```

## 문제 해결

### 데이터가 표시되지 않는 경우

1. 데이터프레임이 비어있는지 확인:
```python
df = processor.get_dataframe()
print(f"데이터 포인트 수: {len(df)}")
```

2. 타임스탬프가 올바른지 확인:
```python
print(df["timestamp"].head())
```

### 단계가 감지되지 않는 경우

1. 온도가 임계값에 도달했는지 확인:
```python
print(f"현재 온도: {bean_temp}°C")
print(f"1차 크랙 범위: {TEMPERATURE_THRESHOLDS['first_crack_min']}-{TEMPERATURE_THRESHOLDS['first_crack_max']}°C")
```

2. RoR 패턴 확인:
```python
print(f"현재 RoR: {ror}°C/분")
print(f"RoR 히스토리: {[point['ror'] for point in stage_detector.stage_history[-5:]]}")
```

### 예측이 부정확한 경우

1. RoR이 너무 낮거나 높은지 확인:
```python
if current_ror < 0.5:
    print("RoR이 너무 낮습니다. 예측이 부정확할 수 있습니다.")
```

2. 충분한 데이터 포인트가 있는지 확인:
```python
if len(processor.data_history) < 10:
    print("데이터 포인트가 부족합니다. 더 많은 데이터를 수집하세요.")
```

