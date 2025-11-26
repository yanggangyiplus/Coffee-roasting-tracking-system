# 시스템 아키텍처 문서

## 개요

커피 로스팅 추적 시스템은 센서 데이터를 실시간으로 수집, 처리, 분석하여 로스팅 단계를 추적하고 목표 배전도 도달 시간을 예측하는 데이터 기반 시스템입니다.

## 시스템 구조

### 계층 구조

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│      (Streamlit Dashboard)              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Application Layer                │
│  - Stage Detection                      │
│  - Roast Level Prediction               │
│  - Profile Management                   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Data Processing Layer          │
│  - Sensor Data Processing              │
│  - RoR Calculation                     │
│  - Data Validation                     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Data Storage Layer             │
│  - SQLite Database                     │
│  - CSV Export                           │
└────────────────────────────────────────┘
```

## 핵심 모듈

### 1. SensorDataProcessor (`src/data/processor.py`)

**책임**: 센서 데이터 수집 및 RoR 계산

**주요 기능**:
- 센서 데이터 포인트 추가
- Rate of Rise (RoR) 계산
- 데이터 히스토리 관리
- DataFrame 변환

**입력**:
- Bean Temperature (원두 온도)
- Drum Temperature (드럼 온도)
- Humidity (습도)
- Heating Power (가열량)
- Timestamp (타임스탬프)

**출력**:
- 처리된 데이터 포인트 (RoR 포함)
- DataFrame 형식의 데이터

### 2. RoastingStageDetector (`src/algorithms/stage_detector.py`)

**책임**: 로스팅 단계 자동 감지

**주요 기능**:
- 건조 단계 감지
- 갈변 단계 감지
- 1차 크랙 감지
- 발열/발화 구간 감지
- 2차 크랙 감지

**알고리즘**:
- 온도 임계값 기반 단계 분류
- RoR 패턴 분석을 통한 크랙 감지
- 히스토리 기반 패턴 매칭

**출력**:
- 현재 로스팅 단계 (RoastingStage enum)
- 단계 변경 히스토리

### 3. RoastLevelPredictor (`src/prediction/roast_predictor.py`)

**책임**: 목표 배전도 도달 시간 예측

**주요 기능**:
- 목표 배전도 도달 시간 예측
- 진행률 계산
- 목표 도달 여부 확인

**예측 모델**:
- 선형 예측: `예상시간 = 온도차이 / 현재 RoR`
- RoR 감쇠 보정: 지수 감쇠 모델 적용
- 히스토리 기반 추세 분석

**출력**:
- 예상 도달 시간 (초, 분)
- 진행률 (%)
- 목표 도달 여부

### 4. ProfileManager (`src/data/profile_manager.py`)

**책임**: 로스팅 프로파일 저장 및 관리

**주요 기능**:
- 프로파일 저장 (SQLite)
- 프로파일 로드
- 프로파일 목록 조회 (필터링 지원)
- 프로파일 삭제
- 프로파일 비교

**데이터베이스 스키마**:
- `profiles` 테이블: 프로파일 메타데이터
- `profile_data` 테이블: 센서 데이터

## 데이터 흐름

### 실시간 데이터 처리 흐름

```
센서 데이터 입력
    │
    ▼
SensorDataProcessor.add_data_point()
    │
    ├─► 타임스탬프 추가
    ├─► RoR 계산
    └─► 데이터 히스토리 업데이트
    │
    ▼
RoastingStageDetector.detect_stage()
    │
    ├─► 온도 임계값 체크
    ├─► RoR 패턴 분석
    └─► 단계 감지
    │
    ▼
RoastLevelPredictor.predict_time_to_target()
    │
    ├─► 목표 온도 계산
    ├─► 예상 시간 계산
    └─► 진행률 계산
    │
    ▼
대시보드 시각화
```

### 프로파일 저장 흐름

```
로스팅 완료
    │
    ▼
ProfileManager.save_profile()
    │
    ├─► 메타데이터 저장 (profiles 테이블)
    └─► 센서 데이터 저장 (profile_data 테이블)
    │
    ▼
SQLite Database
```

## 알고리즘 상세

### RoR 계산 알고리즘

```python
def calculate_ror(current_temp, prev_temp, time_diff_seconds):
    temp_diff = current_temp - prev_temp
    time_diff_minutes = time_diff_seconds / 60.0
    ror = temp_diff / time_diff_minutes
    return ror  # 단위: 도/분
```

### 1차 크랙 감지 알고리즘

```python
def detect_first_crack(bean_temp, ror, history):
    # 조건 1: 온도 범위 체크
    if not (190 <= bean_temp <= 205):
        return False
    
    # 조건 2: RoR 급감 패턴 체크
    if len(history) >= 3:
        recent_rors = [point.ror for point in history[-3:]]
        ror_drop = recent_rors[0] - recent_rors[-1]
        if ror_drop >= 5:  # 5도/분 이상 감소
            return True
    
    return False
```

### 배전도 도달 예측 알고리즘

```python
def predict_time_to_target(current_temp, target_temp, current_ror):
    # 기본 선형 예측
    temp_diff = target_temp - current_temp
    estimated_minutes = temp_diff / current_ror
    
    # RoR 감쇠 보정
    if ror_decreasing_trend:
        estimated_minutes *= 1.2  # 20% 증가
    
    return estimated_minutes
```

## 데이터베이스 스키마

### profiles 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | Primary Key |
| profile_name | TEXT | 프로파일 이름 |
| bean_type | TEXT | 원두 종류 |
| target_level | TEXT | 목표 배전도 |
| created_at | TIMESTAMP | 생성 시간 |
| total_time_seconds | REAL | 총 로스팅 시간 (초) |
| final_temp | REAL | 최종 온도 |
| notes | TEXT | 메모 |

### profile_data 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | Primary Key |
| profile_id | INTEGER | Foreign Key (profiles.id) |
| timestamp | TIMESTAMP | 타임스탬프 |
| bean_temp | REAL | 원두 온도 |
| drum_temp | REAL | 드럼 온도 |
| humidity | REAL | 습도 |
| heating_power | REAL | 가열량 |
| ror | REAL | Rate of Rise |
| stage | TEXT | 로스팅 단계 |
| elapsed_time | REAL | 경과 시간 (초) |

## 확장 가능성

### 향후 개선 사항

1. **실제 센서 연동**
   - Arduino/Raspberry Pi 기반 센서 하드웨어 연동
   - 실시간 데이터 스트리밍

2. **머신러닝 모델 통합**
   - LSTM 기반 시계열 예측
   - 딥러닝 기반 단계 분류 모델

3. **클라우드 배포**
   - Streamlit Cloud 배포
   - AWS/GCP 기반 서버리스 아키텍처

4. **모바일 앱**
   - React Native 기반 모바일 앱
   - 푸시 알림 기능

5. **고급 분석**
   - 다중 프로파일 비교 분석
   - 통계적 분석 및 리포트 생성

## 성능 고려사항

- **데이터 처리**: 실시간 데이터 처리 최적화
- **메모리 관리**: 히스토리 크기 제한 (최근 50개 포인트)
- **데이터베이스**: SQLite 인덱싱 최적화
- **시각화**: Plotly 기반 효율적인 그래프 렌더링

