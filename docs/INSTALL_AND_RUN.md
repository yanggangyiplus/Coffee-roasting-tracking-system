# 설치 및 실행 가이드

## 필수 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)

## 설치 단계

### 1. 저장소 클론

```bash
git clone <repository-url>
cd Coffee-roasting-tracking-system
```

### 2. 가상환경 생성 및 활성화

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 샘플 데이터 생성 (선택사항)

```bash
python scripts/generate_sample_data.py
```

이 스크립트는 다음 샘플 프로파일을 생성합니다:
- 에티오피아 예가체프 (약배전)
- 콜롬비아 수프리모 (중배전)
- 브라질 산토스 (중강배전)
- 인도네시아 만델링 (강배전)

## 실행 방법

### Streamlit 대시보드 실행

**방법 1: 직접 실행**
```bash
streamlit run app/main.py
```

**방법 2: 스크립트 사용 (macOS/Linux)**
```bash
bash scripts/run_dashboard.sh
```

브라우저에서 자동으로 열리거나, `http://localhost:8501`로 접속하세요.

## 사용 방법

### 1. 로스팅 시작

1. 사이드바에서 프로파일 이름 입력
2. 원두 종류 입력
3. 목표 배전도 선택 (약배전/중배전/중강배전/강배전)
4. "로스팅 시작" 버튼 클릭

### 2. 센서 데이터 입력

1. "센서 데이터 입력" 섹션에서 다음 정보 입력:
   - 원두 온도 (°C)
   - 드럼 온도 (°C)
   - 습도 (%)
   - 가열량 (%)
2. "데이터 추가" 버튼 클릭
3. 실시간으로 그래프가 업데이트됩니다

### 3. 로스팅 모니터링

- 현재 로스팅 단계 확인
- 온도 및 RoR 그래프 확인
- 목표 배전도 도달 예상 시간 확인
- 진행률 확인

### 4. 프로파일 저장

1. 로스팅 완료 후 "프로파일 저장" 버튼 클릭
2. 프로파일이 SQLite 데이터베이스에 저장됩니다

### 5. 프로파일 관리

1. 사이드바에서 "프로파일 목록 보기" 클릭
2. 저장된 프로파일 목록 확인
3. 프로파일 선택하여 상세 정보 및 그래프 확인
4. 필요시 프로파일 삭제

## 문제 해결

### 포트가 이미 사용 중인 경우

다른 포트로 실행:
```bash
streamlit run app/main.py --server.port 8502
```

### 모듈을 찾을 수 없는 경우

프로젝트 루트 디렉토리에서 실행했는지 확인하세요:
```bash
pwd  # 현재 디렉토리 확인
cd /path/to/Coffee-roasting-tracking-system  # 프로젝트 루트로 이동
```

### 데이터베이스 오류

데이터베이스 파일이 손상된 경우:
```bash
rm data/processed/roasting_profiles.db
# 앱을 다시 실행하면 자동으로 재생성됩니다
```

## 추가 설정

### 설정 파일 수정

`configs/config.yaml` 파일을 수정하여 다음 설정을 변경할 수 있습니다:
- 온도 임계값
- RoR 임계값
- 습도 임계값
- 데이터베이스 경로
- 대시보드 새로고침 간격

## 개발 모드

개발 중 코드 변경사항을 자동으로 반영하려면:
```bash
streamlit run app/main.py --server.runOnSave true
```

