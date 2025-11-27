# 문제 해결 가이드

## ModuleNotFoundError: No module named 'src'

### 원인
Streamlit 실행 시 Python 경로가 제대로 설정되지 않아 발생하는 문제입니다.

### 해결 방법

#### 방법 1: 프로젝트 루트에서 실행 (권장)

```bash
# 프로젝트 루트 디렉토리로 이동
cd /Users/yanggangyi/Desktop/PORTFOLIO_PROJECT/Coffee-roasting-tracking-system

# Streamlit 실행
streamlit run app/main.py
```

#### 방법 2: 실행 스크립트 사용

```bash
# 실행 스크립트 사용 (자동으로 경로 설정)
bash scripts/run_dashboard.sh
```

#### 방법 3: PYTHONPATH 설정

```bash
# macOS/Linux
export PYTHONPATH="${PYTHONPATH}:/Users/yanggangyi/Desktop/PORTFOLIO_PROJECT/Coffee-roasting-tracking-system"
streamlit run app/main.py

# Windows
set PYTHONPATH=%PYTHONPATH%;C:\path\to\Coffee-roasting-tracking-system
streamlit run app/main.py
```

#### 방법 4: 패키지 설치 (개발 모드)

```bash
# 프로젝트 루트에서
pip install -e .
```

이렇게 하면 프로젝트가 패키지로 설치되어 어디서든 import 가능합니다.

### 확인 방법

다음 명령어로 경로가 제대로 설정되었는지 확인할 수 있습니다:

```python
import sys
print(sys.path)
```

프로젝트 루트 경로가 포함되어 있어야 합니다.

## 기타 문제

### 포트가 이미 사용 중인 경우

```bash
# 다른 포트로 실행
streamlit run app/main.py --server.port 8502
```

### 가상환경 문제

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 데이터베이스 파일 권한 문제

```bash
# 데이터베이스 파일 권한 확인
ls -la data/processed/roasting_profiles.db

# 필요시 권한 변경
chmod 644 data/processed/roasting_profiles.db
```

### 모델 파일을 찾을 수 없는 경우

머신러닝 모델을 사용하려면 먼저 모델을 학습시켜야 합니다:

```bash
# 센서 데이터 모델 학습
python scripts/train_sensor_model.py

# 이미지 모델 학습
python scripts/train_image_model.py
```

## 디버깅 팁

1. **경로 확인**: `app/main.py`에서 `project_root` 변수 출력하여 확인
2. **import 테스트**: Python 인터프리터에서 직접 import 테스트
3. **로그 확인**: Streamlit 실행 시 출력되는 오류 메시지 확인

