#!/bin/bash

# Streamlit 대시보드 실행 스크립트

echo "커피 로스팅 추적 시스템 대시보드 시작 중..."

# 스크립트가 있는 디렉토리로 이동 (프로젝트 루트)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "프로젝트 루트: $PROJECT_ROOT"

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "가상환경 활성화됨"
fi

# PYTHONPATH 설정
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Streamlit 실행
streamlit run app/main.py --server.port 8501 --server.address localhost

