#!/bin/bash

# Streamlit 대시보드 실행 스크립트

echo "커피 로스팅 추적 시스템 대시보드 시작 중..."

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Streamlit 실행
streamlit run app/main.py --server.port 8501 --server.address localhost

