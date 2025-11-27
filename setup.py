"""
프로젝트 설정 파일
"""

from setuptools import setup, find_packages

setup(
    name="coffee-roasting-tracking-system",
    version="1.0.0",
    description="커피 로스팅 단계 추적 및 예측 시스템",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "streamlit>=1.28.0",
        "sqlalchemy>=2.0.0",
        "openpyxl>=3.1.0",
        "python-dateutil>=2.8.0",
        "pydantic>=2.0.0",
        "albumentations>=1.3.0",
        "tsaug>=0.2.1",
        "scipy>=1.10.0",
    ],
)

