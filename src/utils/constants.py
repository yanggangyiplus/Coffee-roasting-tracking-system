"""
로스팅 단계 및 상수 정의
"""

from enum import Enum


class RoastingStage(Enum):
    """로스팅 단계 열거형"""
    DRYING = "건조 단계"
    MAILLARD = "갈변 단계"
    FIRST_CRACK = "1차 크랙"
    DEVELOPMENT = "발열/발화 구간"
    SECOND_CRACK = "2차 크랙"
    COMPLETED = "완료"


class RoastLevel(Enum):
    """배전도 레벨"""
    GREEN = "생원두"  # 로스팅 전
    LIGHT = "약배전"
    MEDIUM = "중배전"
    MEDIUM_DARK = "중강배전"
    DARK = "강배전"


class BeanColor(Enum):
    """원두 색상 상태"""
    GREEN = "생원두 (Green)"
    YELLOW = "노란색 (Yellow)"
    LIGHT_BROWN = "밝은 갈색 (Light Brown)"
    BROWN = "갈색 (Brown)"
    DARK_BROWN = "진한 갈색 (Dark Brown)"
    VERY_DARK = "매우 진한 갈색 (Very Dark)"


# 온도 임계값 (섭씨)
TEMPERATURE_THRESHOLDS = {
    "drying_start": 50,      # 건조 단계 시작 온도
    "maillard_start": 150,   # 갈변 단계 시작 온도
    "first_crack_min": 190,  # 1차 크랙 최소 온도
    "first_crack_max": 205,  # 1차 크랙 최대 온도
    "second_crack_min": 220, # 2차 크랙 최소 온도
    "second_crack_max": 240, # 2차 크랙 최대 온도
}

# 목표 배전도별 온도 범위
ROAST_LEVEL_TEMPERATURES = {
    RoastLevel.LIGHT: (195, 205),      # 약배전: 1차 크랙 직후
    RoastLevel.MEDIUM: (205, 215),     # 중배전
    RoastLevel.MEDIUM_DARK: (215, 225), # 중강배전
    RoastLevel.DARK: (225, 240),       # 강배전: 2차 크랙 이후
}

# RoR 변화 임계값 (도/분)
ROR_THRESHOLDS = {
    "first_crack_drop": 5,   # 1차 크랙 감지를 위한 RoR 급감 임계값
    "second_crack_drop": 3,  # 2차 크랙 감지를 위한 RoR 급감 임계값
}

# 습도 임계값 (%)
HUMIDITY_THRESHOLDS = {
    "drying_high": 80,       # 건조 단계 고습도
    "drying_low": 30,        # 건조 단계 저습도
}

# 생원두 감지 임계값
GREEN_BEAN_THRESHOLDS = {
    "max_temp": 50,          # 생원두 최대 온도 (50°C 이하)
    "max_humidity": 90,      # 생원두 최대 습도
}

# 원두 색상별 온도 범위 (대략적)
BEAN_COLOR_TEMPERATURES = {
    BeanColor.GREEN: (0, 50),
    BeanColor.YELLOW: (50, 100),
    BeanColor.LIGHT_BROWN: (100, 150),
    BeanColor.BROWN: (150, 190),
    BeanColor.DARK_BROWN: (190, 220),
    BeanColor.VERY_DARK: (220, 300),
}

