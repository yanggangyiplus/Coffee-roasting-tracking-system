"""
샘플 로스팅 데이터 생성 스크립트
실제 로스팅 프로파일을 시뮬레이션한 샘플 데이터를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.processor import SensorDataProcessor
from src.data.profile_manager import ProfileManager
from src.utils.constants import RoastLevel


def generate_roasting_profile(
    bean_type: str,
    target_level: RoastLevel,
    duration_minutes: float = 12.0,
    ambient_temp: float = 25.0,
    ambient_humidity: float = 50.0
) -> pd.DataFrame:
    """
    로스팅 프로파일 생성 (시뮬레이션)
    
    Args:
        bean_type: 원두 종류
        target_level: 목표 배전도
        duration_minutes: 총 로스팅 시간 (분)
        ambient_temp: 주변 온도
        ambient_humidity: 주변 습도
        
    Returns:
        센서 데이터 DataFrame
    """
    # 시간 간격 설정 (1초마다)
    total_seconds = int(duration_minutes * 60)
    time_points = np.arange(0, total_seconds, 1)
    
    # 목표 온도 설정
    target_temps = {
        RoastLevel.LIGHT: 200,
        RoastLevel.MEDIUM: 210,
        RoastLevel.MEDIUM_DARK: 220,
        RoastLevel.DARK: 235,
    }
    target_temp = target_temps[target_level]
    
    # 온도 곡선 생성 (S자 곡선)
    bean_temps = []
    drum_temps = []
    humidities = []
    heating_powers = []
    
    for t in time_points:
        t_minutes = t / 60.0
        
        # 원두 온도 곡선 (S자 곡선)
        # 초반: 급격한 상승, 중반: 완만한 상승, 후반: 다시 급격한 상승 후 완만해짐
        progress = t_minutes / duration_minutes
        
        if progress < 0.3:
            # 건조 단계: 급격한 상승
            temp = ambient_temp + (150 - ambient_temp) * (progress / 0.3) ** 1.5
        elif progress < 0.6:
            # 갈변 단계: 완만한 상승
            temp = 150 + (190 - 150) * ((progress - 0.3) / 0.3) ** 0.8
        elif progress < 0.75:
            # 1차 크랙 구간: 급격한 상승
            temp = 190 + (205 - 190) * ((progress - 0.6) / 0.15) ** 1.2
        elif progress < 0.9:
            # 발열 구간: 완만한 상승
            temp = 205 + (target_temp - 205) * ((progress - 0.75) / 0.15) ** 0.7
        else:
            # 최종 구간: 매우 완만한 상승
            temp = target_temp + (target_temp * 1.05 - target_temp) * ((progress - 0.9) / 0.1) ** 0.5
        
        # 노이즈 추가 (현실적인 변동)
        temp += np.random.normal(0, 0.5)
        bean_temps.append(max(ambient_temp, temp))
        
        # 드럼 온도 (원두 온도보다 약간 높음)
        drum_temp = temp + np.random.uniform(5, 15) + np.random.normal(0, 1)
        drum_temps.append(max(ambient_temp + 5, drum_temp))
        
        # 습도 (시간이 지날수록 감소)
        humidity = ambient_humidity * (1 - progress * 0.6) + np.random.normal(0, 2)
        humidities.append(max(10, min(100, humidity)))
        
        # 가열량 (초반 높음, 후반 낮아짐)
        if progress < 0.2:
            power = 90 - progress * 20
        elif progress < 0.7:
            power = 70 - (progress - 0.2) * 30
        else:
            power = 40 - (progress - 0.7) * 20
        
        power += np.random.normal(0, 3)
        heating_powers.append(max(20, min(100, power)))
    
    # DataFrame 생성
    df = pd.DataFrame({
        "timestamp": [datetime.now() - timedelta(seconds=total_seconds-t) for t in time_points],
        "bean_temp": bean_temps,
        "drum_temp": drum_temps,
        "humidity": humidities,
        "heating_power": heating_powers,
    })
    
    # RoR 계산
    processor = SensorDataProcessor()
    processor.load_from_dataframe(df)
    df = processor.get_dataframe()
    
    return df


def main():
    """메인 함수"""
    print("샘플 로스팅 데이터 생성 중...")
    
    # 출력 디렉토리 생성
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 여러 프로파일 생성
    profiles = [
        ("에티오피아 예가체프", RoastLevel.LIGHT, 11.0),
        ("콜롬비아 수프리모", RoastLevel.MEDIUM, 12.5),
        ("브라질 산토스", RoastLevel.MEDIUM_DARK, 13.0),
        ("인도네시아 만델링", RoastLevel.DARK, 14.0),
    ]
    
    profile_manager = ProfileManager()
    
    for bean_type, target_level, duration in profiles:
        print(f"생성 중: {bean_type} - {target_level.value} ({duration}분)")
        
        # 프로파일 생성
        df = generate_roasting_profile(
            bean_type=bean_type,
            target_level=target_level,
            duration_minutes=duration
        )
        
        # CSV 저장
        csv_path = output_dir / f"{bean_type}_{target_level.value}_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  → CSV 저장: {csv_path}")
        
        # 데이터베이스에 저장
        profile_id = profile_manager.save_profile(
            profile_name=f"{bean_type}_{target_level.value}",
            data_df=df,
            bean_type=bean_type,
            target_level=target_level,
            notes=f"샘플 데이터 - {duration}분 로스팅"
        )
        print(f"  → DB 저장: Profile ID {profile_id}")
    
    print("\n✅ 샘플 데이터 생성 완료!")


if __name__ == "__main__":
    main()

