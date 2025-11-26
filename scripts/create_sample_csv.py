"""
샘플 로스팅 데이터 CSV 파일 생성 스크립트
테스트용으로 사용할 수 있는 실제 로스팅 프로파일 데이터를 생성합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def create_sample_roasting_csv(
    output_path: str = "data/raw/sample_roasting_data.csv",
    duration_minutes: float = 12.0,
    bean_type: str = "에티오피아 예가체프",
    target_level: str = "약배전"
):
    """
    샘플 로스팅 데이터 CSV 파일 생성
    
    Args:
        output_path: 출력 파일 경로
        duration_minutes: 로스팅 시간 (분)
        bean_type: 원두 종류
        target_level: 목표 배전도
    """
    # 시간 간격 설정 (1초마다)
    total_seconds = int(duration_minutes * 60)
    time_points = np.arange(0, total_seconds, 1)
    
    # 목표 온도 설정
    target_temps = {
        "약배전": 200,
        "중배전": 210,
        "중강배전": 220,
        "강배전": 235,
    }
    target_temp = target_temps.get(target_level, 200)
    
    # 데이터 생성
    data = []
    start_time = datetime.now() - timedelta(seconds=total_seconds)
    
    for i, t in enumerate(time_points):
        t_minutes = t / 60.0
        progress = t_minutes / duration_minutes
        
        # 원두 온도 곡선 (S자 곡선)
        if progress < 0.3:
            bean_temp = 25 + (150 - 25) * (progress / 0.3) ** 1.5
        elif progress < 0.6:
            bean_temp = 150 + (190 - 150) * ((progress - 0.3) / 0.3) ** 0.8
        elif progress < 0.75:
            bean_temp = 190 + (205 - 190) * ((progress - 0.6) / 0.15) ** 1.2
        elif progress < 0.9:
            bean_temp = 205 + (target_temp - 205) * ((progress - 0.75) / 0.15) ** 0.7
        else:
            bean_temp = target_temp + (target_temp * 1.05 - target_temp) * ((progress - 0.9) / 0.1) ** 0.5
        
        # 노이즈 추가
        bean_temp += np.random.normal(0, 0.5)
        bean_temp = max(25, bean_temp)
        
        # 드럼 온도
        drum_temp = bean_temp + np.random.uniform(5, 15) + np.random.normal(0, 1)
        drum_temp = max(bean_temp + 5, drum_temp)
        
        # 습도 (시간이 지날수록 감소)
        humidity = 80 * (1 - progress * 0.6) + np.random.normal(0, 2)
        humidity = max(10, min(100, humidity))
        
        # 가열량
        if progress < 0.2:
            heating_power = 90 - progress * 20
        elif progress < 0.7:
            heating_power = 70 - (progress - 0.2) * 30
        else:
            heating_power = 40 - (progress - 0.7) * 20
        heating_power += np.random.normal(0, 3)
        heating_power = max(20, min(100, heating_power))
        
        # 주변 온도 (약간의 변동)
        ambient_temp = 25 + np.random.normal(0, 1)
        ambient_temp = max(20, min(30, ambient_temp))
        
        # 주변 습도
        ambient_humidity = 50 + np.random.normal(0, 5)
        ambient_humidity = max(30, min(70, ambient_humidity))
        
        # 원두 색상 (온도 기반)
        if bean_temp < 50:
            bean_color = "Green"
        elif bean_temp < 100:
            bean_color = "Yellow"
        elif bean_temp < 150:
            bean_color = "Light Brown"
        elif bean_temp < 190:
            bean_color = "Brown"
        elif bean_temp < 220:
            bean_color = "Dark Brown"
        else:
            bean_color = "Very Dark"
        
        # 타임스탬프
        timestamp = start_time + timedelta(seconds=t)
        
        data.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "time": t,  # 경과 시간 (초)
            "bean_temp": round(bean_temp, 2),
            "drum_temp": round(drum_temp, 2),
            "humidity": round(humidity, 2),
            "heating_power": round(heating_power, 2),
            "ambient_temp": round(ambient_temp, 2),
            "ambient_humidity": round(ambient_humidity, 2),
            "bean_color": bean_color,
            "bean_type": bean_type,
            "target_level": target_level,
        })
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    
    # 출력 디렉토리 생성
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV 저장
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 샘플 CSV 파일 생성 완료: {output_path}")
    print(f"   - 데이터 포인트 수: {len(df)}")
    print(f"   - 총 시간: {duration_minutes}분")
    print(f"   - 원두 종류: {bean_type}")
    print(f"   - 목표 배전도: {target_level}")
    
    return output_path


def main():
    """메인 함수"""
    print("샘플 로스팅 데이터 CSV 파일 생성 중...\n")
    
    # 여러 샘플 파일 생성
    samples = [
        ("에티오피아 예가체프", "약배전", 11.0),
        ("콜롬비아 수프리모", "중배전", 12.5),
        ("브라질 산토스", "중강배전", 13.0),
        ("인도네시아 만델링", "강배전", 14.0),
    ]
    
    for bean_type, target_level, duration in samples:
        filename = f"sample_{bean_type.replace(' ', '_')}_{target_level}.csv"
        output_path = f"data/raw/{filename}"
        create_sample_roasting_csv(
            output_path=output_path,
            duration_minutes=duration,
            bean_type=bean_type,
            target_level=target_level
        )
        print()
    
    print("✅ 모든 샘플 파일 생성 완료!")


if __name__ == "__main__":
    main()

