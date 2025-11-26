"""
실시간 센서 스트림 인터페이스
실제 센서 하드웨어와 연동할 수 있는 인터페이스를 제공합니다.
"""

import time
import threading
from typing import Optional, Callable, Dict
from datetime import datetime
from abc import ABC, abstractmethod


class SensorStreamInterface(ABC):
    """센서 스트림 인터페이스 (추상 클래스)"""
    
    @abstractmethod
    def read_sensor_data(self) -> Dict:
        """
        센서 데이터 읽기
        
        Returns:
            센서 데이터 딕셔너리
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        센서 연결 상태 확인
        
        Returns:
            연결 여부
        """
        pass


class MockSensorStream(SensorStreamInterface):
    """모의 센서 스트림 (테스트용)"""
    
    def __init__(self, sample_rate: float = 1.0):
        """
        Args:
            sample_rate: 샘플링 주기 (초)
        """
        self.sample_rate = sample_rate
        self.connected = True
        self.start_time = datetime.now()
        self.counter = 0
    
    def read_sensor_data(self) -> Dict:
        """
        모의 센서 데이터 생성
        
        Returns:
            모의 센서 데이터
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        # 시뮬레이션: 시간에 따라 온도가 증가하는 로스팅 프로파일
        base_temp = 25.0
        target_temp = 200.0
        progress = min(1.0, elapsed / 720.0)  # 12분 기준
        
        bean_temp = base_temp + (target_temp - base_temp) * progress + self.counter * 0.1
        
        return {
            "bean_temp": round(bean_temp, 1),
            "drum_temp": round(bean_temp + 10 + self.counter * 0.05, 1),
            "humidity": max(20, 80 - progress * 50 + self.counter * 0.02),
            "heating_power": max(30, 90 - progress * 40),
            "ambient_temp": 25.0 + self.counter * 0.01,
            "ambient_humidity": 50.0 + self.counter * 0.02,
            "bean_color": self._get_bean_color(bean_temp),
            "timestamp": datetime.now(),
        }
    
    def _get_bean_color(self, temp: float) -> str:
        """온도에 따른 원두 색상 결정"""
        if temp < 50:
            return "Green"
        elif temp < 100:
            return "Yellow"
        elif temp < 150:
            return "Light Brown"
        elif temp < 190:
            return "Brown"
        elif temp < 220:
            return "Dark Brown"
        else:
            return "Very Dark"
    
    def is_connected(self) -> bool:
        return self.connected
    
    def disconnect(self):
        """연결 해제"""
        self.connected = False


class RealSensorStream(SensorStreamInterface):
    """
    실제 센서 스트림 클래스
    실제 하드웨어 센서와 연동하려면 이 클래스를 상속받아 구현하세요.
    
    예시:
        class ArduinoSensorStream(RealSensorStream):
            def __init__(self, port="/dev/ttyUSB0"):
                super().__init__()
                self.serial = serial.Serial(port, 9600)
            
            def read_sensor_data(self):
                line = self.serial.readline().decode()
                # 파싱 로직
                return parsed_data
    """
    
    def __init__(self):
        """초기화"""
        self.connected = False
    
    def is_connected(self) -> bool:
        return self.connected
    
    def connect(self):
        """센서 연결"""
        self.connected = True
    
    def disconnect(self):
        """센서 연결 해제"""
        self.connected = False


class SensorStreamReader:
    """센서 스트림 리더 (실시간 데이터 수집)"""
    
    def __init__(self, sensor_stream: SensorStreamInterface, callback: Optional[Callable] = None):
        """
        Args:
            sensor_stream: 센서 스트림 인터페이스
            callback: 데이터 수집 시 호출할 콜백 함수
        """
        self.sensor_stream = sensor_stream
        self.callback = callback
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self, sample_rate: float = 1.0):
        """
        스트림 읽기 시작
        
        Args:
            sample_rate: 샘플링 주기 (초)
        """
        if self.is_running:
            return
        
        self.is_running = True
        self.sample_rate = sample_rate
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """스트림 읽기 중지"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _read_loop(self):
        """읽기 루프 (백그라운드 스레드)"""
        while self.is_running:
            if not self.sensor_stream.is_connected():
                time.sleep(self.sample_rate)
                continue
            
            try:
                data = self.sensor_stream.read_sensor_data()
                if self.callback:
                    self.callback(data)
            except Exception as e:
                print(f"센서 데이터 읽기 오류: {e}")
            
            time.sleep(self.sample_rate)

