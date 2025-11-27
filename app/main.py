"""
Streamlit 기반 로스팅 추적 대시보드
파일 업로드 및 실시간 센서 스트림 지원
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import io
from pathlib import Path

# 프로젝트 루트 경로 추가
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
project_root_str = str(project_root)

# 경로를 sys.path에 추가 (중복 방지)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# 작업 디렉토리 변경
try:
    os.chdir(project_root_str)
except:
    pass

# 모듈 import
try:
    from src.data.processor import SensorDataProcessor
    from src.data.file_loader import FileLoader
    from src.data.sensor_stream import MockSensorStream, SensorStreamReader, RealSensorStream
    from src.algorithms.stage_detector import RoastingStageDetector
    from src.prediction.roast_predictor import RoastLevelPredictor
    from src.data.profile_manager import ProfileManager
    from src.utils.constants import RoastingStage, RoastLevel, BeanColor
except ImportError as e:
    # Streamlit이 실행되기 전이므로 print 사용
    print(f"경로 설정: {project_root_str}")
    print(f"sys.path: {sys.path[:3]}")
    print(f"Import 오류: {e}")
    print("\n해결 방법:")
    print("1. 프로젝트 루트에서 실행: cd /path/to/Coffee-roasting-tracking-system && streamlit run app/main.py")
    print("2. 또는 실행 스크립트 사용: bash scripts/run_dashboard.sh")
    print("3. 또는 패키지 설치: pip install -e .")
    raise

# 머신러닝 모델 (선택적)
try:
    from src.models.image_classifier import ImageClassifierPredictor
    from src.models.sensor_classifier import SensorDataClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    ImageClassifierPredictor = None
    SensorDataClassifier = None


# 페이지 설정
st.set_page_config(
    page_title="커피 로스팅 추적 시스템",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "processor" not in st.session_state:
    st.session_state.processor = SensorDataProcessor()
if "stage_detector" not in st.session_state:
    st.session_state.stage_detector = RoastingStageDetector()
if "predictor" not in st.session_state:
    st.session_state.predictor = RoastLevelPredictor()
if "is_roasting" not in st.session_state:
    st.session_state.is_roasting = False
if "target_level" not in st.session_state:
    st.session_state.target_level = None
if "profile_name" not in st.session_state:
    st.session_state.profile_name = ""
if "bean_type" not in st.session_state:
    st.session_state.bean_type = ""
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "manual"  # manual, file, realtime
if "sensor_stream" not in st.session_state:
    st.session_state.sensor_stream = None
if "stream_reader" not in st.session_state:
    st.session_state.stream_reader = None
if "use_ml_model" not in st.session_state:
    st.session_state.use_ml_model = False
if "sensor_classifier" not in st.session_state:
    st.session_state.sensor_classifier = None
if "image_classifier" not in st.session_state:
    st.session_state.image_classifier = None


def main():
    """메인 함수"""
    st.title("☕ 커피 로스팅 추적 시스템")
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 데이터 모드 선택
        data_mode = st.radio(
            "데이터 입력 모드",
            options=["수동 입력", "파일 업로드", "실시간 센서"],
            index=0 if st.session_state.data_mode == "manual" else (1 if st.session_state.data_mode == "file" else 2)
        )
        
        if data_mode == "수동 입력":
            st.session_state.data_mode = "manual"
        elif data_mode == "파일 업로드":
            st.session_state.data_mode = "file"
        else:
            st.session_state.data_mode = "realtime"
        
        st.markdown("---")
        
        # 파일 업로드 모드
        if st.session_state.data_mode == "file":
            uploaded_file = st.file_uploader(
                "CSV 또는 엑셀 파일 업로드",
                type=["csv", "xlsx", "xls"],
                help="로스팅 센서 데이터가 포함된 파일을 업로드하세요"
            )
            
            if uploaded_file is not None:
                try:
                    file_loader = FileLoader()
                    
                    # 파일 확장자에 따라 로드
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # 컬럼 이름 표준화 및 검증
                    df = file_loader.normalize_column_names(df)
                    df = file_loader.validate_and_clean(df)
                    
                    # 프로세서에 로드
                    st.session_state.processor.load_from_dataframe(df)
                    st.success(f"파일 로드 완료: {len(df)}개 데이터 포인트")
                    
                    # 자동으로 로스팅 시작
                    if not st.session_state.is_roasting:
                        st.session_state.is_roasting = True
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"파일 로드 오류: {str(e)}")
        
        # 실시간 센서 모드
        elif st.session_state.data_mode == "realtime":
            sensor_type = st.selectbox(
                "센서 타입",
                options=["모의 센서 (테스트)", "실제 센서"],
                help="실제 센서를 사용하려면 RealSensorStream을 구현해야 합니다"
            )
            
            if sensor_type == "모의 센서 (테스트)":
                if st.button("센서 연결", type="primary"):
                    st.session_state.sensor_stream = MockSensorStream(sample_rate=1.0)
                    st.session_state.stream_reader = SensorStreamReader(
                        st.session_state.sensor_stream,
                        callback=lambda data: st.session_state.processor.add_data_point(**data)
                    )
                    st.session_state.stream_reader.start(sample_rate=1.0)
                    st.session_state.is_roasting = True
                    st.success("모의 센서 연결됨")
                    st.rerun()
            
            if st.session_state.sensor_stream and st.session_state.sensor_stream.is_connected():
                st.success("✅ 센서 연결됨")
                if st.button("센서 연결 해제"):
                    if st.session_state.stream_reader:
                        st.session_state.stream_reader.stop()
                    st.session_state.is_roasting = False
                    st.rerun()
        
        st.markdown("---")
        
        # 로스팅 시작/중지 (수동 모드)
        if st.session_state.data_mode == "manual":
            if not st.session_state.is_roasting:
                st.session_state.profile_name = st.text_input(
                    "프로파일 이름",
                    value=f"로스팅_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                st.session_state.bean_type = st.text_input("원두 종류", value="")
                st.session_state.target_level = st.selectbox(
                    "목표 배전도",
                    options=[r for r in RoastLevel if r != RoastLevel.GREEN],
                    format_func=lambda x: x.value
                )
                
                if st.button("🟢 로스팅 시작", type="primary", use_container_width=True):
                    st.session_state.is_roasting = True
                    st.session_state.processor.reset()
                    st.session_state.stage_detector.reset()
                    st.session_state.predictor.reset()
                    st.rerun()
            else:
                st.warning("로스팅 진행 중...")
                if st.button("🔴 로스팅 중지", type="secondary", use_container_width=True):
                    st.session_state.is_roasting = False
                    st.rerun()
        
        st.markdown("---")
        
        # 머신러닝 모델 설정
        st.header("🤖 머신러닝 모델")
        if ML_AVAILABLE:
            use_ml = st.checkbox(
                "머신러닝 모델 사용",
                value=st.session_state.use_ml_model,
                help="학습된 모델을 사용하여 더 정확한 배전도 예측"
            )
            
            if use_ml != st.session_state.use_ml_model:
                st.session_state.use_ml_model = use_ml
                
                # 센서 데이터 분류 모델 로드
                if use_ml and st.session_state.sensor_classifier is None:
                    sensor_model_path = "models/sensor_classifier/model.pkl"
                    if Path(sensor_model_path).exists():
                        try:
                            st.session_state.sensor_classifier = SensorDataClassifier()
                            st.session_state.sensor_classifier.load_model(sensor_model_path)
                            st.session_state.stage_detector = RoastingStageDetector(
                                use_ml_model=True,
                                sensor_model_path=sensor_model_path
                            )
                            st.success("머신러닝 모델 로드 완료!")
                        except Exception as e:
                            st.error(f"모델 로드 실패: {e}")
                            st.session_state.use_ml_model = False
                    else:
                        st.warning(f"모델 파일을 찾을 수 없습니다: {sensor_model_path}")
                        st.info("먼저 모델을 학습시켜야 합니다: python scripts/train_sensor_model.py")
                        st.session_state.use_ml_model = False
                elif not use_ml:
                    st.session_state.stage_detector = RoastingStageDetector(use_ml_model=False)
        else:
            st.info("머신러닝 라이브러리가 설치되지 않았습니다. pip install torch torchvision")
        
        st.markdown("---")
        
        # 프로파일 관리
        st.header("📁 프로파일 관리")
        profile_manager = ProfileManager()
        
        if st.button("프로파일 목록 보기", use_container_width=True):
            st.session_state.show_profiles = True
        
        if st.button("새 프로파일로 시작", use_container_width=True):
            st.session_state.is_roasting = False
            st.session_state.processor.reset()
            st.session_state.stage_detector.reset()
            st.session_state.predictor.reset()
            if st.session_state.stream_reader:
                st.session_state.stream_reader.stop()
            st.rerun()
    
    # 메인 컨텐츠
    if st.session_state.is_roasting:
        show_roasting_dashboard()
    elif st.session_state.get("show_profiles", False):
        show_profile_management(profile_manager)
        st.session_state.show_profiles = False
    else:
        show_welcome_screen()


def show_welcome_screen():
    """환영 화면"""
    st.info("👈 사이드바에서 데이터 입력 모드를 선택하세요!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("로스팅 단계 추적", "실시간")
    with col2:
        st.metric("배전도 예측", "자동")
    with col3:
        st.metric("프로파일 저장", "지원")
    
    st.markdown("### 주요 기능")
    st.markdown("""
    - 🔥 **실시간 온도 추적**: 원두 온도와 드럼 온도를 실시간으로 모니터링
    - 📊 **RoR 계산**: Rate of Rise를 자동으로 계산하여 로스팅 진행 상황 파악
    - 🎯 **단계 감지**: 생원두, 건조, 갈변, 1차 크랙, 발열, 2차 크랙 단계 자동 감지 (규칙 기반 + ML)
    - 🤖 **머신러닝 모델**: RandomForest/GradientBoosting 센서 분류 + ResNet18 CNN 이미지 분류 (선택 가능)
    - 🌡️ **환경 데이터**: 날씨 온도/습도 추적
    - 🎨 **원두 색상 감지**: 온도 기반 원두 색상 자동 감지
    - ⏱️ **도달 시간 예측**: 목표 배전도 도달까지 예상 시간 예측
    - 💾 **프로파일 저장**: 로스팅 프로파일을 저장하고 비교 분석
    - 📁 **다양한 입력 모드**: 수동 입력, 파일 업로드, 실시간 센서 스트림 지원
    - 📥 **데이터 다운로드**: 실시간 데이터를 CSV로 다운로드
    """)


def show_roasting_dashboard():
    """로스팅 대시보드"""
    # 수동 입력 모드일 때만 센서 데이터 입력 폼 표시
    if st.session_state.data_mode == "manual":
        with st.expander("📝 센서 데이터 입력", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                bean_temp = st.number_input("원두 온도 (°C)", min_value=0.0, max_value=300.0, value=25.0, step=0.1)
            with col2:
                drum_temp = st.number_input("드럼 온도 (°C)", min_value=0.0, max_value=300.0, value=25.0, step=0.1)
            with col3:
                humidity = st.number_input("습도 (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            with col4:
                heating_power = st.number_input("가열량 (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            
            col5, col6, col7 = st.columns(3)
            with col5:
                ambient_temp = st.number_input("주변 온도 (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
            with col6:
                ambient_humidity = st.number_input("주변 습도 (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
            with col7:
                bean_color = st.selectbox(
                    "원두 색상",
                    options=["", "Green", "Yellow", "Light Brown", "Brown", "Dark Brown", "Very Dark"]
                )
            
            if st.button("데이터 추가", type="primary"):
                # 데이터 포인트 추가
                data_point = st.session_state.processor.add_data_point(
                    bean_temp=bean_temp,
                    drum_temp=drum_temp,
                    humidity=humidity,
                    heating_power=heating_power,
                    ambient_temp=ambient_temp if ambient_temp else None,
                    ambient_humidity=ambient_humidity if ambient_humidity else None,
                    bean_color=bean_color if bean_color else None
                )
                
                # 단계 감지
                current_stage = st.session_state.stage_detector.detect_stage(
                    bean_temp=data_point["bean_temp"],
                    drum_temp=data_point["drum_temp"],
                    humidity=data_point["humidity"],
                    ror=data_point["ror"],
                    elapsed_time=data_point["elapsed_time"],
                    heating_power=data_point["heating_power"]
                )
                
                st.rerun()
    
    # 실시간 센서 모드일 때 자동 업데이트
    elif st.session_state.data_mode == "realtime":
        if st.session_state.stream_reader and st.session_state.stream_reader.is_running:
            st.info("🔄 실시간 센서 데이터 수집 중...")
            # 자동 새로고침 (선택사항)
            if st.checkbox("자동 새로고침 활성화", value=True):
                time.sleep(1)  # 1초 대기
                st.rerun()
    
    # 현재 상태 표시
    df = st.session_state.processor.get_dataframe()
    
    if len(df) > 0:
        current_data = df.iloc[-1]
        current_stage = st.session_state.stage_detector.detect_stage(
            bean_temp=current_data["bean_temp"],
            drum_temp=current_data["drum_temp"],
            humidity=current_data["humidity"],
            ror=current_data["ror"],
            elapsed_time=current_data["elapsed_time"],
            heating_power=current_data["heating_power"]
        )
        
        # 배전도 레벨 감지 (머신러닝 모델 사용 여부에 따라)
        sensor_data_dict = {
            "bean_temp": current_data["bean_temp"],
            "drum_temp": current_data["drum_temp"],
            "humidity": current_data["humidity"],
            "heating_power": current_data["heating_power"],
            "ror": current_data["ror"],
            "elapsed_time": current_data["elapsed_time"],
        }
        
        roast_level, prediction_info = st.session_state.stage_detector.detect_roast_level(
            bean_temp=current_data["bean_temp"],
            bean_color=current_data.get("bean_color"),
            sensor_data=sensor_data_dict if st.session_state.use_ml_model else None
        )
        
        # 원두 색상 감지
        bean_color = st.session_state.stage_detector.detect_bean_color(current_data["bean_temp"])
        
        # 상태 메트릭
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("현재 단계", current_stage.value)
        with col2:
            # 배전도 표시 (머신러닝 모델 사용 시 신뢰도 표시)
            if prediction_info.get("method") == "ml_model":
                confidence = prediction_info.get("confidence", 0)
                st.metric(
                    "배전도 (ML)",
                    f"{roast_level.value}",
                    delta=f"{confidence*100:.1f}% 신뢰도"
                )
            else:
                st.metric("배전도", roast_level.value)
        with col3:
            st.metric("원두 온도", f"{current_data['bean_temp']:.1f}°C")
        with col4:
            st.metric("RoR", f"{current_data['ror']:.2f}°C/분")
        with col5:
            st.metric("경과 시간", f"{current_data['elapsed_time']/60:.1f}분")
        
        # 추가 정보
        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("원두 색상", bean_color.value)
        with col7:
            if "ambient_temp" in current_data:
                st.metric("주변 온도", f"{current_data['ambient_temp']:.1f}°C")
        with col8:
            if "ambient_humidity" in current_data:
                st.metric("주변 습도", f"{current_data['ambient_humidity']:.1f}%")
        
        # 목표 배전도 예측 (생원두가 아닌 경우만)
        if st.session_state.target_level and roast_level != RoastLevel.GREEN:
            prediction = st.session_state.predictor.predict_time_to_target(
                current_temp=current_data["bean_temp"],
                current_ror=current_data["ror"],
                target_level=st.session_state.target_level,
                elapsed_time=current_data["elapsed_time"]
            )
            
            st.markdown("### 🎯 목표 배전도 예측")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                if prediction["target_reached"]:
                    st.success(f"✅ 목표 도달: {st.session_state.target_level.value}")
                else:
                    st.info(f"목표: {st.session_state.target_level.value}")
            
            with pred_col2:
                if not prediction["target_reached"]:
                    st.metric(
                        "예상 시간",
                        f"{prediction['estimated_time_minutes']:.1f}분"
                    )
            
            with pred_col3:
                st.progress(prediction["progress_percent"] / 100)
                st.caption(f"진행률: {prediction['progress_percent']:.1f}%")
        
        # 그래프 시각화
        st.markdown("### 📊 실시간 그래프")
        
        # 온도 그래프
        fig_temp = make_subplots(
            rows=3, cols=1,
            subplot_titles=("온도 변화", "RoR 변화", "습도 변화"),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 원두 온도
        fig_temp.add_trace(
            go.Scatter(
                x=df["elapsed_time"] / 60,
                y=df["bean_temp"],
                name="원두 온도",
                line=dict(color="red", width=2)
            ),
            row=1, col=1
        )
        
        # 드럼 온도
        fig_temp.add_trace(
            go.Scatter(
                x=df["elapsed_time"] / 60,
                y=df["drum_temp"],
                name="드럼 온도",
                line=dict(color="orange", width=2)
            ),
            row=1, col=1
        )
        
        # 주변 온도 (있는 경우)
        if "ambient_temp" in df.columns:
            fig_temp.add_trace(
                go.Scatter(
                    x=df["elapsed_time"] / 60,
                    y=df["ambient_temp"],
                    name="주변 온도",
                    line=dict(color="blue", width=1, dash="dash")
                ),
                row=1, col=1
            )
        
        # RoR
        fig_temp.add_trace(
            go.Scatter(
                x=df["elapsed_time"] / 60,
                y=df["ror"],
                name="RoR",
                line=dict(color="green", width=2),
                fill="tozeroy"
            ),
            row=2, col=1
        )
        
        # 습도
        fig_temp.add_trace(
            go.Scatter(
                x=df["elapsed_time"] / 60,
                y=df["humidity"],
                name="로스팅기 습도",
                line=dict(color="purple", width=2)
            ),
            row=3, col=1
        )
        
        # 주변 습도 (있는 경우)
        if "ambient_humidity" in df.columns:
            fig_temp.add_trace(
                go.Scatter(
                    x=df["elapsed_time"] / 60,
                    y=df["ambient_humidity"],
                    name="주변 습도",
                    line=dict(color="cyan", width=1, dash="dash")
                ),
                row=3, col=1
            )
        
        fig_temp.update_xaxes(title_text="시간 (분)", row=3, col=1)
        fig_temp.update_yaxes(title_text="온도 (°C)", row=1, col=1)
        fig_temp.update_yaxes(title_text="RoR (°C/분)", row=2, col=1)
        fig_temp.update_yaxes(title_text="습도 (%)", row=3, col=1)
        fig_temp.update_layout(height=800, showlegend=True)
        
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # 가열량 그래프
        col1, col2 = st.columns(2)
        
        with col1:
            fig_power = go.Figure()
            fig_power.add_trace(
                go.Scatter(
                    x=df["elapsed_time"] / 60,
                    y=df["heating_power"],
                    name="가열량",
                    line=dict(color="purple", width=2),
                    fill="tozeroy"
                )
            )
            fig_power.update_layout(
                title="가열량 변화",
                xaxis_title="시간 (분)",
                yaxis_title="가열량 (%)",
                height=300
            )
            st.plotly_chart(fig_power, use_container_width=True)
        
        with col2:
            # 원두 색상 정보 (있는 경우)
            if "bean_color" in df.columns:
                color_counts = df["bean_color"].value_counts()
                fig_color = go.Figure(data=[
                    go.Bar(x=color_counts.index, y=color_counts.values)
                ])
                fig_color.update_layout(
                    title="원두 색상 분포",
                    xaxis_title="색상",
                    yaxis_title="데이터 포인트 수",
                    height=300
                )
                st.plotly_chart(fig_color, use_container_width=True)
        
        # 데이터 다운로드 버튼
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # CSV 다운로드
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 데이터 다운로드 (CSV)",
                data=csv,
                file_name=f"roasting_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("💾 프로파일 저장", type="primary", use_container_width=True):
                profile_manager = ProfileManager()
                profile_id = profile_manager.save_profile(
                    profile_name=st.session_state.profile_name,
                    data_df=df,
                    bean_type=st.session_state.bean_type,
                    target_level=st.session_state.target_level,
                    notes=""
                )
                st.success(f"프로파일이 저장되었습니다! (ID: {profile_id})")
    else:
        st.info("센서 데이터를 입력하거나 파일을 업로드하세요.")


def show_profile_management(profile_manager: ProfileManager):
    """프로파일 관리 화면"""
    st.header("📁 프로파일 관리")
    
    # 필터
    col1, col2 = st.columns(2)
    
    with col1:
        bean_filter = st.text_input("원두 종류 필터", value="")
    with col2:
        level_filter = st.selectbox(
            "배전도 필터",
            options=[None] + [r for r in RoastLevel if r != RoastLevel.GREEN],
            format_func=lambda x: "전체" if x is None else x.value
        )
    
    # 프로파일 목록
    profiles_df = profile_manager.list_profiles(
        bean_type=bean_filter if bean_filter else None,
        target_level=level_filter
    )
    
    if len(profiles_df) > 0:
        # 탭으로 구분: 목록/상세보기, 비교 분석
        tab1, tab2 = st.tabs(["📋 프로파일 목록", "📊 프로파일 비교"])
        
        with tab1:
            st.dataframe(profiles_df, use_container_width=True)
            
            # 프로파일 상세 보기
            selected_id = st.selectbox(
                "프로파일 선택",
                options=profiles_df["id"].tolist(),
                format_func=lambda x: f"ID {x}: {profiles_df[profiles_df['id']==x]['profile_name'].iloc[0]}"
            )
            
            if selected_id:
                show_profile_detail(profile_manager, selected_id, profiles_df)
        
        with tab2:
            show_profile_comparison(profile_manager, profiles_df)
        
    else:
        st.info("저장된 프로파일이 없습니다.")


def show_profile_detail(profile_manager: ProfileManager, selected_id: int, profiles_df: pd.DataFrame):
    """프로파일 상세 보기"""
    profile = profile_manager.load_profile(selected_id)
    
    if profile:
        st.markdown("### 프로파일 상세")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 시간", f"{profile['metadata']['total_time_seconds']/60:.1f}분")
        with col2:
            st.metric("최종 온도", f"{profile['metadata']['final_temp']:.1f}°C")
        with col3:
            st.metric("목표 배전도", profile['metadata']['target_level'] or "N/A")
        
        # 통계 정보
        stats = profile_manager.calculate_statistics(profile)
        if stats:
            st.markdown("#### 통계 정보")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("평균 온도", f"{stats['avg_temp']:.1f}°C")
            with stat_col2:
                st.metric("평균 RoR", f"{stats['avg_ror']:.2f}°C/분")
            with stat_col3:
                st.metric("최대 RoR", f"{stats['max_ror']:.2f}°C/분")
            with stat_col4:
                st.metric("온도 상승률", f"{stats['temp_rise_rate']:.2f}°C/분")
        
        # 그래프 표시
        data_df = profile["data"]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("온도 곡선", "RoR 곡선"),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_df["elapsed_time"] / 60,
                y=data_df["bean_temp"],
                name="원두 온도",
                line=dict(color="red")
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data_df["elapsed_time"] / 60,
                y=data_df["ror"],
                name="RoR",
                line=dict(color="blue")
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="시간 (분)", row=2, col=1)
        fig.update_yaxes(title_text="온도 (°C)", row=1, col=1)
        fig.update_yaxes(title_text="RoR (°C/분)", row=2, col=1)
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 데이터 다운로드
        csv = data_df.to_csv(index=False)
        st.download_button(
            label="📥 프로파일 데이터 다운로드 (CSV)",
            data=csv,
            file_name=f"profile_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # 삭제 버튼
        if st.button("🗑️ 프로파일 삭제", type="secondary"):
            if profile_manager.delete_profile(selected_id):
                st.success("프로파일이 삭제되었습니다.")
                st.rerun()


def show_profile_comparison(profile_manager: ProfileManager, profiles_df: pd.DataFrame):
    """다중 프로파일 비교 분석 화면"""
    st.markdown("### 프로파일 비교 분석")
    
    # 비교할 프로파일 선택 (다중 선택)
    available_profiles = [
        (row["id"], f"ID {row['id']}: {row['profile_name']} ({row.get('bean_type', 'N/A')})")
        for _, row in profiles_df.iterrows()
    ]
    
    selected_profile_ids = st.multiselect(
        "비교할 프로파일 선택 (2개 이상)",
        options=[pid for pid, _ in available_profiles],
        format_func=lambda x: next(label for pid, label in available_profiles if pid == x),
        help="최소 2개 이상의 프로파일을 선택하세요"
    )
    
    if len(selected_profile_ids) >= 2:
        # 프로파일 비교 실행
        comparison = profile_manager.compare_profiles(selected_profile_ids)
        
        if "error" in comparison:
            st.error(comparison["error"])
        else:
            # 통계 비교 테이블
            st.markdown("#### 통계 비교")
            stats_data = []
            for i, (profile_info, stats) in enumerate(zip(comparison["profiles"], comparison["statistics"])):
                stats_data.append({
                    "프로파일": profile_info["name"],
                    "원두 종류": profile_info["bean_type"] or "N/A",
                    "목표 배전도": profile_info["target_level"] or "N/A",
                    "총 시간 (분)": f"{profile_info['total_time']/60:.1f}",
                    "최종 온도 (°C)": f"{profile_info['final_temp']:.1f}",
                    "평균 온도 (°C)": f"{stats['avg_temp']:.1f}",
                    "평균 RoR (°C/분)": f"{stats['avg_ror']:.2f}",
                    "최대 RoR (°C/분)": f"{stats['max_ror']:.2f}",
                    "온도 상승률 (°C/분)": f"{stats['temp_rise_rate']:.2f}",
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # 유사도 행렬
            st.markdown("#### 프로파일 유사도 행렬")
            similarity_df = pd.DataFrame(
                comparison["similarity_matrix"],
                index=[p["name"] for p in comparison["profiles"]],
                columns=[p["name"] for p in comparison["profiles"]]
            )
            st.dataframe(similarity_df.style.format("{:.2%}"), use_container_width=True)
            
            # 온도 곡선 비교 그래프
            st.markdown("#### 온도 곡선 비교")
            fig_temp = go.Figure()
            
            colors = ["red", "blue", "green", "orange", "purple", "brown"]
            for i, curve in enumerate(comparison["temperature_curves"]):
                time_minutes = [t / 60.0 for t in curve["time"]]
                fig_temp.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=curve["temp"],
                        name=curve["name"],
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
            
            fig_temp.update_layout(
                title="온도 곡선 비교",
                xaxis_title="시간 (분)",
                yaxis_title="온도 (°C)",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # RoR 곡선 비교 그래프
            st.markdown("#### RoR 곡선 비교")
            fig_ror = go.Figure()
            
            for i, curve in enumerate(comparison["ror_curves"]):
                time_minutes = [t / 60.0 for t in curve["time"]]
                fig_ror.add_trace(
                    go.Scatter(
                        x=time_minutes,
                        y=curve["ror"],
                        name=curve["name"],
                        line=dict(color=colors[i % len(colors)], width=2)
                    )
                )
            
            fig_ror.update_layout(
                title="RoR 곡선 비교",
                xaxis_title="시간 (분)",
                yaxis_title="RoR (°C/분)",
                height=500,
                hovermode="x unified"
            )
            st.plotly_chart(fig_ror, use_container_width=True)
            
            # 유사도 히트맵
            st.markdown("#### 유사도 히트맵")
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=comparison["similarity_matrix"],
                x=[p["name"] for p in comparison["profiles"]],
                y=[p["name"] for p in comparison["profiles"]],
                colorscale="RdYlGn",
                text=[[f"{val:.2%}" for val in row] for row in comparison["similarity_matrix"]],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="유사도")
            ))
            fig_heatmap.update_layout(
                title="프로파일 유사도 히트맵",
                height=400
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("비교하려면 최소 2개 이상의 프로파일을 선택하세요.")


if __name__ == "__main__":
    import time
    main()
