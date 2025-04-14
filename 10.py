import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import av
import time
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# 配置参数
MODEL_PATH = "best.pt"
ALARM_SOUND = "alarm.mp3"
MAX_HISTORY = 5
CONF_THRESHOLD = 0.5
ALARM_DURATION = 1


# 初始化模型
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()


# 自定义样式
def set_custom_style():
    st.markdown(f"""
        <style>
            .main {{ background: #f8f9fa; }}
            .stAlert {{ border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .video-container {{
                position: relative;
                padding: 20px;
                background: white;
                border-radius: 15px;
                margin: 15px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }}
            .status-badge {{
                position: absolute;
                top: 25px;
                right: 25px;
                z-index: 100;
                padding: 8px 15px;
                border-radius: 20px;
                font-weight: bold;
            }}
            #alarm-audio {{ display: none; }}
        </style>
    """, unsafe_allow_html=True)


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.last_update = 0
        self.danger_start_time = None
        self.alarm = False
        self.alarm_triggered = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        if current_time - self.last_update > 0.1:
            results = self.model.predict(img, conf=CONF_THRESHOLD)
            annotated_frame = results[0].plot()

            detected_classes = []
            for box in results[0].boxes:
                if box.conf > CONF_THRESHOLD:
                    class_id = int(box.cls)
                    detected_classes.append(self.model.names[class_id])

            current_danger = any(cls in ['Fire', 'smoke'] for cls in detected_classes)

            if current_danger:
                if self.danger_start_time is None:
                    self.danger_start_time = current_time
                else:
                    duration = current_time - self.danger_start_time
                    if duration >= ALARM_DURATION and not self.alarm_triggered:
                        self.alarm = True
                        self.alarm_triggered = True
            else:
                self.danger_start_time = None
                self.alarm = False
                self.alarm_triggered = False

            self.last_update = current_time
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
        return frame


def main():
    st.set_page_config(
        page_title="基于YOLO的火灾检测系统设计与实现",
        page_icon="🔥",
        layout="wide"
    )
    set_custom_style()
    model = load_model()

    # 初始化会话状态
    session_defaults = {
        "history": [],
        "video_processing": False,
        "current_mode": None,
        "audio_permitted": False
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 预加载音频
    st.markdown(f'''
        <audio id="alarmAudio" controls style="display:none;">
            <source src="{ALARM_SOUND}" type="audio/mpeg">
        </audio>
    ''', unsafe_allow_html=True)

    st.title("🔥 基于YOLO的火灾检测系统设计与实现")
    st.markdown("---")

    with st.sidebar:
        st.header("控制面板")
        detection_mode = st.radio(
            "检测模式",
            ["图片检测", "视频检测", "实时检测"],
            index=0
        )
        st.markdown("---")
        st.subheader("检测历史")
        for idx, item in enumerate(st.session_state.history[-MAX_HISTORY:]):
            status_color = "#dc3545" if item["has_Fire"] else "#28a745"
            st.markdown(f"""
                <div class="video-container">
                    <div class="status-badge" style="background: {status_color}; color: white;">
                        {item['result']}
                    </div>
                    <div class="stats">
                        时间: {item['time']}<br>
                        处理: {item['process_time']:.2f}s<br>
                        帧数: {item.get('frames', 1)}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        # 修改点1：使用st.rerun()
        if st.button("清空历史"):
            st.session_state.history = []
            st.rerun()

    if detection_mode != st.session_state.current_mode:
        st.session_state.video_processing = False
        st.session_state.current_mode = detection_mode

    if detection_mode == "图片检测":
        handle_image_detection(model)
    elif detection_mode == "视频检测":
        handle_video_detection(model)
    elif detection_mode == "实时检测":
        handle_realtime_detection()


def handle_image_detection(model):
    st.subheader("图片检测")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"], key="image_uploader")

    if uploaded_file:
        if not os.path.exists(ALARM_SOUND):
            st.error(f"警报音频文件不存在: {ALARM_SOUND}")
            return

        start_time = time.time()
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("正在分析..."):
            results = model.predict(img, conf=CONF_THRESHOLD)
            annotated_img = results[0].plot()

            detected_classes = []
            for box in results[0].boxes:
                if box.conf > CONF_THRESHOLD:
                    detected_classes.append(model.names[int(box.cls)])

            fire_detected = any(cls in ['Fire', 'smoke'] for cls in detected_classes)
            process_time = time.time() - start_time

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "has_Fire": fire_detected,
                "result": "发现危险" if fire_detected else "环境安全",
                "process_time": process_time,
                "type": "image"
            }
            st.session_state.history.append(record)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, channels="BGR", caption="原始图片")
            with col2:
                st.image(annotated_img, channels="BGR", caption="检测结果")

            if fire_detected:
                st.error("## 🚨 检测到火灾危险！")
                st.markdown('''
                    <script>
                        document.getElementById('alarmAudio').play();
                    </script>
                ''', unsafe_allow_html=True)
            else:
                st.success("## ✅ 环境安全")


def handle_video_detection(model):
    st.subheader("视频检测")
    uploaded_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"], key="video_uploader")

    if uploaded_file and not st.session_state.video_processing:
        if not st.session_state.audio_permitted:
            with st.expander("⚠️ 需要音频播放权限"):
                st.write("请点击下方按钮允许系统播放警报声音")
                # 修改点2：使用st.rerun()
                if st.button("允许播放警报"):
                    st.session_state.audio_permitted = True
                    st.rerun()
            return

        if not os.path.exists(ALARM_SOUND):
            st.error(f"警报音频文件不存在: {ALARM_SOUND}")
            return

        st.session_state.video_processing = True
        video_path = f"temp_{uploaded_file.name}"

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        required_frames = max(1, int(fps * ALARM_DURATION))

        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        alarm_zone = st.empty()
        consecutive_danger_frames = 0
        alarm_triggered = False
        alarm_flag = False
        processed_frames = 0
        start_time = time.time()

        while st.session_state.video_processing and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=CONF_THRESHOLD)
            annotated_frame = results[0].plot()

            detected_classes = []
            for box in results[0].boxes:
                if box.conf > CONF_THRESHOLD:
                    detected_classes.append(model.names[int(box.cls)])

            fire_detected = any(cls in ['Fire', 'smoke'] for cls in detected_classes)

            if fire_detected:
                consecutive_danger_frames += 1
                duration = consecutive_danger_frames / fps

                if consecutive_danger_frames >= required_frames:
                    if not alarm_triggered:
                        st.markdown('''
                            <script>
                                document.getElementById('alarmAudio').play();
                            </script>
                        ''', unsafe_allow_html=True)
                        alarm_triggered = True
                        alarm_flag = True
                    alarm_zone.error(f"## 🚨 持续危险！已持续 {duration:.1f} 秒")
                else:
                    alarm_zone.warning(f"## ⚠️ 检测到危险！已持续 {duration:.1f} 秒")
            else:
                consecutive_danger_frames = 0
                alarm_triggered = False
                alarm_zone.empty()

            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
            progress_bar.progress((processed_frames + 1) / total_frames)
            processed_frames += 1

        process_time = time.time() - start_time
        cap.release()
        st.session_state.video_processing = False

        if alarm_flag:
            st.error(f"## 🚨 检测到持续{ALARM_DURATION}秒的火灾危险！")
        else:
            st.success("## ✅ 视频分析完成，未发现持续危险")

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "has_Fire": alarm_flag,
            "result": "发现危险" if alarm_flag else "环境安全",
            "process_time": process_time,
            "frames": processed_frames,
            "type": "video"
        }
        st.session_state.history.append(record)

        # 修改点3：使用st.rerun()
        if st.button("停止检测"):
            st.session_state.video_processing = False
            st.rerun()


def handle_realtime_detection():
    st.subheader("实时检测")
    webrtc_ctx = webrtc_streamer(
        key="realtime-fire-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        alarm = webrtc_ctx.video_processor.alarm
        if alarm:
            st.markdown("""
            <div class="video-container">
                <div class="status-badge" style="background: #dc3545; color: white;">
                    🚨 持续检测到火灾危险！
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f'''
                <script>
                    document.getElementById('alarmAudio').play();
                </script>
            ''', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="video-container">
                <div class="status-badge" style="background: #28a745; color: white;">
                    ✅ 环境安全
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()