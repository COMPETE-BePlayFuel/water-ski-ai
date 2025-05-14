from ultralytics import YOLO
import time
import streamlit as st
import cv2
from collections import defaultdict
import settings
import numpy as np

def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("SKI AI Model", ('On', 'Off'))
    is_display_tracker = True if display_tracker == 'On' else False

    return is_display_tracker

def display_fps_options():
    display_fps = st.radio("Display FPS", ('On', 'Off'))
    is_display_fps = True if display_fps == 'On' else False

    return is_display_fps

def display_detection_results(results, col, current_second, speed_estimates):
    with col:
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        with st.expander(f"Detection Results for: {ordinal(current_second)} second:"):
            st.write("Degrees: {:.2f} inches".format(28))
            st.write("Detection Frames: {:.2f} inches".format(60))
            for track_id, speed in speed_estimates.items():
                st.write(f"Object ID {track_id}: Speed: {speed:.2f} km/h")

# Placeholder for real-world meters per pixel ratio
meters_per_pixel = 0.055  # Adjust this based on your scenario

def calculate_speed(track_history, track_id, meters_per_pixel, frame_interval):
    if len(track_history[track_id]) < 2:
        return 0.0  # Not enough points to calculate speed
    (x1, y1), (x2, y2) = track_history[track_id][-2], track_history[track_id][-1]
    distance_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance_meters = distance_px * meters_per_pixel
    speed_mps = distance_meters / frame_interval
    speed_kmh = speed_mps * 3.6
    return speed_kmh

def _display_frame_with_detections(conf, model, frame, fps,track_history, is_display_tracking=None,is_display_fps=False):
    unique_obj_ids = set()
    speed_estimates = {}
    frame_interval = 1.0 / fps if fps > 0 else 1.0
    # Resize frame for consistency
    frame = cv2.resize(frame, (720, int(720 * (9 / 16))))

    # Perform tracking or object detection
    if is_display_tracking:
        results = model.track(frame, conf=conf, persist=True)
    else:
        results = model.predict(frame, conf=conf)

    annotated_frame = frame.copy()  # Default frame to avoid issues if no results are found

    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[int(cls_id)] for cls_id in class_ids]
        confidences = result.boxes.conf.cpu().numpy()
        unique_obj_ids.update(track_ids)

        annotated_frame = result.plot()  # Redefine with detected results if present
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)

            points = np.array([(int(p[0]), int(p[1])) for p in track]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

            speed_kmh = calculate_speed(track_history, track_id, meters_per_pixel, frame_interval)
            speed_estimates[track_id] = speed_kmh
            # Display tracking and detection information
            obj_count_text = f"{len(unique_obj_ids)} Objects Detected."
            confidence_text = f"Confidence: {confidences[0]:.2f}" if confidences.size > 0 else "Confidence: N/A"
            # fps_text = f"Video FPS: {fps:.2f} fps"
            tracking_detection_text = "Tracking: ON" if is_display_tracking else "Tracking: OFF"
            pose_detection_text = "Pose Detection: ON"

            cv2.putText(annotated_frame, f"{class_names[0] if class_names else 'Object'} ID: {track_id}", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
            cv2.putText(annotated_frame, confidence_text, (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
            cv2.putText(annotated_frame, obj_count_text, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
            if is_display_fps:
                cv2.putText(annotated_frame, f"Video FPS: {fps:.2f} fps", (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
            cv2.putText(annotated_frame, tracking_detection_text, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
            cv2.putText(annotated_frame, pose_detection_text, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
            cv2.putText(annotated_frame, f"Speed: {speed_kmh:.2f} km/h", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)


    return annotated_frame, results, speed_estimates

def play_stored_video(conf, model):
    # Add file uploader for both video columns
    uploaded_video_1 = st.sidebar.file_uploader("Upload first video file...", type=["mp4", "avi", "mov"])
    uploaded_video_2 = st.sidebar.file_uploader("Upload second video file...", type=["mp4", "avi", "mov"])

    # Save uploaded videos to the 'videos' folder
    if uploaded_video_1 is not None:
        video_path_1 = f"videos/{uploaded_video_1.name}"
        with st.spinner('Uploading first video...'):
            with open(video_path_1, "wb") as f:
                f.write(uploaded_video_1.getbuffer())
    else:
        video_path_1 = settings.MULTIPLE_VIDEOS_DICT.get(st.sidebar.selectbox("Choose first video file...", settings.MULTIPLE_VIDEOS_DICT.keys()))

    if uploaded_video_2 is not None:
        video_path_2 = f"videos/{uploaded_video_2.name}"
        with st.spinner('Uploading second video...'):
            with open(video_path_2, "wb") as f:
                f.write(uploaded_video_2.getbuffer())
    else:
        video_path_2 = settings.MULTIPLE_VIDEOS_DICT.get(st.sidebar.selectbox("Choose second video file...", settings.MULTIPLE_VIDEOS_DICT.keys()))

    is_display_tracker = display_tracker_options()
    is_display_fps = display_fps_options()  # Get FPS display setting

    col1, col2 = st.columns(2)
    # Display selected videos as preview
    with col1:
        if uploaded_video_1 is not None:
            st.video(video_path_1)
        else:
            with open(video_path_1, 'rb') as video_file:
                video_bytes = video_file.read()
                if video_bytes:
                    st.video(video_bytes)

    with col2:
        if uploaded_video_2 is not None:
            st.video(video_path_2)
        else:
            with open(video_path_2, 'rb') as video_file_2:
                video_bytes_2 = video_file_2.read()
                if video_bytes_2:
                    st.video(video_bytes_2)

    if st.sidebar.button('Run Water SKI Model'):
        st.subheader("AI Comparison Analysis")
        user_input = st.text_area("Results will be available after processing the whole videos. Please wait!", "Hello testing", height=None)    

        try:
            # Open video captures for both selected videos
            vid_cap_1 = cv2.VideoCapture(video_path_1)
            vid_cap_2 = cv2.VideoCapture(video_path_2)

            # Initialize FPS calculation variables
            frame_count_1 = frame_count_2 = 0
            fps_1 = fps_2 = 0
            last_update_1 = last_update_2 = time.time()

            # Create placeholders for the video frames
            st_frame_1 = col1.empty()
            st_frame_2 = col2.empty()
            # Initialize results variables
            results_1 = None
            results_2 = None
            # Initialize start times right before the loop starts
            start_time_1 = time.time()
            start_time_2 = time.time()
            track_history_1 = defaultdict(list)
            track_history_2 = defaultdict(list)
            # Initialize speed estimates for both videos
            speed_estimates_1 = {}
            speed_estimates_2 = {}
            # Process frames by alternating between videos
            while vid_cap_1.isOpened() or vid_cap_2.isOpened():
                if vid_cap_1.isOpened():
                    success_1, frame_1 = vid_cap_1.read()
                    if success_1:
                        frame_count_1 += 1
                        current_time = time.time()
                        if current_time - last_update_1 >= 1:
                            fps_1 = frame_count_1 / (current_time - last_update_1)
                            frame_count_1 = 0
                            last_update_1 = current_time
                            display_detection_results(results_1, col1, int(current_time - start_time_1), speed_estimates_1)
                        annotated_frame_1, results_1, speed_estimates_1 = _display_frame_with_detections(
                            conf, model, frame_1, fps_1, track_history_1, is_display_tracker, is_display_fps
                        )
                        st_frame_1.image(annotated_frame_1, channels="BGR", use_column_width=True)
                    else:
                        vid_cap_1.release()

                if vid_cap_2.isOpened():
                    success_2, frame_2 = vid_cap_2.read()
                    if success_2:
                        frame_count_2 += 1
                        current_time = time.time()
                        if current_time - last_update_2 >= 1:
                            fps_2 = frame_count_2 / (current_time - last_update_2)
                            frame_count_2 = 0
                            last_update_2 = current_time
                            display_detection_results(results_2, col2, int(current_time - start_time_2), speed_estimates_2)
                        annotated_frame_2, results_2, speed_estimates_2 = _display_frame_with_detections(
                            conf, model, frame_2, fps_2, track_history_2, is_display_tracker, is_display_fps
                        )
                        st_frame_2.image(annotated_frame_2, channels="BGR", use_column_width=True)
                    else:
                        vid_cap_2.release()

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))