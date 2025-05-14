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
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def display_detected_frames(conf, model, st_frame, video_capture, display_tracking=True, tracker=None):
    # Initialize variables for tracking and FPS
    track_history = defaultdict(lambda: [])
    frame_count = 0
    fps = 0
    fps_update_interval = 1
    last_fps_update = time.time()

    # Loop through video frames
    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        # Resize frame for consistency
        frame = cv2.resize(frame, (720, int(720 * (9/16))))

        # Perform tracking or object detection
        if display_tracking:
            results = model.track(frame, conf=conf, persist=True, tracker=tracker)
        else:
            results = model.predict(frame, conf=conf)

        unique_obj_ids = set()
        frame_count += 1

        # Calculate FPS
        current_time = time.time()
        if current_time - last_fps_update >= fps_update_interval:
            fps = frame_count / (current_time - last_fps_update)
            frame_count = 0
            last_fps_update = current_time

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]
            confidences = result.boxes.conf.cpu().numpy()
            unique_obj_ids.update(track_ids)

            annotated_frame = result.plot()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                # Draw tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

                # Display tracking and detection information
                obj_count_text = f"{len(unique_obj_ids)} Objects Detected."
                confidence_text = f"Confidence: {confidences[0]:.2f}"
                fps_text = f"Video FPS: {fps:.2f} fps"
                tracking_detection_text = "Tracking : ON" if display_tracking else "Tracking : OFF"
                pose_detection_text = "Pose Detection : ON"

                cv2.putText(annotated_frame, f"{class_names[0]} ID: {track_id}", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, confidence_text, (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, obj_count_text, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, fps_text, (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, tracking_detection_text, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, pose_detection_text, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)

            # Display the annotated frame
            st_frame.image(annotated_frame, caption='Water SKI AI Model', channels="BGR", use_column_width=True)
            

import os

def play_stored_video(conf, model):
    # Ensure the videos directory exists
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)

    # Upload video file
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    # Save the uploaded video to the videos directory
    if uploaded_file is not None:
        video_path = os.path.join(videos_dir, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        settings.VIDEOS_DICT["Uploaded Video"] = video_path

    # Select the video source
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    # Get display options for tracking
    is_display_tracker, tracker = display_tracker_options()

    # Display the selected video
    video_path = settings.VIDEOS_DICT.get(source_vid)
    if source_vid == "Uploaded Video":
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
    else:
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()

    if video_bytes:
        st.video(video_bytes)

    # Button to start detection
    if st.sidebar.button('Run AI Model'):
        try:
            vid_cap = cv2.VideoCapture(video_path)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    display_detected_frames(conf, model, st_frame, vid_cap, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))