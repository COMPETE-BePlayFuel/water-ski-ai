from ultralytics import YOLO
import streamlit as st
import cv2
import time
import settings
from collections import defaultdict
import numpy as np

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, video_source, is_display_tracking=None, tracker=None):
    # Initialize variables for tracking and FPS
    track_history = defaultdict(lambda: [])
    frame_count = 0
    fps = 0
    fps_update_interval = 1
    last_fps_update = time.time()

    while video_source.isOpened():
        success, frame = video_source.read()
        if not success:
            break

        # Resize frame for consistency
        frame = cv2.resize(frame, (720, int(720 * (9 / 16))))

        # Perform tracking or object detection
        if is_display_tracking:
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
                tracking_detection_text = "Tracking: ON" if is_display_tracking else "Tracking: OFF"
                pose_detection_text = "Pose Detection: ON"

                cv2.putText(annotated_frame, f"{class_names[0]} ID: {track_id}", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, confidence_text, (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, obj_count_text, (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, fps_text, (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, tracking_detection_text, (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)
                cv2.putText(annotated_frame, pose_detection_text, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (194, 247, 50), 2)

            # Display the annotated frame
            st_frame.image(annotated_frame, caption='Water SKI AI Model', channels="BGR", use_column_width=True)

    # Release resources
    video_source.release()

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose first video file...", settings.MULTIPLE_VIDEOS_DICT.keys())
    source_vid_2 = st.sidebar.selectbox("Choose second video file...", settings.MULTIPLE_VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()

    col1, col2 = st.columns(2)

    # Display selected videos as preview
    with col1:
        with open(settings.MULTIPLE_VIDEOS_DICT.get(source_vid), 'rb') as video_file:
            video_bytes = video_file.read()
            if video_bytes:
                st.video(video_bytes)

    with col2:
        with open(settings.MULTIPLE_VIDEOS_DICT.get(source_vid_2), 'rb') as video_file_2:
            video_bytes_2 = video_file_2.read()
            if video_bytes_2:
                st.video(video_bytes_2)

    if st.sidebar.button('Detect Video Objects'):
        try:
            # Open video captures for both selected videos
            vid_cap_1 = cv2.VideoCapture(str(settings.MULTIPLE_VIDEOS_DICT.get(source_vid)))
            vid_cap_2 = cv2.VideoCapture(str(settings.MULTIPLE_VIDEOS_DICT.get(source_vid_2)))

            # Create placeholders for the video frames
            st_frame_1 = col1.empty()
            st_frame_2 = col2.empty()

            # Process both videos in parallel without threading
            while vid_cap_1.isOpened() or vid_cap_2.isOpened():
                if vid_cap_1.isOpened():
                    _display_detected_frames(conf, model, st_frame_1, vid_cap_1, is_display_tracker, tracker)

                if vid_cap_2.isOpened():
                    _display_detected_frames(conf, model, st_frame_2, vid_cap_2, is_display_tracker, tracker)

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
