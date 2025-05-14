from ultralytics import YOLO
import time
import streamlit as st
import cv2
from collections import defaultdict
import settings
import numpy as np
import settings

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


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None):

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True,)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    with st.expander("Detection Results:"):
       boxes = res[0].boxes.xywh.cpu()
       for box in boxes:
         x, y, w, h = box
         #st.write("Length of Fish: {} inches".format(h/10))
         st.write("Length of Fish: {:.2f} inches".format(w/28))
         l=w/28
         g=(w/36)*.51
         st.write("Degrees:{:.2f} inches".format(h/60)) 
         st.write("Angles between Hips & Shoulder {:.2f} inches".format((w/36)*.51))  

def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose first video file...", settings.MULTIPLE_VIDEOS_DICT.keys())
    source_vid_2 = st.sidebar.selectbox("Choose second video file...", settings.MULTIPLE_VIDEOS_DICT.keys())
    is_display_tracker = display_tracker_options()
    
    col1, col2 = st.columns(2)

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
        st.subheader("AI Comparison Analysis")
        user_input = st.text_area("Results will be available after processing the whole videos. Please wait!", "Hello testing", height=None)
        try:
            vid_cap_1 = cv2.VideoCapture(str(settings.MULTIPLE_VIDEOS_DICT.get(source_vid)))
            vid_cap_2 = cv2.VideoCapture(str(settings.MULTIPLE_VIDEOS_DICT.get(source_vid_2)))
        # Create placeholders for the video frames outside the loop
            cols = st.columns(2)
            st_frame_1 = cols[0].empty()
            st_frame_2 = cols[1].empty()
            
            while vid_cap_1.isOpened() and vid_cap_2.isOpened():
                success_1, image_1 = vid_cap_1.read()
                success_2, image_2 = vid_cap_2.read()
                if success_1 and success_2:
                    with cols[0]:  # Ensure the first column context is used
                        _display_detected_frames(conf,
                                             model,
                                             st_frame_1,
                                             image_1,
                                             is_display_tracker,
                                             
                                             )
                    with cols[1]:  # Ensure the second column context is used
                        _display_detected_frames(conf,
                                             model,
                                             st_frame_2,
                                             image_2,
                                             is_display_tracker,
                                             
                                             )
                else:
                    vid_cap_1.release()
                    vid_cap_2.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
