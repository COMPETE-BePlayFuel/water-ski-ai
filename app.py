# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper,helper3,helper_new,helper_copy 
#new,helper_deep,helper_2

# Setting page layout
st.set_page_config(
    page_title="Water SKI Model By Compete Co.",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Water SKI Model ML App🌊")
st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Water SKI Model Configurations")
# Add sidebar button and text area for user input
st.subheader("AI Comparison Analysis")
user_input = st.text_area("Results will be available after processing the whole videos. Please wait!", "", height=None)
# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Scoring Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Scoring Detection':
    model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
            
        else:
            if st.sidebar.button('Detect Fish'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results(Raw details)"):
                        for box in boxes:
                            st.write(box.data)
                    with st.expander("Detection Results:"):
                        boxes = res[0].boxes.xywh.cpu()
                        for box in boxes:
                          x, y, w, h = box
                          #st.write("Length of Fish: {} inches".format(h/10))                          
                          l=w/24
                          h=h/35
                          g=(w/24.5)*.90
                          if l>h:
                           st.write("Alignment of fish: X-Axis/Horizontally")
                           st.write("Length of Fish: {:.2f} inches".format(l))
                           st.write("Width/Height  of Fish: {:.2f} inches".format(h)) 
                           st.write("Girth of Fish: {:.2f} inches".format(l*.85))  
                           st.write("Estimated weight of Fish: {:.2f} lbs.".format((l**2)*g/1200))
                          else:
                           st.write("Alignment of fish: Y-Axis/Vertically")
                           st.write("Width/Height of Fish: {:.2f} inches".format(l))
                           st.write("Length of Fish: {:.2f} inches".format(h)) 
                           st.write("Girth of Fish: {:.2f} inches".format(h*.85))  
                           st.write("Estimated weight of Fish: {:.2f} lbs.".format(((h**2)*g)/120))
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)
elif source_radio == settings.MULTIPLE_VIDEOS_2:
    helper_2.play_stored_video(confidence, model)    
elif source_radio == settings.MULTIPLE_VIDEOS:
    helper_new.play_stored_video(confidence, model)
elif source_radio == settings.DEEP_ANALYSIS:
    helper_deep.play_stored_video(confidence, model)

else:
    st.error("Please select a valid source type!")



