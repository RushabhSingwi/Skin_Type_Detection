import traceback

import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np

from RealTimeSkinDetection import process_video_frame, get_camera, load_and_prep

IMAGE_SHAPE = (224, 224)
classes = ["Dry Skin", "Oily Skin"]

st.title("Mirror.AI")


@st.cache_resource
def load_model():
    try:
        is_model = tf.keras.models.load_model("RealTimeDetections.h5")
        return is_model, None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, str(e)


model, error_message = load_model()

if model is None:
    st.error("Error loading the model. Please check the model file.")
    st.stop()


def load_and_prep_image(image):
    try:
        # Convert byte content to image
        image = Image.open(BytesIO(image))
        # Convert image to numpy array
        image = np.array(image)
        # Ensure image is in RGB format
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 4:
            image = image[..., :3]  # Keep only RGB channels
        # Resize image and normalize
        image = cv2.resize(image, IMAGE_SHAPE, interpolation=cv2.INTER_AREA)
        image = image / 255.0  # Normalize to [0, 1]
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


# def url_uploader():
#     st.text("https://i.ibb.co/Tg62Zjv/oily-6.jpg")
#     path = st.text_input("Enter image URL to classify...", "https://i.ibb.co/Tg62Zjv/oily-6.jpg")
#     if path:
#         try:
#             content = requests.get(path).content
#             st.write("Predicted Skin type :")
#             with st.spinner("Classifying....."):
#                 img = load_and_prep_image(content)
#                 if img is not None:
#                     label = model.predict(tf.expand_dims(img, axis=0))
#                     st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
#                     image = Image.open(BytesIO(content))
#                     st.image(image, caption="Classifying the Skin", use_column_width=True)
#         except Exception as e:
#             st.error("Error processing image from URL: {}".format(e))


def camera_input():
    st.title('Camera Input')
    cap = get_camera()

    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return

    frame_placeholder = st.empty()
    capture_button = st.button("Capture Image")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error('Error: Failed to capture frame.')
            break

        frame, bbox = process_video_frame(frame)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

            frame_placeholder.image(frame, channels="BGR", caption="Video Feed", use_column_width=True)

            if capture_button:
                img_pred = frame[y:y + h, x:x + w]
                st.write("Predicted Skin type :")
                with st.spinner("Classifying....."):
                    try:
                        img = load_and_prep(img_pred)
                        if img is not None:
                            label = model.predict(tf.expand_dims(img, axis=0))
                            st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
                        st.write("")
                        st.image(img_pred, caption="Classifying the Skin", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def file_uploader():
    file = st.file_uploader("Upload file", type=["png", "jpeg", "jpg"])
    if not file:
        st.info("Upload a picture of the skin you want to predict.")
        return

    content = file.getvalue()
    st.write("Predicted Skin type :")
    with st.spinner("Classifying....."):
        img = load_and_prep_image(content)
        if img is not None:
            label = model.predict(tf.expand_dims(img, axis=0))
            st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption="Classifying the Skin", use_column_width=True)


option = st.radio("Choose upload method:", ("Camera", "File"))

if option == 'Camera':
    camera_input()
else:
    file_uploader()
