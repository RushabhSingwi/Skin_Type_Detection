import cv2
import tensorflow as tf

import FaceDetectionModule as ftm

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
class_names = ["Dry Skin", "Oily Skin"]
IMG_SIZE = (224, 224)

# loading the model
model = tf.keras.models.load_model("RealTimeDetections.h5")
detector = ftm.FaceDetector(cascade_path)


def load_and_prep(file_path):
    img_path = tf.io.read_file(file_path)
    decoded_img = tf.io.decode_image(img_path)
    final_img = tf.image.resize(decoded_img, IMG_SIZE)
    return final_img


def process_video_frame(frame):
    frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
    frame, bboxs = detector.find_faces(frame)
    if bboxs:
        x, y, w, h = bboxs[0]
        return frame, (x, y, w, h)
    return frame, None


def get_camera():
    cap = cv2.VideoCapture(0)
    return cap
