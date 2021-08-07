import sys

import cv2
import numpy as np
import paddle
import matplotlib.pyplot as plt
from PIL import Image

from dataset import preprocess_input
from model import MiniXception, SimpleCNN


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')


def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model


def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)


def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)


def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)


def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
              font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors


# parameters for loading data and images
image_path = 'images/test_image.jpg'
detection_model_path = 'trained_models/detection_models/haarcascade_frontalface_default.xml'
gender_model_path = 'trained_models/gender_models/SimpleCNN/best.pdparams'
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (10, 10)

# loading models
paddle.set_device('cpu')
face_detection = load_detection_model(detection_model_path)
gender_model = SimpleCNN(3, 2)
gender_model.set_state_dict(paddle.load(gender_model_path))
gender_model.eval()

# loading images
rgb_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    face = rgb_image[y1:y2, x1:x2]

    try:
        face = cv2.resize(face, (64, 64))
    except:
        continue

    rgb_face = preprocess_input(face,True)
    rgb_face = rgb_face.transpose((2, 0, 1))
    # rgb_face = np.expand_dims(rgb_face, 0)
    rgb_face = np.expand_dims(rgb_face, 0)
    gender_prediction = gender_model(paddle.to_tensor(rgb_face))
    gender_label_arg = paddle.argmax(gender_prediction).numpy()[0]
    gender_text = gender_labels[gender_label_arg]

    if gender_text == gender_labels[0]:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)

rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('images/predicted_test_image.png', rgb_image)
