import cv2
import streamlit as st
import mediapipe as mp
import matplotlib as plt
import tensorflow as tf
import numpy as np
from PIL import ImageFont, ImageDraw, Image

s1 = st.empty()
s2 = st.empty()
s3 = st.empty()
s4 = st.empty()
s5 = st.empty()
s6 = st.empty()
s7 = st.empty()
imgDim =128
def pred_letter(img ,select):
    #img = img[2, :, :]
    s1.image(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s2.image(img)
    img = cv2.GaussianBlur(img, (5, 5), 3)
    s3.image(img)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    s4.image(img)
    ret, img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    s5.image(img)

    img = cv2.resize(img, (imgDim, imgDim))
    img = np.float32(img) / 255.
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)


    pred = model.predict(img)
    print(pred.argmax(axis = 1)[0])
    return pred.argmax(axis = 1)[0]

def is_final_letter(q):
    l = ("מ","נ","כ","פ","צ")
    if(q in l):
        return True
    else:
        return False

def write_sentence(sentence, pred):
    if pred != 24:
        if pred == 22:
            if len(sentence) > 0:
                if is_final_letter(sentence[-1]):
                    c = chr(ord(sentence[-1]) - 1)
                    sentence = sentence[:-1]
                    sentence += c
                sentence += "_"
        else:
            if len(sentence) > 0 and sentence[-1] == "_":
                sentence = sentence[:-1]
                sentence += " "
            elif pred == 23:
                if len(sentence) > 0:
                    sentence = sentence[:-1]
            else:
                l = "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת".split()
                print(l)
                c = l[pred]
                sentence += c
    return sentence

otiyot = "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת".split()
font = ImageFont.truetype(font='fonts/arial.ttf', size=20)
st.title("זיהוי שפת הסימנים")

select = st.selectbox("בחר מודל"
             ,["CNN12 Model", "ResNet Model"])
if select == "CNN12 Model":
    model = tf.keras.models.load_model(r'C:\Users\97254\Desktop\Project\modle1_CNN12_1.h5')
else:
    model = tf.keras.models.load_model(r"C:\Users\97254\Desktop\Project\modle2_inception1.h5")

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils


FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
_, frame = camera.read()
frame = cv2.flip(frame,1)
h, w, c = frame.shape

counter = 0
last = -1
sentence = ""
sen = st.empty()

while camera.isOpened():
    _, frame = camera.read()
    frame = cv2.flip(frame,1)
    h, w, c = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #hands, frame = detector.findHands(frame)
    #lmList, bbox = detector.findPosition(frame)

    result = hands.process(frame)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y *h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y


            img = frame[y_min - 25: y_max + 25, x_min - 25: x_max + 25]

            if(img.shape[0] > img.shape[1]):
                whiteDim = img.shape[0]
                whiteImg = np.ones((whiteDim,whiteDim,3), np.uint8) * 169
                move = int((whiteDim - img.shape[1])/2)
                whiteImg[0:img.shape[0],move: img.shape[1] + move] = img
                img = whiteImg

            elif (img.shape[0] < img.shape[1]):
                whiteDim = img.shape[1]
                whiteImg = np.ones((whiteDim, whiteDim, 3), np.uint8) * 169
                move = int((whiteDim - img.shape[0]) / 2)
                whiteImg[move:img.shape[0] + move, 0: img.shape[1]] = img
                img = whiteImg

            pred = pred_letter(img, select)
            x_diff = x_max - x_min
            y_diff = y_max - y_min
            cv2.rectangle(frame, (x_min - 25 , y_min - 25), (x_max + 25 , y_max + 25), (0, 255, 0), 2)
            if pred == last:
                counter += 1
                if counter == 15:
                    sentence = write_sentence(sentence, pred)
                    sen.header(sentence)
                    print(sentence)
                    counter = 0
            else:
                last = pred
            if pred < len(otiyot):
                tx = otiyot[pred]
            if pred == 22:
                tx = "רווח"
            else:
                tx = "מחק"
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            draw.text((2, 2),tx, font = font)
            img = np.array(img_pil)
    FRAME_WINDOW.image(frame)


