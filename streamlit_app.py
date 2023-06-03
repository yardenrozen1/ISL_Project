
#טעינת ספריות
import cv2
import streamlit as st
import mediapipe as mp
import tensorflow as tf
import numpy as np
from PIL import ImageFont, ImageDraw, Image

#פעולות
def pred_letter(img ,select):
    """
    פעולה מתקבלת תמונה ומחירה של המודל ומחזירה את החיזוי של המודל
    return: מספר המציין את חיזוי המודל
    """
    try:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #מעביר את התמונה מגווי RGB לגווני אפור
    except:
        return 24
    else:
        img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_CONSTANT) #מבצע טשטוש לתמונה על מנת להפחית את הרעש של תמונה
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # מסמן על התמונה את הגבולות של היד
        if (img.shape[0] > img.shape[1]): #מגדיל את התמונה כך שכאשר משנים גודל לא יאבדו הפרופורציות
            whiteDim = img.shape[0]
            whiteImg = np.ones((whiteDim, whiteDim), np.uint8) * 255
            move = int((whiteDim - img.shape[1]) / 2)
            whiteImg[0:img.shape[0], move: img.shape[1] + move] = img
            img = whiteImg

        elif (img.shape[0] < img.shape[1]):
            whiteDim = img.shape[1]
            whiteImg = np.ones((whiteDim, whiteDim), np.uint8) * 255
            move = int((whiteDim - img.shape[0]) / 2)
            whiteImg[move:img.shape[0] + move, 0: img.shape[1]] = img
            img = whiteImg
        if select == "ResNet Model": # אם נבחר המודל ResNet צריך להעביר לגוונים צבעונים
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (imgDim, imgDim)) # משנה את הגודל
        img = np.float32(img) / 255.
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img) #שולח למודל לזיהוי
        return pred.argmax(axis = 1)[0] # מחזיר את חיזוי

def is_final_letter(q):
    """
    פעולה שמקבלת אות ומחזירה אמת אם האות היא סופית אחרת מחזירה שקר
    :param q: אות
    :return: Boolean
    """
    l = ("מ","נ","כ","פ","צ")
    if(q in l):
        return True
    else:
        return False

def write_sentence(sentence, pred):
    """
    פעולה המחקבלת משפט וחיזוי של המודל ומוסיפה את החיזוי למשפט
    :param sentence: המשפט עד עכשיו
    :param pred: חיזוי המודל
    :return: המשפט אחרי ההוספה
    """
    if pred != 24:
        if pred == 22:
            if len(sentence) > 0:
                if is_final_letter(sentence[-1]):
                    c = chr(ord(sentence[-1]) - 1)
                    sentence = sentence[:-1]
                    sentence += c
                sentence += "_"
        else:
            if pred == 23:
                if len(sentence) > 0:
                    sentence = sentence[:-1]
            else:
                if len(sentence) > 0 and sentence[-1] == "_":
                    sentence = sentence[:-1]
                    sentence += " "
                l = "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת".split()
                c = l[pred]
                sentence += c
    return sentence

#משתנים
imgDim =128 #גודל התמונה
otiyot = "א ב ג ד ה ו ז ח ט י כ ל מ נ ס ע פ צ ק ר ש ת".split() #הרשימה של האותיות
font = ImageFont.truetype(font='fonts/arial.ttf', size=40,  layout_engine=ImageFont.LAYOUT_RAQM) #פונט Arial

st.title("זיהוי שפת הסימנים") #הוספת כותרת
select = st.selectbox("בחר מודל"
             ,["CNN12 Model", "ResNet Model"]) # הוספת בחירת מודל

if select == "CNN12 Model": # בודק איזה מודל נבחר
    model = tf.keras.models.load_model(r"C:\Users\97254\PycharmProjects\ISL_Project\modle1_CNN12_1.h5")
else:
    model = tf.keras.models.load_model(r"C:\Users\97254\PycharmProjects\ISL_Project\modle2_inception1.h5")

col1, col2 = st.columns(2) # מחלק את האזור ל2 עמודות

#טוען את זיהוי הידיים
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

FRAME_WINDOW = col1.image([], use_column_width=True)
camera = cv2.VideoCapture(0) #פתיחת מצלמה
_, frame = camera.read()
frame = cv2.flip(frame,1) #הפיכה אופקית של התמונה
h, w, c = frame.shape

counter = 0
last = -1
sentence = ""
sen = col2.empty()

while camera.isOpened():
    _, frame = camera.read()
    frame = cv2.flip(frame,1)
    h, w, c = frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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


            pred = pred_letter(img, select)
            x_diff = x_max - x_min
            y_diff = y_max - y_min
            cv2.rectangle(frame, (x_min - 25 , y_min - 25), (x_max + 25 , y_max + 25), (0, 255, 0), 2)
            if pred == last:
                counter += 1
                if counter == 15:
                    sentence = write_sentence(sentence, pred)
                    sen.header(sentence)
                    counter = 0
            else:
                last = pred
            if pred == 22:
                tx = "חוור"
            elif pred == 23:
                tx = "קחמ"
            elif pred == 24:
                tx = ""
            else:
               tx = otiyot[pred]
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x_min -20, y_min -20),tx, font = font,align = "center")
            frame = np.array(img_pil)
    FRAME_WINDOW.image(frame)


