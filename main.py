from PIL import Image, ImageEnhance
import sys
import numpy as np
import cv2
import math
import os

import shutil  # Библиотека для работы с файлами
import glob    # Расширение для использования Unix обозначений при задании пути к файлу

isRecord = False
# Тут будет храниться номер кадра
file_counter = 0
# Тут считаем кадры перед тем, как записать
frame_counter = 0
# Будем записывать каждый 20-й кадр
frame_step = 20
# Директория для записи кадров
videoDir = 'C:\\Users\\User\\Desktop\\cut_my_face(cv2)\\video'


# Класс для работы с видеопотоком с видеокамеры
frame = cv2.VideoCapture(1)

# Создаем объект для обнаружения лица
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Начинаем запись
def start_recording():
	global file_counter

	if os.path.isdir(videoDir):
		shutil.rmtree(videoDir+'/')  # удаляем каталог вместе с содержимым

	os.mkdir(videoDir)
	file_counter = 0


def save_frame(img):
	global file_counter

	file_name = "%05d" % file_counter
	cv2.imwrite(videoDir+'/'+file_name+".png", img)

	file_counter += 1


while True:
    # Получаем кадр
    status, image = frame.read()
    image_done = image.copy()

    faces = face.detectMultiScale(image, scaleFactor=1.8, minNeighbors=6, minSize=(110,110))

    for (x, y, w, h) in faces:
        cv2.rectangle(image_done, (x,y), (x+w,y+h), (0,255,64), 2)

        if isRecord and (len(faces) > 0) and (frame_counter == 0):
            image_face_frame = image[y:y + h, x:x + w]
            image_face_frame = cv2.cvtColor(image_face_frame, cv2.COLOR_BGR2GRAY)
            save_frame(image_face_frame)
        
    frame_counter += 1
    if frame_counter >= frame_step:
        frame_counter = 0

    cv2.imshow("Face", image_done)

    k = cv2.waitKey(30)

    # Обрабатываем нажатие клавиши ESC
    if k == 27:
        break

    # Обрабатываем нажатие клавиши r, которая включает или выключает запись видео
    if k == 114:
        if not isRecord:
            start_recording()  # Начинаем запись
            isRecord = True
        else:
            isRecord = False

frame.release()
cv2.destroyAllWindows()