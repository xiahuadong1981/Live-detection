#coding=utf-8
# 摄像头采集睁眼、闭眼的数据
# 导入包
import numpy as np
import os
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils
import pickle

# 队列（特征向量）
VECTOR_SIZE = 3
def queue_in(queue, data):
    ret = None
    if len(queue) >= VECTOR_SIZE:
        ret = queue.pop(0)
    queue.append(data)
    return ret, queue

# 采集数据前准备
def eye_aspect_ratio(eye):
    # print(eye)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

pwd = os.getcwd()
model_path = os.path.join(pwd, 'model')
shape_detector_path = os.path.join(model_path, 'shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_detector_path)

cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)

# 对应特征点的序号
RIGHT_EYE_START = 37 - 1
RIGHT_EYE_END = 42 - 1
LEFT_EYE_START = 43 - 1
LEFT_EYE_END = 48 - 1

# 采集眼睛睁开时的样本
print('准备好睁着眼睛收集图片')
print('按s开始收集图像.')
print('按e键停止采集图像.')
print('按q退出')
flag = 0
txt = open('train_open.txt', 'w')
data_counter = 0
ear_vector = []
while(1):
    ret, frame = cap.read()
    # 功能：等待x ms后，才允许用户按键触发，如果用户没有按下 键,则接续等待(循环)
    # 参数：x：等待x ms
    # 返回值：返回按下按键的ASCII码，否则返回-1
    key = cv2.waitKey(1)
    if key & 0xFF == ord("s"):
        print('Start collecting images.')
        flag = 1
    elif key & 0xFF == ord("e"):
        print('Stop collecting images.')
        flag = 0
    elif key & 0xFF == ord("q"):
        print('quit')
        break

    if flag == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
            # points = shape.parts()
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # print('leftEAR = {0}'.format(leftEAR))
            # print('rightEAR = {0}'.format(rightEAR))

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            ret, ear_vector = queue_in(ear_vector, ear)
            print(ear_vector)
            if(len(ear_vector) == VECTOR_SIZE):
                # print(ear_vector)
                # input_vector = []
                # input_vector.append(ear_vector)
                # 功能：向文件中写入指定字符串
                # 说明：如果文件打开模式带 b，那写入文件内容时，
                #       str (参数)要用 encode 方法转为 bytes 形式，
                #       否则报错：TypeError: a bytes-like object is required, not 'str'。
                # 参数：str：要写入文件的字符串。
                # 返回值：写入的字符长度。
                txt.write(str(ear_vector))
                txt.write('\n')
                data_counter += 1
                print(data_counter)

            cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("frame", frame)
txt.close()

# 采集眼睛闭合时的样本
print('-'*40)
print('准备好闭上眼睛收集图像')
print('按s开始收集图像.')
print('按e键停止采集图像.')
print('按q退出')
flag = 0
txt = open('train_close.txt', 'w')
data_counter = 0
ear_vector = []
while(1):
    ret, frame = cap.read()
    key = cv2.waitKey(1)
    if key & 0xFF == ord("s"):
        print('Start collecting images.')
        flag = 1
    elif key & 0xFF == ord("e"):
        print('Stop collecting images.')
        flag = 0
    elif key & 0xFF == ord("q"):
        print('quit')
        break

    if flag == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
            # points = shape.parts()
            leftEye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
            rightEye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # print('leftEAR = {0}'.format(leftEAR))
            # print('rightEAR = {0}'.format(rightEAR))

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            ret, ear_vector = queue_in(ear_vector, ear)
            if(len(ear_vector) == VECTOR_SIZE):
                # print(ear_vector)
                # input_vector = []
                # input_vector.append(ear_vector)

                txt.write(str(ear_vector))
                txt.write('\n')

                data_counter += 1
                print(data_counter)

            cv2.putText(frame, "EAR:{:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("frame", frame)
txt.close()

# 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
