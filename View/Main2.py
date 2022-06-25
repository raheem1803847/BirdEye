#!/usr/bin/env python
# coding: utf-8

# In[9]:
import threading
import warnings
from math import hypot
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import multiprocessing
import time
import mediapipe as mp
import dlib
import pyautogui
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import sys
import cv2 as cv
import concurrent.futures
import numpy as np
from datetime import datetime
from reportlab.pdfgen.canvas import Canvas
#
now = datetime.now()

cheating = []

# newselect = cheating.append(["aaaa","type"])
# newselect2= a.append(["bbbb","type"])
# print(a)



warnings.filterwarnings("ignore")


font = cv2.FONT_HERSHEY_PLAIN

whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
classesFile = "coco.names"
classNames =[]
with open(classesFile, 'rt') as f:
    classNames = f.read().splitlines()

## Model Files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)   #create network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)



# # Face detection

# In[11]:


detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = load_model('facefeatures_new_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):#done
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    counter=0

    if faces == ():
        return None

    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]
        counter = counter + 1
        if counter >= 2:
            print("more than one person found")
            current_time = now.strftime("%H:%M:%S")
            cheating.append(["more than one person found", current_time])#type-time


    return cropped_face




def get_gaze_ratio(eye_points, facial_landmarks):#done#eye tracking
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


# # Face gestures

# In[13]:


import cv2
import math
class mpFaceMesh:
    import mediapipe as mp
    def __init__(self,still=False,numFaces=3,tol1=.5,tol2=.5,drawMesh=True):
        self.myFaceMesh=self.mp.solutions.face_mesh.FaceMesh()
        self.myDraw=self.mp.solutions.drawing_utils
        self.draw=drawMesh
    def Marks(self,frame):
        global width
        global height
        drawSpecCircle=self.myDraw.DrawingSpec(thickness=0,circle_radius=0,color=(0,0,255))
        drawSpecLine=self.myDraw.DrawingSpec(thickness=1,circle_radius=2,color=(255,0,0))
        frameRGB=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
        results=self.myFaceMesh.process(frameRGB)
        facesMeshLandmarks=[]
        left_length=0
        right_length = 0
        upLeft_length = 0
        upRight_length = 0

        if results.multi_face_landmarks !=None:
            for faceMesh in results.multi_face_landmarks:
                faceMeshLandmarks=[]
                for lm in faceMesh.landmark:
                    loc=(int(lm.x*width),int(lm.y*height))
                    faceMeshLandmarks.append(loc)
                facesMeshLandmarks.append(faceMeshLandmarks)

                left_length=math.sqrt( (faceMeshLandmarks[0][0]-faceMeshLandmarks[49][0])**2+(faceMeshLandmarks[0][0]-faceMeshLandmarks[49][1])**2)   # left

                right_length= math.sqrt( (faceMeshLandmarks[0][0]-faceMeshLandmarks[279][0])**2+(faceMeshLandmarks[0][0]-faceMeshLandmarks[279][1])**2) # right

                upLeft_length =math.sqrt((faceMeshLandmarks[0][0] - faceMeshLandmarks[65][0]) ** 2 + (faceMeshLandmarks[0][0] - faceMeshLandmarks[65][1]) ** 2)  # up left

                upRight_length=math.sqrt((faceMeshLandmarks[0][0] - faceMeshLandmarks[295][0]) ** 2 + (faceMeshLandmarks[0][0] - faceMeshLandmarks[295][1]) ** 2) # up right


                if left_length<200:
                    print("looking right")
                    current_time = now.strftime("%H:%M:%S")
                    cheating.append(["looking right", current_time])  # type-time
                if left_length>360:
                    print("looking left")
                    current_time = now.strftime("%H:%M:%S")
                    cheating.append(["looking left", current_time])  # type-time
                # if upLeft_length<200:
                #     print("looking up left")
                # if upRight_length<200:
                #     print("looking up right")



                if self.draw==True:
                    self.myDraw.draw_landmarks(frame,faceMesh,self.mp.solutions.face_mesh.FACEMESH_TESSELATION,drawSpecCircle,drawSpecLine)
        return facesMeshLandmarks

class mpFace:   ###read the face and get the topLeft/bottomRight of detection box
    import mediapipe as mp
    def __init__(self):
        self.myFace=self.mp.solutions.face_detection.FaceDetection()
    def Marks(self,frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.myPose.process(frameRGB)
        poseLandmarks = []
        faceBoundBoxs=[]
        if results.detections != None:
            for face in results.detections:
                bBox=face.location_data.relative_bounding_box
                topLeft=(int(bBox.xmin*width),int(bBox.ymin*height))
                bottomRight=(int((bBox.xmin+bBox.width)*width),int((bBox.ymin+bBox.height)*height))
                faceBoundBoxs.append((topLeft,bottomRight))

        return faceBoundBoxs

class mpPose:
    import mediapipe as mp
    def __init__(self,still=False,upperBody=False, smoothData=True, tol1=.5, tol2=.5):
        self.myPose=self.mp.solutions.pose.Pose(still,upperBody,smoothData,tol1,tol2)
    def Marks(self,frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.myPose.process(frameRGB)
        poseLandmarks = []
        poseLandmarks=[]
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                poseLandmarks.append((int(lm.x*width),int(lm.y*height)))
        return poseLandmarks


width=1280
height=720



findFace=mpFace()
# findPose=mpPose()
findMesh=mpFaceMesh(drawMesh=True)


# # object detection

# In[14]:


def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox =[]         ##contain x-y depth and height
    classIds =[]
    confs = []
    for output in outputs:
        for det in output:
            scores = det [5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # print(classId)
            # print(confidence)
            if confidence > confThreshold:
                w, h = int(det[2]*wT) , int(det[3]*hT)
                x, y=int((det [0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append( [x,y,w,h])
                if classId==67:
                    print('cell phone detection')
                    current_time = now.strftime("%H:%M:%S")
                    cheating.append(["cell phone detection", current_time])  # type-time

                classIds.append(classId)
                confs.append(float(confidence))



    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x,y,w,h= box [0],box[1],box[2],box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames [classIds[i]].upper()} {int(confs[i]*100)}%',
        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)




# # Screen recording

# In[15]:


# Specify resolution
resolution = (1920, 1080)

# Specify video codec
codec = cv2.VideoWriter_fourcc(*"XVID")

# Specify name of Output file
filename = "Recording.avi"

# Specify frames rate. We can choose any 
# value and experiment with it
fps = 10.0
# Creating a VideoWriter object
out = cv2.VideoWriter(filename, codec, fps, resolution)


# # Multiprocessing

# In[26]:

t1=time.time()
face=threading.Thread(target=face_extractor)
eye=threading.Thread(target=get_gaze_ratio)
obj=threading.Thread(target=findObjects)
face.start()
eye.start()
obj.start()

face.join()
eye.join()

# In[ ]:
t2=time.time()
t3=t2-t1
print(t3)
video_capture = cv2.VideoCapture(0)
video = VideoWriter('webcam.avi', VideoWriter_fourcc(*'MP42'), 5.0, (640, 480))
while True:
    _, frame = video_capture.read()
    _, frame2 = video_capture.read()
    stream_ok, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesMeshLM=findMesh.Marks(frame2)
    faces = detector(gray)
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)


    else:
     cv2.putText(frame,"No Face Found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
     current_time = now.strftime("%H:%M:%S")
     cheating.append(["No Face Found", current_time])  # type-time

    for face in faces:#eye
        landmarks = predictor(gray, face)
        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
        if gaze_ratio <= 1:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            current_time = now.strftime("%H:%M:%S")
            cheating.append(["looking right", current_time])  # type-time
            new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio < 2.11:
            cv2.putText(frame, "Left", (50, 100), font, 2, (0, 0, 255), 3)
            current_time = now.strftime("%H:%M:%S")
            cheating.append(["looking left", current_time])  # type-time
        else:
            new_frame[:] = (255, 0, 0)
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)

    success, img = video_capture.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0,0,0],1,crop=False)      ###network accept input in type blob, so change image to blob here ##whT, whT=>width and wight and target
    net.setInput(blob)
    layersNames = net.getLayerNames()  ###names of all our layers(extraction layer , out put layers)
    outputNames=[]

    outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)
    img3 = pyautogui.screenshot()
############################################################screen and vedio rec
    # Convert the screenshot to a numpy array
    frame3 = np.array(img3)
    frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
    # Write it to the output file
    out.write(frame3)
##############################################################
#   duration=60
# recording = sd.rec(int(duration * freq),
#                 samplerate=freq, channels=2)
# Record audio for the given number of seconds
# sd.wait()
# This will convert the NumPy array to an audio
# file with the given sampling frequency
# write("recording0.wav", freq, recording)
    if stream_ok:
        # write frame to the video file
        video.write(frame)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

print(cheating)


video_capture.release()
cv2.destroyAllWindows()

