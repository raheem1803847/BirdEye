#!/usr/bin/env python
# coding: utf-8

# In[9]:
import concurrent.futures
import math
import cv2
import dlib
import numpy as np
import pyautogui
from PIL import Image
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor as ep
import time
# In[10]:

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
font = cv2.FONT_HERSHEY_PLAIN
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
classNames = []
classFile ="coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("n").split("n")

    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

# # Face detection
detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces == ():
        return None

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


# In[12]:


def get_gaze_ratio(eye_points, facial_landmarks):
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



                if left_length<200:
                    print("looking right")
                if left_length>400:
                    print("looking left")



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
findMesh=mpFaceMesh(drawMesh=False)

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


thres = 0.45
nms_threshold = 0.2
video_capture = cv2.VideoCapture(0)
video = VideoWriter('webcam2.avi', VideoWriter_fourcc(*'MP42'), 5.0, (640, 480))
duration = 45
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor()as ex:
    while True:
        _, frame = video_capture.read()
        _, frame2 = video_capture.read()
        stream_ok, frame = video_capture.read()
        new_frame = np.zeros((500, 500, 3), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facesMeshLM=ex.submit(findMesh.Marks(frame2))
        faces =detector(gray)
        face=face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv2.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            # Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension 128x128x3 into 1x128x128x3
            img_array = np.expand_dims(img_array, axis=0)
        else:
         cv2.putText(frame,"No Face Found",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        for face in faces:
            landmarks = predictor(gray, face)
            # Gaze detection
            gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
            if gaze_ratio <= 1:
                cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                new_frame[:] = (0, 0, 255)
            elif 1 < gaze_ratio < 2.11:
                cv2.putText(frame, "Left", (50, 100), font, 2, (0, 0, 255), 3)
            else:
                new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)

    ############################################################

        classIds, confs, bbox = net.detect(frame, confThreshold=thres)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId == 77:
                    print("alert :mobile phone is detected")

        if stream_ok:
            # write frame to the video file
            video.write(frame)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == 27: break
        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed > 45:
            break


video_capture.release()
cv2.destroyAllWindows()