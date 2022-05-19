import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

#### LOAD MODEL
## Coco Names
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
                    print('alert!!!')
                classIds.append(classId)
                confs.append(float(confidence))



    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x,y,w,h= box [0],box[1],box[2],box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames [classIds[i]].upper()} {int(confs[i]*100)}%',
        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
        success, img = cap.read()
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0,0,0],1,crop=False)      ###network accept input in type blob, so change image to blob here ##whT, whT=>width and wight and target
        net.setInput(blob)
        layersNames = net.getLayerNames()  ###names of all our layers(extraction layer , out put layers)
        outputNames=[]

        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs, img)

        cv2.imshow('Image', img)
        cv2.waitKey(1)                  ###delay in reading one milli sec
