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
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
                if left_length>400:
                    print("looking left")
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
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.myFace.process(frameRGB)
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
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.myPose.process(frameRGB)
        poseLandmarks=[]
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                poseLandmarks.append((int(lm.x*width),int(lm.y*height)))
        return poseLandmarks


width=1280
height=720
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))


findFace=mpFace()
# findPose=mpPose()
findMesh=mpFaceMesh(drawMesh=True)

font=cv2.FONT_HERSHEY_SIMPLEX
fontColor=(0,0,255)
fontSize=.1
fontThick=1

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',width+50,0)
cv2.resizeWindow('Trackbars',400,150)


while True:
    ignore,  frame = cam.read()
    frame=cv2.resize(frame,(width,height))
    faceLoc=findFace.Marks(frame)
    # poseLM=findPose.Marks(frame)
    facesMeshLM=findMesh.Marks(frame)
    # if poseLM != []:
    #     for ind in [13,14,15,16]:
    #         cv2.circle(frame,poseLM[ind],20,(0,255,0),-1)

    for face in faceLoc:
        cv2.rectangle(frame,face[0],face[1],(255,0,0),3)

    for faceMeshLM in facesMeshLM:
        cnt=0
        for lm in faceMeshLM:
            cv2.putText(frame,str(cnt),lm,font,fontSize,fontColor,fontThick)
            cnt=cnt+1

    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam',0,0)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()