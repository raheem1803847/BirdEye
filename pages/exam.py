#!/usr/bin/env python
# coding: utf-8

# In[1]:
import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import time





# open the webcam video stream
webcam = cv2.VideoCapture(0)
freq = 44100
duration = 45
start_time = time.time()


# open output video file stream
def startrecord():
    video = VideoWriter('webcam.mp4', VideoWriter_fourcc(*'MP42'), 25.0, (640, 480))
    while True:
        # get the frame from the webcam
        stream_ok, frame = webcam.read()

        # if webcam stream is ok
        if stream_ok:
            # display current frame
            cv2.imshow('Webcam', frame)

            # write frame to the video file
            video.write(frame)
            # escape condition
            if cv2.waitKey(1) & 0xFF == 27: break
            end_time = time.time()
            elapsed = end_time - start_time
            if elapsed > 7:
                break

    # clean ups
    cv2.destroyAllWindows()

    # release web camera stream
    webcam.release()

    # release video output file stream
    video.release()

if __name__ == '__main__':
    startrecord()