import cv2
import numpy as np
videoframe=cv2.VideoCapture(0)

configurationfile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightfile = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(weightfile,configurationfile)

classnames=[]
objectfilename='objectlabels.txt'
with open(objectfilename,'rt') as f:
    classnames=f.read().rstrip('\n').split('\n')

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

font0scale =3
font=cv2.FONT_HERSHEY_PLAIN
while True:
    success, liveframe=videoframe.read()
    ClassIndex, confidece, bbox = model.detect(liveframe, confThreshold=0.55)
    print(ClassIndex)
    if(len(ClassIndex!=0)):
        for ClassInd,conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(liveframe,boxes,(255,0,0), 2)
                cv2.putText(liveframe,classnames[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font0scale, color=(0,255,0), thickness=3)
    cv2.imshow('livefeed',liveframe)

    if(cv2.waitKey(2) & 0xFF == ord('q')):
        break

videoframe.release()
cv2.destroyAllWindows()

