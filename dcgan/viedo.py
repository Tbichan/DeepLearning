import numpy as np
import cv2
import os
import re

fileList = os.listdir("res/")

fileList2 = []

for file in (fileList):
    m = re.search("[0-9]+", file)
    tuple = (m.group(), file)
    fileList2.append(tuple)

fileList2.sort(key = lambda x: int(x[0]))

# 動画作成
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('video.mp4', fourcc, 10.0, (768, 768))

i = 0
for file in (fileList2):
    
    img = cv2.imread('res/' + file[1])
    video.write(img)
    print((int(i / len(fileList2) * 100)))
    i+=1

video.release()
print("Video Created!!")
