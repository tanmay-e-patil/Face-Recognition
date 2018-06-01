import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
import os

dir_path = "C:/Users/Tanmay Patil/Downloads/Face Recognition/images/"

detector = MTCNN()

name = input("Enter the name of the person ")

nrof_images = int (input("Enter the number of images to be taken "))


dir_path = dir_path + name
print(dir_path)

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def take_picture():
    camera_port = 0
    ramp_frames = 15
    camera = cv2.VideoCapture(camera_port)
    for i in range(ramp_frames):
        is_captured,image = camera.read()
    is_captured,image = camera.read()
    del(camera)
    return image

progress_bar = tqdm(total=nrof_images)
i = 0
while i < nrof_images:
    image = take_picture()
    detected_faces = detector.detect_faces(image)
    if (len(detected_faces) == 1):
        cv2.imwrite(dir_path + "/" + str(i) + ".png",image)
        i += 1
        progress_bar.update(1)
    elif (len(detected_faces) > 1):
        print("Only one face allowed")
    else:
        print("Align the camera correctly")
    






