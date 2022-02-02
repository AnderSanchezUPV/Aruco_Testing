import cv2
import numpy as np
import os
from datetime import date, datetime
from PIL import Image

def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

#   BBDD path
imgDataStorePath='/home/ibai/Desktop/EHU_UPV/Base_Datos'
now=datetime.now()
new_folder_name = now.strftime("%Y_%m_%d_%H_%M_%S")
full_path=os.path.join(imgDataStorePath,new_folder_name)
os.mkdir(full_path)

#   Cargar Imagen

img=Image.open('images/Aruco/prueba.bmp')
img=np.array(img)
#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Procesar Imagen
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                  parameters=arucoParameters)

print(ids)
#   Dibujar marcadores Aruco
frame = cv2.aruco.drawDetectedMarkers(img, corners)

i=1
frame_name='Image_{}.jpg'.format(i)
frame_path=os.path.join(full_path,frame_name)
	
cv2.imwrite(frame_path, frame) 


cv2.imshow('Display', frame)
cv2.waitKey(0)

#   Finalizar programa
cv2.destroyAllWindows()