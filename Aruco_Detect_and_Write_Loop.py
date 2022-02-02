import cv2
import numpy as np
import os
from datetime import date, datetime

def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

#   Definir objeto de la camara
cam = cv2.VideoCapture('v4l2src device=/dev/video2 ! jpegdec ! videoconvert  ! video/x-raw, width=1920, height=1080 ! appsink drop=true sync=false',cv2.CAP_GSTREAMER)# Ubuntu
#cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)


#   BBDD path
imgDataStorePath='/home/ibai/Desktop/EHU_UPV/Base_Datos'
now=datetime.now()
new_folder_name = now.strftime("%Y_%m_%d_%H_%M_%S")
full_path=os.path.join(imgDataStorePath,new_folder_name)
os.mkdir(full_path)


#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Propiedades de texto en pantalla

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,50)
fontScale              =  0.7 
fontColor              = (255,255,255)
lineThickness          = 1

#   Loop Captura de Imagen via webcam
i=1
while True:
    try:
        #   Tomar Imagen
        cv_flag ,img = cam.read() 
        if not(cv_flag):  
            print('Error al capturar imagen')
            break
        #   Procesar Imagen
        #img=np.mean(img,-1)
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)

        
        #   Dibujar marcadores Aruco
        frame = cv2.aruco.drawDetectedMarkers(img, corners)

        #   Calcular distancia y colocar ID
        if corners!= []:
            #   Distancia
            dist=pixel_dist(corners[0][0][0],corners[0][0][1])
        
            #   Texto en pantalla
            image_text='ID: --> {}  Distancia: {:.2f}'.format(ids,dist)
            frame=cv2.putText(frame,image_text,
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineThickness)

        #  Mostrar Imagen procesada 

        cv2.imshow('Display', frame)
        if cv2.waitKey(1) & 0xFF == 32:
            cam.release()
            cv2.destroyAllWindows()
            break

        #   Grabar imagen en DIsco
        frame_name='Image_{}.jpg'.format(i)
        frame_path=os.path.join(full_path,frame_name)
	
        cv2.imwrite(frame_path, frame) 
        i=i+1
    except:
        print('Error en loop principal')
        cam.release()
        cv2.destroyAllWindows()        
        break

#   Finalizar programa
cam.release()
cv2.destroyAllWindows()