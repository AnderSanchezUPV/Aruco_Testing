import cv2
import numpy as np
import os
import pickle
from datetime import date, datetime
import Pose_stimation_functions as Ps

#   Definicion de funciones
def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

#   Definir objeto de la camara
#cam = cv2.VideoCapture('v4l2src device=/dev/video2 ! jpegdec ! videoconvert  ! video/x-raw, width=1920, height=1080 ! appsink drop=true sync=false',cv2.CAP_GSTREAMER)# Ubuntu
cam = cv2.VideoCapture(0)

#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)  # Arucos impresos en mercedes
# aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)  # Arucos del Grid De Calibracion
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Propiedades de texto en pantalla

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,50)
fontScale              =  0.7 
fontColor              = (255,255,0)
lineThickness          = 1

#   Loop Detectar orientacion
aruco_marker_side_length=0.065

while True:
    try:
        #   Tomar Imagen
        cv_flag ,img = cam.read() 
        if not(cv_flag):  
            print('Error al capturar imagen')
            break
        #   Obtener Posicion del aruco e identificador
        #img=np.mean(img,-1)
        # img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)

        
        #   Dibujar marcadores de los Aruco
        frame = cv2.aruco.drawDetectedMarkers(img, corners)

        # # Calcular distancia y colocar ID
        # if corners!= []:
        # #      Distancia
        #    dist=pixel_dist(corners[0][0][0],corners[0][0][1])
        
        
        #   Pose Stimation
        #   Load calibration data
        with open('calibration.pckl', 'rb') as f:
            mtx, dist = pickle.load(f)
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                                    corners,
                                    aruco_marker_side_length,
                                    mtx,
                                    dist)

        #  Determinar orientacion Arucos  
        
        if ids is not None and len(ids) > 0: 
            # print('Echo')
            frame, axis_values =Ps.pose_stimation(frame,ids,mtx,dist,rvecs,tvecs)
            print(axis_values)

            #   Texto en pantalla
            image_text='ID: --> {}  Pitchy: {:.2f} Yawz: {:.2f}'.format(ids[0],
                                                                      axis_values[0][1],
                                                                      axis_values[0][2]) 
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
        #frame_name='Image_{}.jpg'.format(i)
        #frame_path=os.path.join(full_path,frame_name)
	
        #cv2.imwrite(frame_path, frame) 
        #i=i+1
    except:
        print('Error en loop principal')
        cam.release()
        cv2.destroyAllWindows()        
        break

#   Finalizar programa
cam.release()
cv2.destroyAllWindows()