import cv2
import numpy as np
import os
import pickle
from datetime import date, datetime
# import Pose_stimation_functions as Ps
from scipy.spatial.transform import Rotation as R
import math # Math library

#   Definicion de funciones

def pose_stimation(frame, marker_ids, mtx, dst, rvecs, tvecs):

    for i, marker_id in enumerate(marker_ids):
       
        # Store the translation (i.e. position) information
        transform_translation_x = tvecs[i][0][0]
        transform_translation_y = tvecs[i][0][1]
        transform_translation_z = tvecs[i][0][2]
 
        # Store the rotation information
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quat = r.as_quat()   
         
        # Quaternion format     
        transform_rotation_x = quat[0] 
        transform_rotation_y = quat[1] 
        transform_rotation_z = quat[2] 
        transform_rotation_w = quat[3] 
         
        # Euler angle format in radians
        roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                       transform_rotation_y, 
                                                       transform_rotation_z, 
                                                       transform_rotation_w)
         
        roll_x = math.degrees(roll_x)
        pitch_y = math.degrees(pitch_y)
        yaw_z = math.degrees(yaw_z)
        print("transform_translation_x: {}".format(transform_translation_x))
        print("transform_translation_y: {}".format(transform_translation_y))
        print("transform_translation_z: {}".format(transform_translation_z))
        print("roll_x: {}".format(roll_x))
        print("pitch_y: {}".format(pitch_y))
        print("yaw_z: {}".format(yaw_z))
        print()
         
        # Draw the axes on the marker
        cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)

    return  frame

def euler_from_quaternion(x, y, z, w):
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z # in radians


def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

#   Definir propiedades de los Aruco
# aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)  # Arucos impresos en mercedes
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)  # Arucos del Grid De Calibracion
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Cargar imagen
img = cv2.imread('Aruco_Grid.jpg')

#   Load calibration data
with open('calibration.pckl', 'rb') as f:
    mtx, dist = pickle.load(f)
    

#   Loop Detectar orientacion

aruco_marker_side_length=0.0255


try:
        
    # img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # Imagene a escala de grises

    # Detectar posicion de los arucos
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)

        
    #   Dibujar marcadores de los Aruco
    frame = cv2.aruco.drawDetectedMarkers(img, corners)

        # # Calcular distancia y colocar ID
        # if corners!= []:
        # #      Distancia
        #    dist=pixel_dist(corners[0][0][0],corners[0][0][1])
        
        # #      Texto en pantalla
        #    image_text='ID: --> {}  Distancia: {:.2f}'.format(ids,dist)
        #    frame=cv2.putText(frame,image_text,
        #                          bottomLeftCornerOfText, 
        #                        font, 
        #                        fontScale,
        #                        fontColor,
        #                        lineThickness)

        # Calcular vectores
    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                                    corners,
                                    aruco_marker_side_length,
                                    mtx,
                                    dist)    

        #  Determinar orientacion Arucos  
        
    if len(ids) > 0 and ids is not None : 
        frame=pose_stimation(frame,ids,mtx,dist,rvecs,tvecs)
        
        #  Mostrar Imagen procesada         

    cv2.imshow('Display', frame)
    if  cv2.waitKey(0) & 0xFF == 32:
        cv2.destroyAllWindows()            

      
except:
    print('Error De proceso')    
    cv2.destroyAllWindows()        
    

#   Finalizar programa
cv2.destroyAllWindows()