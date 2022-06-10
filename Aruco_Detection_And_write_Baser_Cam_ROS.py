'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import numpy as np
import cv2
import pickle
import Pose_stimation_functions as Ps
import os
from datetime import date, datetime
import time
import sys

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# #   Defincion de funciones
# def pixel_dist(point1,point2):
#     dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
#     return dist
#   Definir Path destino
#   BBDD path
imgDataStorePath=r'/home/comar/UPV-EHU/BBDD'
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
fontColor              = (255,255,0)
lineThickness          = 2
cv2.namedWindow('title', cv2.WINDOW_FULLSCREEN )

#   Load calibration data

with open('calibration_basler.pckl', 'rb') as f:
    mtx, dist = pickle.load(f)
aruco_marker_side_length=0.065

                        ####        Conexion camera GIGE        ####
# # conecting to the first available camera
# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# # Grabing Continusely (video) with minimal delay
# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
# converter = pylon.ImageFormatConverter()

# # converting to opencv bgr format
# converter.OutputPixelFormat = pylon.PixelType_BGR8packed
# converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
i=0
cv_bridge = CvBridge()

def capturar_imagen(ros_img : Image):
    global dist
    global i

    i=i+1
    id= ros_img.header.seq
    img = cv_bridge.imgmsg_to_cv2(ros_img)

    
    # Access the image data
    org_img=np.copy(img)    

    # Buscar Arucos
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                parameters=arucoParameters)  

    # Pose Stimation
    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                            corners,
                            aruco_marker_side_length,
                            mtx,
                            dist)
    
    # Dibujar marcadores Aruco y ejes
    frame = cv2.aruco.drawDetectedMarkers(img, corners, ids)
                        
    if ids is not None and len(ids) > 0:                            
        frame, axis_values =Ps.pose_stimation(frame,ids,mtx,dist,rvecs,tvecs)
        # rospy.loginfo('Aruco Detectado')
        
    
        #   Grabar imagen en DIsco            
        frame_name='Image_{}.jpg'.format(str(id).zfill(15))
        frame_path=os.path.join(full_path,frame_name)
    
        if (i % 10) == 0:
            cv2.imwrite(frame_path, org_img)
            rospy.loginfo(frame_name) 
        
    
                        ####            Main Loop           ####
rospy.init_node("pylon_camera", argv=sys.argv)

image_sub=rospy.Subscriber('/comar/back_cam/img', Image, capturar_imagen, queue_size=5)

while not rospy.is_shutdown():
    rospy.spin()





    
# Releasing the resource    
# camera.StopGrabbing()
# cv2.destroyAllWindows()