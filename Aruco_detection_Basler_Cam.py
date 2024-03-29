'''
A simple Program for grabing video from basler camera and converting it to opencv img.
Tested on Basler acA1300-200uc (USB3, linux 64bit , python 3.5)

'''
from pypylon import pylon
import numpy as np
import cv2
import pickle
import Pose_stimation_functions as Ps

#   Defincion de funciones
def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist
#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Propiedades de texto en pantalla

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,50)
fontScale              =  1
fontColor              = (255,255,0)
lineThickness          = 2  
cv2.namedWindow('title', cv2.WINDOW_FULLSCREEN )

#   Load calibration data
with open('calibration_basler.pckl', 'rb') as f:
    mtx, dist = pickle.load(f)
aruco_marker_side_length=0.065

                        ####        Conexion camera GIGE        ####
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


                        ####            Main Loop           ####

while camera.IsGrabbing():
    try:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()

            # Reducir para depuracion Visual
            # img=cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4))) 

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
            #   Calcular distancia 
                #   Distancia
                dist=pixel_dist(corners[0][0][0],corners[0][0][1])
            
                #   Texto en pantalla
                #image_text='Definicion: {:.2f} mm/px  Theta:{:.2f} Deg'.format(
                image_text='Definicion: {:.2f} mm/px  ID: {}'.format(
                            aruco_marker_side_length*1000/dist,
                            ids[0])
                frame=cv2.putText(frame,image_text,
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineThickness)

            # Reducir imagen para mostrar en depuracion
            # frame=cv2.resize(frame,(int(frame.shape[1]/4),int(frame.shape[0]/4))) 
            cv2.imshow('title', frame)
            if cv2.waitKey(1) & 0xFF == 32:
                camera.StopGrabbing()
                cv2.destroyAllWindows()
                break
        grabResult.Release()

    except:
        print('Error en loop principal')
        camera.StopGrabbing()
        cv2.destroyAllWindows()        
        break


    
# Releasing the resource    
camera.StopGrabbing()
cv2.destroyAllWindows()