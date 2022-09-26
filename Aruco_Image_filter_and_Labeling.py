import cv2
import numpy as np
import os
from datetime import date, datetime
from tkinter import Tk, filedialog
import pickle
import Pose_stimation_functions as Ps

def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()
aruco_marker_side_length=0.065 # Tamaño del lado del Aruco en m

with open('calibration_basler.pckl', 'rb') as f:
            mtx, dist = pickle.load(f)


#   Definir carpeta de origen y Crear carpeta en la que colocar las nuevas imagenes
Path_Origen = filedialog.askdirectory(initialdir = r'D:\CoMAr Data\BBDD\Bag Images') # Returns opened path as str
Path_Destino_Img_Filtradas=Path_Origen+"_Filtradas_Ext"
Path_Destino_Etiquetas=Path_Origen+"_Etiquetas"

#   Crear carpeta de destino si no existe
if not os.path.exists(Path_Destino_Etiquetas): os.mkdir(Path_Destino_Etiquetas)
if not os.path.exists(Path_Destino_Img_Filtradas): os.mkdir(Path_Destino_Img_Filtradas)

images = []

for filename in os.listdir(Path_Origen):
    img = cv2.imread(os.path.join(Path_Origen,filename))
    img_copy=np.copy(img)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)
    mainImgName, file_extension = os.path.splitext(filename)                                   
    if img is not None and corners!=[]:
        for i in range(ids.shape[0]):
        #images.append(img)

            #   extract Aruco pose
            _centerY = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
            _centerX = int((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
            _cornerX= corners[i][0][0][0]                                 
            _cornerY= corners[i][0][0][1]
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                                            corners,
                                            aruco_marker_side_length,
                                            mtx,
                                            dist)

            _, axis_values =Ps.pose_stimation(img,ids,mtx,dist,rvecs,tvecs)
            pos_x=tvecs[i][0][0]
            pos_y=tvecs[i][0][1]
            pos_z=tvecs[i][0][2]

            px_x=_centerX   #Pos del cel centro del Aruco
            px_y=_centerY   #Pos del cel centro del Aruco

            px_Cx=_cornerX          #Pos de la esquina caracteristica del Aruco
            px_Cy=_cornerY          #Pos de la esquina caracteristica del Aruco

            roll_x=axis_values[i][0] # Eje verde
            pitch_y=axis_values[i][1]
            yaw_z=axis_values[i][2]

            id_ar=ids[i][0]            


            #   Save labeled Image
            Id_tag="_ID_{}".format(id_ar)
            imgName=mainImgName+Id_tag+file_extension
            cv2.imwrite(os.path.join(Path_Destino_Img_Filtradas,imgName),img_copy)

            #   Save Label Data
            label_name=mainImgName+Id_tag+'.txt'
            f= open(os.path.join(Path_Destino_Etiquetas,label_name),"a+")

            f.write("{},{},{},{:.2f},{:.2f},{:.2f},{},{}\n".format( 
                id_ar,
                int(px_x),int(px_y),
                roll_x,pitch_y,yaw_z,
                int(px_Cx),int(px_Cy)                
                ))
            f.close()



print('Done!')