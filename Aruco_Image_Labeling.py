import cv2
import numpy as np
import os
from datetime import date, datetime
import pickle
import Pose_stimation_functions as Ps

def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

with open('calibration.pckl', 'rb') as f:
            mtx, dist = pickle.load(f)

#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()
aruco_marker_side_length=0.0255 # Tamaño del lado del Aruco en m

#   Definir carpeta de origen y Crear carpeta en la que colocar las nuevas imagenes
Path_Origen=r"C:\Users\ander\Documents\Imagenes CoMAr\2022_01_20_12_40_16_Filtradas"
Path_Destino=Path_Origen+"_Etiquetas"

if not os.path.exists(Path_Destino): os.mkdir(Path_Destino)


for filename in os.listdir(Path_Origen):
    img = cv2.imread(os.path.join(Path_Origen,filename))
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)

    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                                    corners,
                                    aruco_marker_side_length,
                                    mtx,
                                    dist)

    frame, axis_values =Ps.pose_stimation(img,ids,mtx,dist,rvecs,tvecs)

    # Depuracion del calculo de ejes
    # frame=cv2.resize(frame,[int(frame.shape[1]/2),int(frame.shape[0]/2)])
    # cv2.imshow('Display', frame)
    # cv2.waitKey(0)

    #   Generar Etiquetas
    name, ext =os.path.splitext(filename)
    label_name=name+'.txt'
    f= open(os.path.join(Path_Destino,label_name),"w+")
    for i in range(ids.shape[0]):
        x=corners[i][0][0][0]
        y=corners[i][0][0][1]
        theta=axis_values[i][1] # Eje verde
        id_ar=ids[i][0]
        f.write("{},{},{:.2f},{}\n".format(int(x),int(y),theta,id_ar))
        
        
cv2.destroyAllWindows()   
print('Done!')