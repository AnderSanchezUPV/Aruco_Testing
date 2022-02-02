import cv2
import numpy as np
import os
from datetime import date, datetime

def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()

#   Definir carpeta de origen y Crear carpeta en la que colocar las nuevas imagenes
Path_Origen=r"C:\Users\ander\Documents\Imagenes CoMAr\2022_01_20_12_40_16_Filtradas"
Path_Destino=Path_Origen+"_Etiquetas"

if not os.path.exists(Path_Destino): os.mkdir(Path_Destino)


for filename in os.listdir(Path_Origen):
    img = cv2.imread(os.path.join(Path_Origen,filename))
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)
    name, ext =os.path.splitext(filename)
    label_name=name+'.txt'
    f= open(os.path.join(Path_Destino,label_name),"w+")
    for i in range(ids.shape[0]):
        x=corners[i][0][0][0]
        y=corners[i][0][0][1]
        theta=-1
        id_ar=ids[i][0]
        f.write("{},{},{},{}\n".format(int(x),int(y),theta,id_ar))
        
        
    

print('Done!')