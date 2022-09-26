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
Path_Origen=r"D:\CoMAr Data\BBDD\Bag Images\Slow_Ligth"
New_Path=Path_Origen+"_Filtradas"

#   Crear carpeta de destino si no existe
if not os.path.exists(New_Path): os.mkdir(New_Path)

images = []

for filename in os.listdir(Path_Origen):
    img = cv2.imread(os.path.join(Path_Origen,filename))
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)
    mainImgName, file_extension = os.path.splitext(filename)                                   
    if img is not None and corners!=[]:
        #for i in range(ids.shape[0]):
        #images.append(img)

            #Id_tag="_ID_{}".format(i)
            #imgName=mainImgName+Id_tag+file_extension
        cv2.imwrite(os.path.join(New_Path,mainImgName+file_extension),img)
    

print('Done!')