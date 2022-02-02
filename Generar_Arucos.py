## Generar Identificadores Aruco

import cv2
import os
import numpy as np
from PIL import Image

aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
Aruco_folder_path="Patrones_Aruco"
Aruco_dict_name="Dict_6x6_50" #    Poneren consonanica con la variable Aruco_dict


for i in range(5):
    tag = np.zeros((300, 300, 1), dtype="uint8") ## Array de salida
    cv2.aruco.drawMarker(aruco_dict, i, 300, tag, 1)
    aruco_name=Aruco_folder_path + "\ " + Aruco_dict_name + "Aruco_id_"+ str(i) + ".jpg"
    cv2.imwrite(aruco_name, tag)
