import cv2
import numpy as np
import os
from datetime import date, datetime
import pickle
import Pose_stimation_functions as Ps
from tkinter import Tk, filedialog

def pixel_dist(point1,point2):
    dist=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return dist

with open('calibration_basler.pckl', 'rb') as f:
            mtx, dist = pickle.load(f)

#   Definir propiedades de los Aruco
aruco_dict=cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParameters = cv2.aruco.DetectorParameters_create()
aruco_marker_side_length=0.065 # Tamaño del lado del Aruco en m

#   Definir carpeta de origen y Crear carpeta en la que colocar las nuevas imagenes
Path_Origen = filedialog.askdirectory(initialdir = r'D:\CoMAr Data\BBDD\Bag Images') # Returns opened path as str
#Path_Origen=r"C:\Users\ander\Documents\Imagenes CoMAr\2022_05_12_12_14_21"
Path_Destino=Path_Origen+"_Etiquetas"

if not os.path.exists(Path_Destino): os.mkdir(Path_Destino)

kk=0
for filename in os.listdir(Path_Origen):
    img = cv2.imread(os.path.join(Path_Origen,filename))
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict, 
                                        parameters=arucoParameters)

    #   Generar Etiquetas
    name, ext =os.path.splitext(filename)
    label_name=name+'.txt'
    f= open(os.path.join(Path_Destino,label_name),"a+")

    for i in range(ids.shape[0]):
        #if ids.shape[0]>1: 
            #print('!!')
        _centerY = int((corners[i][0][0][1] + corners[i][0][2][1]) / 2)
        _centerX = int((corners[i][0][0][0] + corners[i][0][2][0]) / 2)
        _cornerX= corners[i][0][0][0]                                 
        _cornerY= corners[i][0][0][1]
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                                        corners,
                                        aruco_marker_side_length,
                                        mtx,
                                        dist)

        frame, axis_values =Ps.pose_stimation(img,ids,mtx,dist,rvecs,tvecs)

    


        #   Para guardar el primer aruco detectado
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
        f.write("{},{},{},{:.2f},{:.2f},{:.2f},{},{}\n".format( 
                id_ar,
                int(px_x),int(px_y),
                roll_x,pitch_y,yaw_z,
                int(px_Cx),int(px_Cy)                
                ))
    f.close()

    #   Para guardar todos los Arucos registrados en la imagen 
    # for i in range(ids.shape[0]):
    #     pos_x=tvecs[i][0][0]
    #     pos_y=tvecs[i][0][1]
    #     px_x=corners[i][0][0][0]
    #     px_y=corners[i][0][0][1]
    #     theta=axis_values[i][1] # Eje verde
    #     id_ar=ids[i][0]
    #     f.write("{:.3f},{:.3f},{},{},{:.2f},{}\n".format(float(pos_x),float(pos_y),int(px_x),int(px_y),theta,id_ar))
    #print(kk)
    #kk=kk+1
    #if kk == 186 :
    #    print('he')
    # Depuracion del calculo de ejes
    #print(100*tvecs)
    #print((px_x/4,px_y/4))
    
    #print('#################\n')
    #frame=cv2.resize(frame,[int(frame.shape[1]/4),int(frame.shape[0]/4)])
    #cv2.imshow('Display', frame)
    #cv2.waitKey(0)   
    
cv2.destroyAllWindows()   
print('Done!')