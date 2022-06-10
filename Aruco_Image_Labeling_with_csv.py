import cv2
import numpy as np
import os
from datetime import date, datetime
import pickle
import Pose_stimation_functions as Ps

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
Path_Origen=r"C:\Users\ander\Documents\Imagenes CoMAr\2022_04_21_17_01_27"
Path_Destino=Path_Origen+"_Etiquetas"

if not os.path.exists(Path_Destino): os.mkdir(Path_Destino)

#   Generar archivo csv

e=datetime.now()
csv_file_name=e.strftime("%Y_%m_%d_%H_%M_%S")+'.txt'

f= open(os.path.join(Path_Destino,csv_file_name),"w+")

if not os.path.exists(Path_Destino): os.mkdir(Path_Destino)

kk=0
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


    
    #   Generar Etiquetas
    name, ext =os.path.splitext(filename)
    label_name=name
    

    #   Para guardar el primer aruco detectado
    pos_x=tvecs[0][0][0]
    pos_y=tvecs[0][0][1]
    pos_z=tvecs[0][0][2]

    px_x=corners[0][0][0][0]
    px_y=corners[0][0][0][1]

    roll_x=axis_values[0][0] # Eje verde
    pitch_y=axis_values[0][1]
    yaw_z=axis_values[0][2]

    id_ar=ids[0][0]
    f.write("{:.3f},{:.3f},{:.3f},{},{},{:.2f},{:.2f},{:.2f},{},{}".format( 
            float(pos_x),float(pos_y),float(pos_z),
            int(px_x),int(px_y),roll_x,pitch_y,yaw_z,
            id_ar,label_name))
      

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