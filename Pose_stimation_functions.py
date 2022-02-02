import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math # Math library

def pose_stimation(frame, marker_ids, mtx, dst, rvecs, tvecs):
    axis_values = np.empty((0,3), int)
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

        axis_values=np.append(axis_values, np.array([[roll_x,pitch_y,yaw_z]]), axis=0)
        # print("transform_translation_x: {}".format(transform_translation_x))
        # print("transform_translation_y: {}".format(transform_translation_y))
        # print("transform_translation_z: {}".format(transform_translation_z))
        # print("roll_x: {}".format(roll_x))
        # print("pitch_y: {}".format(pitch_y))
        # print("yaw_z: {}".format(yaw_z))
        # print()
        
        # Draw the axes on the marker
        cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.025)

    return  frame, axis_values

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