import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from importlib.util import spec_from_file_location, module_from_spec

from scipy.linalg import logm

import os

from Path_Following_Package import *

np.set_printoptions(precision=4,suppress=True)

# Path generation
N = 5000

r = 0.5
h = 0.3

Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(2 * np.pi  / N * i), h * np.sin(2 * np.pi / N * i) ] for i in range(N)]

Rotation_List = []
ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3) 

# while (np.abs(ry) < 1.5 and np.abs(rp) < 1.5 and np.abs(rr) < 1.5):
#     ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3) 

for i in range (N):
    yaw   =   2 * np.pi * i / N + np.pi # ry * np.sin(2*np.pi * i / N) # 2 * np.pi * i / N # ry*np.sin(2*np.pi * i / N)
    pitch = 0 # rp # rp*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  + np.pi/2 # rp*np.sin(2*np.pi * i / N)
    roll  = 0 #  rr # rr*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  # 2 * np.pi * i / N  + np.pi/3 # rr*np.sin(2*np.pi * i / N)

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
    cr, sr = np.cos(roll),  np.sin(roll)

    Rotation_List.append(np.array([
        [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
        [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
        [-sp_sin,             cp*sr,             cp*cr       ]
    ]))

Frame = Frame_Path(Point_List, Rotation_List, True, 4, False)

Frame.Visualize_path(L= 0.1, num_of_roation_plots=20, Show_plots=False)

Frame.Visualize_frame(L = 0.1, num_of_frame_plots=20, Show_plots=False)

query_pts = []

for i in range (20):
    query_pts.append([np.cos(2 * np.pi * i / 20), np.sin(2 * np.pi * i / 20), 0.25])

Frame.Visualize_closest_point(query_pts)


Number_of_joint = 7
ALPHA = np.array([0.0, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2])
A = np.zeros(7)
D = np.array([0.0, 0.0, 0.42, 0.0, 0.4, 0.0, 0])
OFFSET = np.zeros(7)

# Inertial data
MASS = np.array([3.94781, 4.50275, 2.45520, 2.61155, 3.41000, 3.38795, 0.35432])
COM = np.array([
    [-0.00351,  0.00160, -0.03139],
    [-0.00767,  0.16669, -0.00355],
    [-0.00225, -0.03492, -0.02652],
    [ 0.00020, -0.05268,  0.03818],
    [ 0.00005, -0.00237, -0.21134],
    [ 0.00049,  0.02019, -0.02750],
    [-0.03466, -0.02324,  0.07138]
])
INERTIA = np.array([
    [0.00455, 0.00454, 0.00029,  0.00000,  0.00000, -0.00000],
    [0.00032, 0.00010, 0.00042,  0.00000,  0.00000,  0.00000],
    [0.00223, 0.00219, 0.00073, -0.00005,  0.00007,  0.00007],
    [0.03844, 0.01144, 0.04988,  0.00088, -0.00112, -0.00011],
    [0.00277, 0.00284, 0.00012, -0.00001,  0.00001,  0.00001],
    [0.00050, 0.00281, 0.00232, -0.00005, -0.00003, -0.00004],
    [0.00795, 0.01089, 0.00294, -0.00022, -0.00029, -0.00029]
])
Fv = np.array([0.24150, 0.37328, 0.11025, 0.10000, 0.10000, 0.12484, 0.10000])
Fs = np.array([0.31909, 0.18130, 0.07302, 0.17671, 0.03463, 0.13391, 0.08710])
Gravity = np.array([0, 0, -9.81])

KukaIiwa14 = Robot(Number_of_joint, ALPHA, A, D, OFFSET, MASS, COM, INERTIA, Fv, Fs, Gravity)

q_test = np.zeros(7)
q_test = np.array([2, 1, 0.2, -0.8, -0.2, 0.9, 0.5])
q_d_test = np.ones(7)
Fk = KukaIiwa14.forward_kinematics_all(q_test)

for index, item in enumerate(Fk):
    print(f"Joint {index} : \n {item}\n")

print(f"J:\n{KukaIiwa14.jacobian(q_test)}")
print(f"J_dot:\n{KukaIiwa14.jacobian_dot(q_test, q_d_test)}")