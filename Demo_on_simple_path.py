import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from importlib.util import spec_from_file_location, module_from_spec


from scipy.linalg import logm

import os


from Path_Following_Package import *
from scipy.interpolate import make_interp_spline

SAVE_DIR = "Demo_Image"
PREFIX_NAME = "Simple_path_2_false_"
os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + ".png"), dpi=300)

# Path generation
N = 5000
PATH_INDEX = 6

# Circular path
if (PATH_INDEX == 1):
    r = 0.5
    h = 0.3

    Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(2 * np.pi  / N * i), h * np.cos(2 * np.pi / N * i ) ] for i in range(N)]
    # Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(2 * np.pi  / N * i), h] for i in range(N)]

    Rotation_List = []
    ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3) 

    # while (np.abs(ry) < 1.5 and np.abs(rp) < 1.5 and np.abs(rr) < 1.5):
    #     ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3) 

    for i in range (N):
        yaw   =  2 * np.pi * i / N + np.pi/ 3 # ry * np.sin(2*np.pi * i / N) # 2 * np.pi * i / N # ry*np.sin(2*np.pi * i / N)
        pitch = rp # rp*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  + np.pi/2 # rp*np.sin(2*np.pi * i / N)
        roll  =  rr # rr*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  # 2 * np.pi * i / N  + np.pi/3 # rr*np.sin(2*np.pi * i / N)

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
        cr, sr = np.cos(roll),  np.sin(roll)

        Rotation_List.append(np.array([
            [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
            [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
            [-sp_sin,             cp*sr,             cp*cr       ]
        ]))

    Frame = Frame_Path(Point_List, Rotation_List, True, 4)

# Linear path
elif (PATH_INDEX == 2):
    Point_List = [[0.1 + 0.3 * i / N, 0.1 + 0.3 * i / N, 0.25 + 0.1 * (i / N)] for i in range(N)]
    # Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(2 * np.pi  / N * i), h] for i in range(N)]

    Rotation_List = []
    ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3)

    # while (np.abs(ry) < 1.5 and np.abs(rp) < 1.5 and np.abs(rr) < 1.5):
    #     ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3)

    for i in range (N):
        yaw   =  0 # ry # ry*np.sin(2*np.pi * i / N)# ry * np.sin(2*np.pi * i / N) # 2 * np.pi * i / N # ry*np.sin(2*np.pi * i / N)
        pitch = 0 # rp # rp*np.sin(2*np.pi * i / N) # rp*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  + np.pi/2 # rp*np.sin(2*np.pi * i / N)
        roll  =  0 # rr # rr*np.sin(2*np.pi * i / N) # rr*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  # 2 * np.pi * i / N  + np.pi/3 # rr*np.sin(2*np.pi * i / N)

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
        cr, sr = np.cos(roll),  np.sin(roll)

        Rotation_List.append(np.array([
            [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
            [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
            [-sp_sin,             cp*sr,             cp*cr       ]
        ]))

    Dense_points = np.array(Point_List)
    Dense_rotations = np.array(Rotation_List)
    stride = max(1, N // 30)
    Point_List = Dense_points[::stride]
    Rotation_List = Dense_rotations[::stride]

    # if not np.allclose(Point_List[-1], Dense_points[-1]):
    #     Point_List = np.vstack([Point_List, Dense_points[-1]])
    #     Rotation_List = np.concatenate([Rotation_List, Dense_rotations[-1:]], axis=0)

    Frame = Frame_Path(Point_List, Rotation_List, False, 4)

# Circular path non looping
elif (PATH_INDEX == 3):
    N = 30

    # Point_List = [[0.5 * np.cos(2 * np.pi / N * i), 0.5 * np.sin(2 * np.pi  / N * i), 0.3 * np.cos(2 * np.pi * 2/ N * i + np.pi/3)] for i in range(N)]
    Point_List = [[0.5 * np.cos(2 * np.pi / N * i), 0.5 * np.sin(2 * np.pi  / N * i), 0.3] for i in range(N)]
    # Point_List = [[0.25 * np.cos(2 * np.pi / N * i) + 0.3 , 0.25 * np.sin(2 * np.pi  / N * i) + 0.3, 0.3 ] for i in range(N)]
    # Point_List = [[0.1 + 0.3 * i / N, 0.1 + 0.3 * i / N, 0.25 + 0.1 * (i / N)] for i in range(N)]


    # rx, ry, rz = np.random.rand(3) * np.pi
    rx, ry, rz = np.ones(3) * np.pi
    # Rotation_List = [R.from_euler('xyz', [rx * np.sin(2 * np.pi * i/N), ry * np.cos(2 * np.pi * i/N), rz * np.sin(2 * np.pi * i/N)]).as_matrix().reshape((3,3)) for i in range(N)]

    # Rotation_List = [R.from_euler('xyz', [0, 0, rz * np.sin(2 * np.pi * i/N)]).as_matrix().reshape((3,3)) for i in range(N)]

    Rotation_List = []
    ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3)

    # while (np.abs(ry) < 1.5 and np.abs(rp) < 1.5 and np.abs(rr) < 1.5):
    #     ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3)

    # ry, rp, rr = np.random.uniform(-np.pi / 2, np.pi / 2, 3) # np.random.rand(3)

    # while (np.abs(ry) < 1 and np.abs(rp) < 1 and np.abs(rr) < 1):
    #     ry, rp, rr = np.random.uniform(-np.pi / 2, np.pi / 2, 3)

    # ry, rp, rr = np.random.uniform(-1, 1, 3) # np.random.rand(3)

    print("1", ry, rp, rr)

    yaw_c, pitch_c, roll_c = np.random.uniform(-1, 1, 3)
    # yaw_c, pitch_c, roll_c = np.random.uniform(-np.pi/2, np.pi/2, 3)
    for i in range (N):
        yaw   =  0 #  ry * np.sin(2*np.pi * i / N) #  2 * np.pi * i / N + np.pi #  ry * np.sin(2*np.pi * i / N) # 2 * np.pi * i / N # ry*np.sin(2*np.pi * i / N)
        pitch = 0 # rp # rp*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  + np.pi/2 # rp*np.sin(2*np.pi * i / N)
        roll  =  0 # rr #  rr*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  # 2 * np.pi * i / N  + np.pi/3 # rr*np.sin(2*np.pi * i / N)

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
        cr, sr = np.cos(roll),  np.sin(roll)

        Rotation_List.append(np.array([
            [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
            [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
            [-sp_sin,             cp*sr,             cp*cr       ]
        ]))

    Frame = Frame_Path(Point_List, Rotation_List, False, 4)

elif (PATH_INDEX == 4):
    
    STRAIGHT_LINE_PATH = {
        "points": [
            [
                0.1,
                0.1,
                0.25
            ],
            [
                0.11034482758620691,
                0.11034482758620691,
                0.253448275862069
            ],
            [
                0.1206896551724138,
                0.1206896551724138,
                0.25689655172413794
            ],
            [
                0.1310344827586207,
                0.1310344827586207,
                0.2603448275862069
            ],
            [
                0.1413793103448276,
                0.1413793103448276,
                0.26379310344827583
            ],
            [
                0.1517241379310345,
                0.1517241379310345,
                0.2672413793103448
            ],
            [
                0.1620689655172414,
                0.1620689655172414,
                0.2706896551724138
            ],
            [
                0.1724137931034483,
                0.1724137931034483,
                0.27413793103448275
            ],
            [
                0.1827586206896552,
                0.1827586206896552,
                0.2775862068965517
            ],
            [
                0.1931034482758621,
                0.1931034482758621,
                0.2810344827586207
            ],
            [
                0.20344827586206898,
                0.20344827586206898,
                0.28448275862068967
            ],
            [
                0.2137931034482759,
                0.2137931034482759,
                0.2879310344827586
            ],
            [
                0.2241379310344828,
                0.2241379310344828,
                0.29137931034482756
            ],
            [
                0.23448275862068968,
                0.23448275862068968,
                0.29482758620689653
            ],
            [
                0.24482758620689657,
                0.24482758620689657,
                0.2982758620689655
            ],
            [
                0.2551724137931035,
                0.2551724137931035,
                0.3017241379310345
            ],
            [
                0.2655172413793104,
                0.2655172413793104,
                0.30517241379310345
            ],
            [
                0.27586206896551724,
                0.27586206896551724,
                0.3086206896551724
            ],
            [
                0.28620689655172415,
                0.28620689655172415,
                0.3120689655172414
            ],
            [
                0.29655172413793107,
                0.29655172413793107,
                0.3155172413793103
            ],
            [
                0.306896551724138,
                0.306896551724138,
                0.3189655172413793
            ],
            [
                0.3172413793103449,
                0.3172413793103449,
                0.32241379310344825
            ],
            [
                0.32758620689655177,
                0.32758620689655177,
                0.3258620689655172
            ],
            [
                0.33793103448275863,
                0.33793103448275863,
                0.3293103448275862
            ],
            [
                0.34827586206896555,
                0.34827586206896555,
                0.3327586206896551
            ],
            [
                0.35862068965517246,
                0.35862068965517246,
                0.3362068965517241
            ],
            [
                0.3689655172413794,
                0.3689655172413794,
                0.33965517241379306
            ],
            [
                0.3793103448275863,
                0.3793103448275863,
                0.34310344827586203
            ],
            [
                0.3896551724137931,
                0.3896551724137931,
                0.346551724137931
            ],
            [
                0.4,
                0.4,
                0.35
            ]
        ],
        "angles": [
            [
                0.0,
                0.0,
                0.0
            ],
            [
                0.0,
                0.0,
                12.0
            ],
            [
                0.0,
                0.0,
                24.0
            ],
            [
                0.0,
                0.0,
                36.0
            ],
            [
                0.0,
                0.0,
                48.0
            ],
            [
                0.0,
                0.0,
                60.0
            ],
            [
                0.0,
                0.0,
                72.0
            ],
            [
                0.0,
                0.0,
                84.0
            ],
            [
                0.0,
                0.0,
                96.0
            ],
            [
                0.0,
                0.0,
                108.0
            ],
            [
                0.0,
                0.0,
                120.0
            ],
            [
                0.0,
                0.0,
                132.0
            ],
            [
                0.0,
                0.0,
                144.0
            ],
            [
                0.0,
                0.0,
                156.0
            ],
            [
                0.0,
                0.0,
                168.0
            ],
            [
                0.0,
                0.0,
                180.0
            ],
            [
                0.0,
                0.0,
                192.0
            ],
            [
                0.0,
                0.0,
                204.0
            ],
            [
                0.0,
                0.0,
                216.0
            ],
            [
                0.0,
                0.0,
                228.0
            ],
            [
                0.0,
                0.0,
                240.0
            ],
            [
                0.0,
                0.0,
                252.0
            ],
            [
                0.0,
                0.0,
                264.0
            ],
            [
                0.0,
                0.0,
                276.0
            ],
            [
                0.0,
                0.0,
                288.0
            ],
            [
                0.0,
                0.0,
                300.0
            ],
            [
                0.0,
                0.0,
                312.0
            ],
            [
                0.0,
                0.0,
                324.0
            ],
            [
                0.0,
                0.0,
                336.0
            ],
            [
                0.0,
                0.0,
                348.0
            ]
        ],
        "Is_loop": False,
        "Rotation": [
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            [
                [
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        ]
    }

    Frame = Frame_Path(STRAIGHT_LINE_PATH["points"], STRAIGHT_LINE_PATH["Rotation"], STRAIGHT_LINE_PATH["Is_loop"], 4)  

elif (PATH_INDEX == 5):
    N = 31
    Point_List = [[0.1 + 0.3 * i / N, 0.1 + 0.3 * i / N, 0.25 + 0.1 * (i / N)] for i in range(N)]
    # Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(2 * np.pi  / N * i), h] for i in range(N)]

    Rotation_List = []
    ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3)

    # while (np.abs(ry) < 1.5 and np.abs(rp) < 1.5 and np.abs(rr) < 1.5):
    #     ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) # np.random.rand(3)

    for i in range (N):
        yaw   =  0 # ry # ry*np.sin(2*np.pi * i / N)# ry * np.sin(2*np.pi * i / N) # 2 * np.pi * i / N # ry*np.sin(2*np.pi * i / N)
        pitch = 0 # rp # rp*np.sin(2*np.pi * i / N) # rp*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  + np.pi/2 # rp*np.sin(2*np.pi * i / N)
        roll  =  0 # rr # rr*np.sin(2*np.pi * i / N) # rr*np.sin(2*np.pi * i / N) # 2 * np.pi * i / N  # 2 * np.pi * i / N  + np.pi/3 # rr*np.sin(2*np.pi * i / N)

        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
        cr, sr = np.cos(roll),  np.sin(roll)

        Rotation_List.append(np.array([
            [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
            [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
            [-sp_sin,             cp*sr,             cp*cr       ]
        ]))

    # Dense_points = np.array(Point_List)
    # Dense_rotations = np.array(Rotation_List)
    # stride = max(1, N // 30)
    # Point_List = Dense_points[::stride]
    # Rotation_List = Dense_rotations[::stride]

    # if not np.allclose(Point_List[-1], Dense_points[-1]):
    #     Point_List = np.vstack([Point_List, Dense_points[-1]])
    #     Rotation_List = np.concatenate([Rotation_List, Dense_rotations[-1:]], axis=0)

    Frame = Frame_Path(Point_List, Rotation_List, False, 4)

elif (PATH_INDEX == 6):
    rng = np.random.default_rng(12345)
    M = 60
    base_point = np.array([0.25, 0.15, 0.2])

    ctrl_count = 8
    ctrl_s = np.linspace(0.0, 1.0, ctrl_count)
    ctrl_offsets = rng.normal(scale=0.05, size=(ctrl_count, 3))
    ctrl_offsets[0] = 0.0
    ctrl_points = base_point + np.cumsum(ctrl_offsets, axis=0)

    s_samples = np.linspace(0.0, 1.0, M)
    Point_List = make_interp_spline(ctrl_s, ctrl_points, k=3)(s_samples)

    angle_ctrl = rng.normal(scale=0.4, size=(ctrl_count, 3))
    angle_ctrl[0] = 0.0
    smoothed_angles = make_interp_spline(ctrl_s, angle_ctrl, k=3)(s_samples)
    Rotation_List = R.from_euler("zyx", smoothed_angles).as_matrix()

    Frame = Frame_Path(Point_List, Rotation_List, False, 4)

# Tracking Signal
print("Total path length:", Frame.total_length)

def Reference_signal(Frame, time):
    # return 0.3, 0, 0
    if (Frame.Is_loop):
        # return 0.5 * np.sin(time * 2 * np.pi / 5) + 0.5, 0.5 * np.cos(time * 2 * np.pi / 5) * 2 * np.pi / 5, 0.5 * -np.sin(time * 2 * np.pi / 5)* 2 * np.pi /5 * 2 * np.pi / 5
    
        return (Frame.total_length / 5.0 * time) % (Frame.total_length), Frame.total_length / 5.0, 0
    
    elif (1):
        if (time < 5):
            Position = 0
            Velocity = 0
        
        elif (time < 15):
            Temp_velocity = Frame.total_length / 10
            Temp_position = Temp_velocity * (time - 5)

            Position = Temp_position
            Velocity = Temp_velocity

        elif (time < 20):
            Position = Frame.total_length
            Velocity = 0

        elif (time < 30):
            Temp_velocity = - Frame.total_length / 10
            Temp_position = Frame.total_length + Temp_velocity * (time - 20)

            Position = Temp_position
            Velocity = Temp_velocity

        else:
            Position = 0
            Velocity = 0

        return Position, Velocity, 0
    else:
        Temp_velocity = Frame.total_length / 10
        Temp_position = Temp_velocity * (time)

        if (Temp_position >= 0 and Temp_position < Frame.total_length):
            Position = Temp_position
            Velocity = Temp_velocity

        elif (Temp_position < 0):
            Position = 0
            Velocity = 0

        else:
            Position = Frame.total_length  
            Velocity = 0

        return Position, Velocity, 0

# Robot definition (Kuka iiwa 14)
NUMBER_OF_JOINT = 7
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
FV = np.array([0.24150, 0.37328, 0.11025, 0.10000, 0.10000, 0.12484, 0.10000])
FS = np.array([0.31909, 0.18130, 0.07302, 0.17671, 0.03463, 0.13391, 0.08710])
GRAVITY = np.array([0, 0, -9.81])
END_EFFECTOR_TRANSFORMATION=np.eye(4)

KukaIiwa14 = Robot(NUMBER_OF_JOINT, ALPHA, A, D, OFFSET, MASS, COM, INERTIA, FV, FS, GRAVITY, End_effector_transformation= END_EFFECTOR_TRANSFORMATION, convention="MDH", joint_types=["R"]*7)

KukaIiwa14_Control_Module =Virtal_Task_Space_Control_Module(Kp = 60, Kd = 50, K1 = 250, K2 = 200)

# Simulation parameters
dt = 0.005
T_total = 40

steps = int(T_total / dt)

# controller parameters
Kp = 500
Kd = 50 

# Log 
time_log = []
q_log = []
x_log = []
eta_1_log = []
xi_1_log = []
xi_3_log = []
reference_signal_log = []
reference_velocity_log = []
x_des_log = []
lambda_log = []
J_hat_determinent_log = []
E_determinent_log = []
beta_log = []

quat_log = []
quat_des_log = []
Rotation_error_log = []

euler_log = []
euler_des_log = []

er_log = []

trace_log = []

u_log = []
v_log = []

J_H_log = []

beta_log = []


# Initial condition
q = np.random.uniform(-np.pi, np.pi, size=7)
qd = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
qdd = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)


for step in tqdm(range(steps), desc="Simulating", unit="step"):
    time = step * dt
    Current_reference_signal = Reference_signal(Frame, time)
    reference_signal_pos = Current_reference_signal[0]
    reference_signal_velocity = Current_reference_signal[1]
    
    v, Virtal_Task_Space_variable = KukaIiwa14_Control_Module.Get_Control_Input(KukaIiwa14, Frame, q, qd, qdd, Current_reference_signal, True, 10)
    v_log.append(v.copy())

    v_7dof = np.zeros((7))    
    h_7dof = np.zeros((7))
    h_d_7dof = np.zeros((7))
    J_H_7dof = np.zeros((7,7))
    J_H_dot_7dof = np.zeros((7,7))

    v_7dof[:-1] = v 
    h_7dof[:-1] = Virtal_Task_Space_variable['h']
    h_d_7dof[:-1] = Virtal_Task_Space_variable['hd']
    J_H_7dof[:-1, :] = Virtal_Task_Space_variable['J_H']
    J_H_dot_7dof[:-1, :] = Virtal_Task_Space_variable['J_H_dot']

    v_7dof[6] = 50.0 * (0 - qd[2]) + 100.0 * (0 - q[2])
    h_7dof[6] = q[2]
    h_d_7dof[6] = qd[2]
    J_H_7dof[6, 2] = 1
    J_H_dot_7dof[6, :] = np.zeros(7)

    u = np.linalg.inv(J_H_7dof) @ (-J_H_dot_7dof @ qd + v_7dof)
    u = np.clip(u, -30, 30)

    
    M, C, Fv, Fc, G = KukaIiwa14.dynamics(q, qd)
    tau = M @ u + C @ qd + Fv @ qd + Fc @ np.sign(qd) + G

    qdd = KukaIiwa14.forward_dynamics(q, qd, tau)
    qd += qdd * dt
    q += qd * dt
    q = (q + np.pi) % (2 * np.pi) - np.pi

    
    eta_1 = Virtal_Task_Space_variable['eta_1']
    eta_2 = Virtal_Task_Space_variable['eta_2']
    xi_1  = Virtal_Task_Space_variable['xi_1']
    xi_2  = Virtal_Task_Space_variable['xi_2']
    xi_3  = Virtal_Task_Space_variable['xi_3']
    xi_4  = Virtal_Task_Space_variable['xi_4']
    s_star = Virtal_Task_Space_variable['s_star']
    E = Virtal_Task_Space_variable['E']
    e_rot = Virtal_Task_Space_variable['e_rot']
    R_err = Virtal_Task_Space_variable['R_err']


    y_star = Frame.P(s_star).reshape((3, 1))

    T_cur = KukaIiwa14.forward_kinematics(q)
    x_cur = T_cur[:3, 3]
    R_cur = T_cur[:3, :3]
    R_des  = Frame.R(reference_signal_pos).as_matrix().reshape((3, 3)) 


    time_log.append(time)
    q_log.append(q.copy())
    x_log.append(x_cur.copy())
    eta_1_log.append(float(eta_1))
    xi_1_log.append(float(xi_1))
    xi_3_log.append(float(xi_3))
    reference_signal_log.append(float(reference_signal_pos))
    reference_velocity_log.append(float(reference_signal_velocity))
    x_des_log.append(y_star.flatten())
    beta_log.append(float(Virtal_Task_Space_variable["beta"]))
    # lambda_log.append(lambda_star)

    J_hat_determinent_log.append(np.linalg.det(J_H_7dof))
    # J_determinent_log.append(np.linalg.matrix_rank(J))
    E_determinent_log.append(np.linalg.det(E))


    quat_log.append(R.from_matrix(R_cur).as_quat())
    quat_des_log.append(R.from_matrix(R_des).as_quat())

    euler_log.append(R.from_matrix(R_cur).as_euler('ZYX', degrees=False))
    euler_des_log.append(R.from_matrix(R_des).as_euler('ZYX', degrees=False)) 

    Rotation_error_log.append(e_rot)

    er_log.append(np.linalg.norm(e_rot))

    trace_log.append(np.trace(R_err))

    u_log.append(u)

    J_H_log.append(J_H_7dof.copy())


    if (np.trace(R_err) == -1):
        print('Trace -1')
        break

q_log = np.array(q_log)
x_log = np.array(x_log)
x_des_log   = np.array(x_des_log)
quat_log = np.array(quat_log)
quat_des_log = np.array(quat_des_log)
Rotation_error_log = np.array(Rotation_error_log)
u_log = np.array(u_log)
v_log = np.array(v_log)

J_H_log = np.array(J_H_log)
J_H_columns = [f"J_H_{row}_{col}" for row in range(J_H_log.shape[1]) for col in range(J_H_log.shape[2])]
J_H_df = pd.DataFrame(J_H_log.reshape(J_H_log.shape[0], -1), columns=J_H_columns)
J_H_df.insert(0, "time", time_log)
J_H_output_path = os.path.join(SAVE_DIR, f"{PREFIX_NAME}J_H_log.xlsx")
J_H_df.to_excel(J_H_output_path, index=False)

# plt.figure()
# plt.plot(time_log, lambda_log)
# plt.title("lambda")

plt.figure()
plt.plot(time_log, x_log[:, 0], color = 'red', label='x')
plt.plot(time_log, x_log[:, 1], color = 'green', label='y')
plt.plot(time_log, x_log[:, 2], color = 'blue', label='z')

plt.plot(time_log, x_des_log[:, 0], '--', color='red',   label='x (ref)')
plt.plot(time_log, x_des_log[:, 1], '--', color='green', label='y (ref)')
plt.plot(time_log, x_des_log[:, 2], '--', color='blue',  label='z (ref)')

plt.title("End-Effector Position Convergence")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}end_effector")


plt.figure()
for i in range(7):
    plt.plot(time_log, q_log[:, i], label=f'q{i+1}')
plt.title("Joint Angles Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Joint Angle [rad]")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}q")


fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection="3d")

ax.plot(
    x_log[:, 0],   # X
    x_log[:, 1],   # Y
    x_log[:, 2],   # Z
    linewidth=2,   # a little thicker so it’s easy to see
    label="Path"
)

s_sample = np.linspace(0, Frame.total_length, 400, endpoint = True)
p_sample = Frame.P(s_sample)    
ax.plot(*p_sample.T, lw=1.5, linestyle= "--", color = "black")

# theta = np.linspace(0, 2 * np.pi, 200)
# xc = RADIUS * np.cos(theta)
# yc = RADIUS * np.sin(theta)
# zc = np.full_like(theta, HEIGHT)

# # plot the circle on the same axes
# ax.plot(xc, yc, zc,
#         linestyle="--",
#         linewidth=1.5,
#         color="black",
#         label=f"Reference circle (r={RADIUS}, z={HEIGHT})")
ax.legend()

# nice-to-have cosmetics
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3-D Path")
ax.legend()
ax.view_init(elev=25, azim=-60)  # adjust viewing angle as you like
plt.tight_layout()
savefig(f"{PREFIX_NAME}end_effector_3d")


plt.figure()
plt.plot(time_log, eta_1_log, label=r'$\eta_1$', color='purple')
plt.plot(time_log, xi_1_log, label=r'$\xi_1$', color='orange')
plt.plot(time_log, xi_3_log, label=r'$\xi_3$', color='teal')
plt.plot(time_log, reference_signal_log, label=r'Reference signal', color='black', linestyle= "--",)


plt.title("Eta and Xi Values Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}Eta_Xi")

plt.figure()
plt.plot(time_log, reference_signal_log, color='black', label='Reference signal')
plt.title("Reference Signal Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Reference Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}reference_signal")

plt.figure()
plt.plot(time_log, reference_velocity_log, color='black', linestyle='--', label='Reference velocity')
plt.title("Reference Velocity Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Reference Velocity")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}reference_velocity")

plt.figure()
plt.plot(time_log, J_hat_determinent_log)

plt.title("J_hat det")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}J_hat_det")

plt.figure()
plt.plot(time_log, E_determinent_log)

plt.title("E det")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}J_det")

plt.figure()
plt.plot(time_log, beta_log, color='blue', label='beta')
plt.title("Beta Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Beta")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}beta")

plt.figure()
for i, lbl, col in zip(range(4), ['qx','qy','qz','qw'],
                    ['red','green','blue','black']):
    plt.plot(time_log, quat_log[:, i],        color=col, label=lbl)
    plt.plot(time_log, quat_des_log[:, i], '--', color=col, label=f'{lbl} (ref)')

plt.title("End-Effector Orientation (Quaternion) Convergence")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion Component")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}Quaternion")

plt.figure()
plt.plot(time_log, Rotation_error_log[:, 0], color = 'red', label='$\zeta_1$')
plt.plot(time_log, Rotation_error_log[:, 1], color = 'green', label='$\zeta_2$')
plt.plot(time_log, Rotation_error_log[:, 2], color = 'blue', label='$\zeta_3$')


plt.title("Rotation error")
plt.xlabel("Time [s]")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.tight_layout()

savefig(f"{PREFIX_NAME}Rerr")

euler_log = np.array(euler_log)
euler_des_log = np.array(euler_des_log)

plt.figure()
labels = ['yaw (Z)', 'pitch (Y)', 'roll (X)']
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.plot(time_log, euler_log[:, i], color=colors[i], label=f'{labels[i]}')
    plt.plot(time_log, euler_des_log[:, i], '--', color=colors[i], label=f'{labels[i]} (ref)')

plt.title("End-Effector Orientation (Euler ZYX) Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}Euler")

plt.figure()
plt.plot(time_log, er_log)
plt.title("$e_R$")
plt.xlabel("Time [s]")
plt.ylabel("error")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}eR")

plt.figure()
plt.plot(time_log, trace_log)
plt.title("$Trace$")
plt.xlabel("Time [s]")
plt.ylabel("Trace")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}trace")

plt.figure()
for i in range(v_log.shape[1]):
    plt.plot(time_log, v_log[:, i], label=f"$v_{i + 1}$")

plt.title("v")
plt.xlabel("Time [s]")
plt.ylabel("v")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}v")

plt.figure()
for i in range(u_log.shape[1]):
    plt.plot(time_log, u_log[:, i], label=f"$u_{i + 1}$")

plt.title("u")
plt.xlabel("Time [s]")
plt.ylabel("u")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}u")

plt.show()
