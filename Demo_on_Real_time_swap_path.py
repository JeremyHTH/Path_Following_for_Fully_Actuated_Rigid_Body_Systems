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
PREFIX_NAME = "Mix_Path_"
os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + ".png"), dpi=300)

# Path generation
N = 5000


r = 0.5
h = 0.3

Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(2 * np.pi  / N * i), h] for i in range(N)]

Rotation_List = []
ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) 

for i in range (N):
    yaw   =  2 * np.pi * i / N + np.pi/ 3 
    pitch = rp 
    roll  =  rr 

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
    cr, sr = np.cos(roll),  np.sin(roll)

    Rotation_List.append(np.array([
        [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
        [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
        [-sp_sin,             cp*sr,             cp*cr       ]
    ]))

Frame_1 = Frame_Path(Point_List, Rotation_List, True, 4)
print("Total path length (Frame 1):", Frame_1.total_length)

r = 0.3
h = 0.5

Point_List = [[r * np.cos(2 * np.pi / N * i), r * np.sin(4 * np.pi  / N * i), h + 0.1 * np.cos(2 * np.pi / N * i)] for i in range(N)]

Rotation_List = []
ry, rp, rr = np.random.uniform(-np.pi, np.pi, 3) 

for i in range (N):
    yaw   =  2 * np.pi * i / N + np.pi/ 3 
    pitch = rp 
    roll  =  rr 

    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp_sin = np.cos(pitch), np.sin(pitch)  # avoid name clash
    cr, sr = np.cos(roll),  np.sin(roll)

    Rotation_List.append(np.array([
        [cy*cp,  cy*sp_sin*sr - sy*cr,  cy*sp_sin*cr + sy*sr],
        [sy*cp,  sy*sp_sin*sr + cy*cr,  sy*sp_sin*cr - cy*sr],
        [-sp_sin,             cp*sr,             cp*cr       ]
    ]))

Frame_2 = Frame_Path(Point_List, Rotation_List, True, 4)


print("Total path length (Frame 2):", Frame_2.total_length)

def Reference_signal(Frame, time):
    if (Frame.Is_loop):
        return (Frame.total_length / 5.0 * time) % (Frame.total_length), Frame.total_length / 5.0, 0
    
    else:
        if (time < 10):
            Position = 0
            Velocity = 0
        
        elif (time < 20):
            Temp_velocity = Frame.total_length / 10
            Temp_position = Temp_velocity * (time - 10)

            Position = Temp_position
            Velocity = Temp_velocity

        elif (time < 30):
            Position = Frame.total_length
            Velocity = 0

        elif (time < 40):
            Temp_velocity = - Frame.total_length / 10
            Temp_position = Frame.total_length + Temp_velocity * (time - 30)

            Position = Temp_position
            Velocity = Temp_velocity

        else:
            Position = 0
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

KukaIiwa14_Control_Module =Virtal_Task_Space_Control_Module(Kp_eta = 60, Kd_eta = 50, Kp_xi = 500, Kd_xi = 50, Kp_rot = 250, Kd_rot = 200)

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

Already_reset_s_star = False

for step in tqdm(range(steps), desc="Simulating", unit="step"):
    if (step * dt < T_total / 2):
        Frame = Frame_1
    else:
        Frame = Frame_2

        if (not Already_reset_s_star):
            KukaIiwa14_Control_Module.reset_s_star()

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

    J_hat_determinent_log.append(np.linalg.det(J_H_7dof))
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
    x_log[:, 0],   
    x_log[:, 1],   
    x_log[:, 2],   
    linewidth=2, 
    label="Path"
)

s_sample = np.linspace(0, Frame_1.total_length, 400, endpoint = True)
p_sample = Frame_1.P(s_sample)    
ax.plot(*p_sample.T, lw=1.5, linestyle= "--", color = "black", label="Reference path 1")


s_sample = np.linspace(0, Frame_2.total_length, 400, endpoint = True)
p_sample = Frame_2.P(s_sample)    
ax.plot(*p_sample.T, lw=1.5, linestyle= "--", color = "grey", label="Reference path 2")
ax.legend()


ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title("3-D Path")
ax.legend()
ax.view_init(elev=25, azim=-60)  
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
plt.plot(time_log, J_hat_determinent_log, label=r"$\det(\hat{J})$")

plt.title("J_hat det")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}J_hat_det")

plt.figure()
plt.plot(time_log, E_determinent_log, label=r"$\det(E)$")

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
plt.plot(time_log, er_log, label=r"$\|e_R\|$")
plt.title("$e_R$")
plt.xlabel("Time [s]")
plt.ylabel("error")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}eR")

plt.figure()
plt.plot(time_log, trace_log, label="trace(R_err)")
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
