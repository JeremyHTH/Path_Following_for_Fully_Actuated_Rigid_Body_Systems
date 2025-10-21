import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from importlib.util import spec_from_file_location, module_from_spec



from scipy.linalg import logm

import os


from Path_Following_Package import *

SAVE_DIR = "Demo_Image"
PREFIX_NAME = "Simple_path_2_"
os.makedirs(SAVE_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(SAVE_DIR, name + ".png"), dpi=300)

# Path generation
N = 5000

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

Frame = Frame_Path(Point_List, Rotation_List, True, 4, False)

# Tracking Signal

def Reference_signal(Frame, time):
    # return 0.1, 0, 0
    if (Frame.Is_loop):
        # return 0.5 * np.sin(time * 2 * np.pi / 5) + 0.5, 0.5 * np.cos(time * 2 * np.pi / 5) * 2 * np.pi / 5, 0.5 * -np.sin(time * 2 * np.pi / 5)* 2 * np.pi /5 * 2 * np.pi / 5
    
        return (Frame.total_length / 5.0 * time) % (Frame.total_length), Frame.total_length / 5.0, 0
    else:
        Temp_velocity = Frame.total_length / 10
        Temp_position = Temp_velocity * (time - 10)

        if (Temp_position >= 0 and Temp_position < Frame.total_length * 0.9999):
            Position = Temp_position
            Velocity = Temp_velocity

        elif (Temp_position < 0):
            Position = 0
            Velocity = 0

        else:
            Position = Frame.total_length * 0.9999
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

KukaIiwa14_Control_Module = Control_Module()

# Simulation parameters
dt = 0.005
T_total = 20

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
x_des_log = []
lambda_log = []
J_hat_determinent_log = []
J_determinent_log = []

quat_log = []
quat_des_log = []
Rotation_error_log = []

euler_log = []
euler_des_log = []

er_log = []

trace_log = []

u_log = []

J_H_log = []


# Initial condition
q = np.random.uniform(0.1, 0.1, size=7)
qd = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
qdd = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)

R_cur = KukaIiwa14.forward_kinematics(q)[:3, :3]
R_des = Frame.R(0).as_matrix().reshape((3, 3))

last_s_star = None
omega_des_prev = np.zeros((3))

for step in tqdm(range(steps), desc="Simulating", unit="step"):
    time = step * dt
    reference_signal_pos, reference_signal_velocity, reference_signal_acceleration = Reference_signal(Frame, time)

    # Forward kinematics
    T_cur = KukaIiwa14.forward_kinematics(q)
    x_cur = T_cur[:3, 3]
    R_cur = T_cur[:3, :3]

    y      = KukaIiwa14.forward_kinematics(q)[:3, 3].reshape((3, 1))
    y_dot  = (KukaIiwa14.jacobian(q)[:3] @ qd).reshape((3, 1))
    y_ddot = (KukaIiwa14.jacobian_dot(q, qd)[:3] @ qd + \
            KukaIiwa14.jacobian(q)[:3] @ qdd).reshape((3, 1))

    s_star, _ = Frame.find_s_star(y.T, last_s = last_s_star)
    last_s_star = s_star

    y_star = Frame.P(s_star).reshape((3, 1))
    y_star_d = Frame._dP_ds(s_star, 1).reshape((3, 1))
    y_star_dd = Frame._dP_ds(s_star, 2).reshape((3, 1))

    T, M1, M2 = Frame.get_frame(s_star)
    T = T.reshape((3, 1))
    M1 = M1.reshape((3, 1))
    M2 = M2.reshape((3, 1))

    # print((T).shape)
    # beta = 1 / (1 -  ((y - y_star).T @ y_star_dd)/ (np.linalg.norm(y_star_d) ** 2))

    epsilon = 1e-8
    beta_raw = 1 - ((y - y_star).T @ y_star_dd) / (np.linalg.norm(y_star_d) ** 2)
    beta = 1.0 / np.clip(beta_raw, epsilon, None)
    # print(beta)

    s_star_d = (beta @ T.T / np.linalg.norm(y_star_d)) @ y_dot

    eta_1 = s_star
    eta_2 = float(s_star_d)    

    xi_1 = M1.T @ (y - y_star)
    xi_3 = M2.T @ (y - y_star)

    xi_2 = M1.T @ y_dot
    xi_4 = M2.T @ y_dot

    ds_dt  = float(eta_2)
    beta   = float(beta)
    beta_d = float((beta**2) * (y_star_dd.T @ y_dot) /
                (np.linalg.norm(y_star_d)**2))
    
    
    # --- third arc-length derivative y_star_ddd ----------------------------
    delta_s = 1e-4 * Frame.total_length        # adaptive step
    P_m2 = Frame.P(s_star - 2*delta_s).reshape((3,1))
    P_m1 = Frame.P(s_star - 1*delta_s).reshape((3,1))
    P_p1 = Frame.P(s_star + 1*delta_s).reshape((3,1))
    P_p2 = Frame.P(s_star + 2*delta_s).reshape((3,1))

    # central finite-difference, O(h²) accurate
    y_star_ddd = ( P_m2 - 2*P_m1 + 2*P_p1 - P_p2 ) / (2*delta_s**3)
    beta_d_extra = float( beta**2 * ((y - y_star).T @ y_star_ddd)
                / np.linalg.norm(y_star_d)**2 ) * ds_dt
    beta_d      += -beta_d_extra 

    small   = 1e-6
    T_p,M1_p,M2_p = Frame.get_frame(s_star + small)
    T_m,M1_m,M2_m = Frame.get_frame(s_star - small)

    T_p = T_p.reshape((3, 1))
    M1_p = M1_p.reshape((3, 1))
    M2_p = M2_p.reshape((3, 1))

    T_m = T_m.reshape((3, 1))
    M1_m = M1_m.reshape((3, 1))
    M2_m = M2_m.reshape((3, 1))

    dT_ds  = (T_p - T_m)  /(2*small)
    dM1_ds = (M1_p- M1_m) /(2*small)

    T_dot     = dT_ds   * ds_dt
    # print(T_dot.shape)

    coeff_perp=  float(T_dot.T @ M1)
    M1_dot    = -(coeff_perp) * T

    M2_dot    = np.cross(T_dot.flatten(), M1.flatten()) \
            + np.cross(T.flatten(),    M1_dot.flatten())
    M2_dot    = M2_dot.reshape(3,1)

    norm_y = float(np.linalg.norm(y_star_d))

    beta_bar     = beta / norm_y
    beta_bar_dot = (beta_d / norm_y
                - beta * (y_star_d.T @ y_star_dd) * ds_dt / norm_y**3)

    E     = np.vstack([beta_bar     * T.T ,
                    M1.T                ,
                    M2.T                ])

    E_dot = np.vstack([beta_bar_dot * T.T + beta_bar * T_dot.T ,
                    M1_dot.T                           ,
                    M2_dot.T                           ])

    
    J = KukaIiwa14.jacobian(q)
    J_dot = KukaIiwa14.jacobian_dot(q, qd)
    Jv = J[:3, :]
    Jw = J[3:, :]

    h = np.zeros((7))
    h[0] = float(eta_1)
    h[1] = float(xi_1)
    h[2] = float(xi_3)
    h[3] = q[2]
    h[4] = q[4]
    h[5] = q[5]
    h[6] = q[2]

    hd = np.zeros((7))
    hd[0] = float(eta_2)
    hd[1] = float(xi_2)
    hd[2] = float(xi_4)
    hd[3] = qd[2]
    hd[4] = qd[4]
    hd[5] = qd[5]
    hd[6] = qd[2]

    J_H = np.zeros((7, 7))
    J_H [:3, :] = E @ Jv
    J_H[3, 2] = 1
    J_H[4, 4] = 1
    J_H[5, 5] = 1
    J_H[6, 2] = 1

    J_H_dot = np.zeros((7, 7))
    J_H_dot[:3, :] = E_dot @ Jv + E @ J_dot[:3, :]

    # R_des = Frame.R(s_star).as_matrix().reshape((3, 3))
    # R_cur = T_cur[:3, :3]

    # # Jacobians (angular part)
    Jw     = KukaIiwa14.jacobian(q)[3:, :]
    Jw_dot = KukaIiwa14.jacobian_dot(q, qd)[3:, :]   # ← fix: keep the ':' for columns

    # current & desired spatial orientations
    R_des  = Frame.R(reference_signal_pos).as_matrix().reshape((3, 3))  # σ_r(s*)
    R_cur  = T_cur[:3, :3]                                # Π_2

    # orientation error: f = Log(Rdᵀ R)
    R_err  = R_des.T @ R_cur               # Rdᵀ R
    e_rot  = log_SO3(R_err)              # f∨ ∈ ℝ³ (vee of Log)

    # ---- desired body-rate ω_d via finite-difference along s ----
    ds       = 1e-5
    R_p      = Frame.R(reference_signal_pos + ds).as_matrix().reshape((3, 3))
    R_m      = Frame.R(reference_signal_pos - ds).as_matrix().reshape((3, 3))
    R_des_d  = (R_p - R_m) / (2.0 * ds) * float(reference_signal_velocity)     # spatial Ṙ_d

    omega_d_hat = R_des.T @ R_des_d                           # ω̂_d = R_dᵀ Ṙ_d (body)
    # project to skew to clean FD noise
    omega_d_hat = 0.5 * (omega_d_hat - omega_d_hat.T)

    omega_des = np.array([omega_d_hat[2,1], omega_d_hat[0,2], omega_d_hat[1,0]])

    # ---- desired body angular acceleration ω̇_d (FD; simple) ----
    R_pp      = Frame.R(reference_signal_pos + 2*ds).as_matrix().reshape((3, 3))
    R_mm      = Frame.R(reference_signal_pos - 2*ds).as_matrix().reshape((3, 3))
    R_des_dd  = (R_pp - 2*R_p + 2*R_m - R_mm) / (2.0 * ds**2) * float(reference_signal_velocity**2)
    # Correct frame/order: Ω̇_d^ = R_dᵀ R̈_d − Ω̂_d Ω̂_d
    omega_d_dot_hat = R_des.T @ R_des_dd - omega_d_hat @ omega_d_hat
    omega_d_dot_hat = 0.5 * (omega_d_dot_hat - omega_d_dot_hat.T)

    omega_des_dot = np.array([omega_d_dot_hat[2,1],
                            omega_d_dot_hat[0,2],
                            omega_d_dot_hat[1,0]])

    # ---- current body rate Ω from spatial Jacobian ----
    omega_spatial = Jw @ qd                 # spatial angular velocity
    Omega         = R_cur.T @ omega_spatial # body angular velocity

    # hats
    Omega_hat      = hat(Omega)
    Omega_d_hat    = hat(omega_des)
    Omega_d_dot_hat= hat(omega_des_dot)

    # convenience
    R_t_Rd = R_cur.T @ R_des   # A = Rᵀ R_d
    Rd_t_R = R_des.T @ R_cur   # B = R_dᵀ R

    # ḟ∨ = Ω − (RᵀR_d) Ω_d
    f_dot = Omega - (R_t_Rd @ omega_des)

    # gains (Hurwitz)
    K1 = 250.0
    K2 = 200.0

    # ---- controller û (matrix/hat form), eq. (17) ----
    term1 = R_t_Rd @ Omega_d_dot_hat @ Rd_t_R
    term2 = (R_t_Rd @ Omega_d_hat - Omega_hat @ R_t_Rd) @ Omega_d_hat @ Rd_t_R
    term3 = R_t_Rd @ Omega_d_hat @ (R_t_Rd @ Omega_hat - Omega_d_hat @ R_t_Rd)
    u_hat = term1 + term2 + term3 \
            - hat(K1 * e_rot) - hat(K2 * f_dot)

    # back to vector (vee) and convert to spatial α if needed upstream
    v_w_body = np.array([u_hat[2,1], u_hat[0,2], u_hat[1,0]])  # body angular accel command
    v_w      = R_cur @ v_w_body                                # spatial angular accel (if needed)

    # ── fill hybrid task stack ─────────────────────────────────────
    h[3:6]       = e_rot
    hd[3:6]      = Omega
    J_H[3:6, :]  = Jw
    J_H_dot[3:6,:]= Jw_dot

    J_H_log.append(J_H.copy())

    M, C, Fv, Fc, G = KukaIiwa14.dynamics(q, qd)

    
    u = np.zeros((7))
    v = np.array([
        reference_signal_acceleration + 50 * (reference_signal_velocity - hd[0]) + 60 * Angular_error(h[0], reference_signal_pos, Frame),
        Kd * (-hd[1]) + Kp * (-h[1]),
        Kd * (-hd[2]) + Kp * (-h[2]),
        Kd * (-hd[3]) + Kp * (-h[3]),
        Kd * (-hd[4]) + Kp * (-h[4]),
        Kd * (-hd[5]) + Kp * (-h[5]),
        Kd * (-hd[6]) + Kp * (-h[6]),
    ])

    v[3:6] = v_w

    u = np.linalg.inv(J_H) @ (-J_H_dot @ qd + v)
    u = np.clip(u, -30, 30)

    u_module, _ = KukaIiwa14_Control_Module.Get_Control_Input(KukaIiwa14, Frame, q, qd, qdd, Reference_signal(Frame, time))
    u_module = np.clip(u_module, -30, 30)
    print("Difference in control input norm:", np.linalg.norm(u - u_module))

    if (time in {4.995, 5.000, 5.005}):
        print("Debug")
        print("time", time)
        print("eta2", (reference_signal_velocity - hd[0]))
        print("eta1", Angular_error(h[0], reference_signal_pos, Frame))
        print("s_star", s_star)
        print("s_star_d", s_star_d)
        print("xi1", h[1])
        print("xi2", hd[1])
        print("xi3", h[2])
        print("xi4", hd[2])
        print("v\n", v)
        print("u\n", u)
        print("R_des\n", R_des)
        print("R_cur\n", R_cur)
        print("R_err\n", R_err)
        print("e_rot\n", e_rot)
        print("u_hat\n", u_hat)
        
        

    tau = M @ u + C @ qd + Fv @ qd + Fc @ np.sign(qd) + G

    qdd = KukaIiwa14.forward_dynamics(q, qd, tau)
    qd += qdd * dt
    q += qd * dt
    q = (q + np.pi) % (2 * np.pi) - np.pi

    time_log.append(time)
    q_log.append(q.copy())
    x_log.append(x_cur.copy())
    eta_1_log.append(float(eta_1))
    xi_1_log.append(float(xi_1))
    xi_3_log.append(float(xi_3))
    reference_signal_log.append(float(reference_signal_pos))
    x_des_log.append(y_star.flatten())
    # lambda_log.append(lambda_star)

    J_hat_determinent_log.append(np.linalg.det(J_H))
    # J_determinent_log.append(np.linalg.matrix_rank(J))
    J_determinent_log.append(np.linalg.det(E))


    quat_log.append(R.from_matrix(R_cur).as_quat())
    quat_des_log.append(R.from_matrix(R_des).as_quat())

    euler_log.append(R.from_matrix(R_cur).as_euler('ZYX', degrees=False))
    euler_des_log.append(R.from_matrix(R_des).as_euler('ZYX', degrees=False)) 

    Rotation_error_log.append(e_rot)

    er_log.append(np.linalg.norm(e_rot))

    trace_log.append(np.trace(R_err))

    u_log.append(u)

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
plt.plot(time_log, J_hat_determinent_log)

plt.title("J_hat det")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}J_hat_det")

plt.figure()
plt.plot(time_log, J_determinent_log)

plt.title("J det")
plt.xlabel("Time [s]")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
savefig(f"{PREFIX_NAME}J_det")

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
