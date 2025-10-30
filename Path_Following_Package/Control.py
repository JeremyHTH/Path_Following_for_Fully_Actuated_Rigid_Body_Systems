import numpy as np
from Path_Following_Package.util import Angular_error, log_SO3, hat

class Virtal_Task_Space_Control_Module:
    def __init__(self, last_s_star = None, Kp_eta = 50.0, Kd_eta = 60.0, Kp_xi = 100.0, Kd_xi = 20.0, Kp_rot = 25.0, Kd_rot = 20.0):
        """
        Configure the virtual task-space controller with initial state and gains.

        Parameters
        ----------
        last_s_star : float or None
            Initial guess for the closest arc-length on the path (used as warm start).
        Kp_eta : float
            Proportional gain regulating tangential progress errors (η terms).
        Kd_eta : float
            Damping gain for tangential velocity errors.
        Kp_xi : float
            Proportional gain for lateral deviation errors (ξ terms).
        Kd_xi : float
            Damping gain for lateral velocity errors.
        Kp_rot : float
            Gain acting on the orientation error vector.
        Kd_rot : float
            Gain acting on the angular velocity error.

        Returns
        -------
        None
        """
        self.last_s_star = last_s_star
        self.Kp_eta = Kp_eta
        self.Kd_eta = Kd_eta
        self.Kp_xi = Kp_xi
        self.Kd_xi = Kd_xi
        self.Kp_rot = Kp_rot
        self.Kd_rot = Kd_rot

    def Get_Control_Input(self, Robot, Frame, q, qd, qdd, Reference_signal, Use_previous_s_star = True, Number_of_seeds_for_s_star = 10):
        """
        Compute joint-space control inputs that drive the robot along the reference path.

        Parameters
        ----------
        Robot : Robot
            Robot model providing kinematics and dynamics information.
        Frame : Frame_Path
            Path frame object used to evaluate closest points and Frenet frames.
        q : ndarray
            Current joint positions (rad or m depending on joint type).
        qd : ndarray
            Current joint velocities.
        qdd : ndarray
            Current joint accelerations (feed-forward term).
        Reference_signal : tuple
            Desired arc-length position, velocity, and acceleration along the path.
        Use_previous_s_star : bool, optional
            Whether to store the latest s* as the next warm-start seed.
        Number_of_seeds_for_s_star : int, optional
            Number of initial guesses used when searching for the closest point.

        Returns
        -------
        tuple
            A pair `(v, Task_space_variable)` where `v` is the 6D task-space
            control effort and `Task_space_variable` bundles intermediate
            diagnostics for logging or analysis.
        """
        reference_signal_pos, reference_signal_velocity, reference_signal_acceleration = Reference_signal

        number_of_joints = q.shape[0]

        T_cur = Robot.forward_kinematics(q)
        x_cur = T_cur[:3, 3]
        R_cur = T_cur[:3, :3]

        y      = Robot.forward_kinematics(q)[:3, 3].reshape((3, 1))
        y_dot  = (Robot.jacobian(q)[:3] @ qd).reshape((3, 1))
        y_ddot = (Robot.jacobian_dot(q, qd)[:3] @ qd + \
                Robot.jacobian(q)[:3] @ qdd).reshape((3, 1))

        # Search the path for the closest point to the current end-effector pose.
        s_star, _ = Frame.find_s_star(y.T, last_s = self.last_s_star, n_init=Number_of_seeds_for_s_star)

        if (Use_previous_s_star):
            self.last_s_star = s_star

        y_star = Frame.P(s_star).reshape((3, 1))
        y_star_d = Frame._dP_ds(s_star, 1).reshape((3, 1))
        y_star_dd = Frame._dP_ds(s_star, 2).reshape((3, 1))

        T, M1, M2 = Frame.get_frame(s_star)
        T = T.reshape((3, 1))
        M1 = M1.reshape((3, 1))
        M2 = M2.reshape((3, 1))

        epsilon = 1e-8
        # beta rescales the tangential velocity to compensate for curvature-induced drift.
        beta_raw = 1 - ((y - y_star).T @ y_star_dd) / (np.linalg.norm(y_star_d) ** 2)
        beta = 1.0 / np.clip(beta_raw, epsilon, None)
        # beta = 1.0 / beta_raw

        # print((y - y_star),"\n", y_star_dd, "\n", np.linalg.norm(y_star_d) ** 2, beta_raw, beta)

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
        
        delta_s = 1e-4 * Frame.total_length    
        # Use a five-point stencil to approximate third derivatives required in betȧ.
        P_m2 = Frame.P(s_star - 2*delta_s).reshape((3,1))
        P_m1 = Frame.P(s_star - 1*delta_s).reshape((3,1))
        P_p1 = Frame.P(s_star + 1*delta_s).reshape((3,1))
        P_p2 = Frame.P(s_star + 2*delta_s).reshape((3,1))

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

        Jw     = Robot.jacobian(q)[3:, :]

        R_des  = Frame.R(reference_signal_pos).as_matrix().reshape((3, 3)) 
        R_cur  = T_cur[:3, :3]                                

        R_err  = R_des.T @ R_cur 
        e_rot  = log_SO3(R_err)  

        ds       = 1e-5
        # Finite difference the path orientation to obtain desired angular rates.
        R_p      = Frame.R(reference_signal_pos + ds).as_matrix().reshape((3, 3))
        R_m      = Frame.R(reference_signal_pos - ds).as_matrix().reshape((3, 3))
        R_des_d  = (R_p - R_m) / (2.0 * ds) * float(reference_signal_velocity)     

        omega_d_hat = R_des.T @ R_des_d                           
        omega_d_hat = 0.5 * (omega_d_hat - omega_d_hat.T)

        omega_des = np.array([omega_d_hat[2,1], omega_d_hat[0,2], omega_d_hat[1,0]])

        R_pp      = Frame.R(reference_signal_pos + 2*ds).as_matrix().reshape((3, 3))
        R_mm      = Frame.R(reference_signal_pos - 2*ds).as_matrix().reshape((3, 3))
        R_des_dd  = (R_pp - 2*R_p + 2*R_m - R_mm) / (2.0 * ds**2) * float(reference_signal_velocity**2)
        omega_d_dot_hat = R_des.T @ R_des_dd - omega_d_hat @ omega_d_hat
        omega_d_dot_hat = 0.5 * (omega_d_dot_hat - omega_d_dot_hat.T)

        omega_des_dot = np.array([omega_d_dot_hat[2,1],
                                omega_d_dot_hat[0,2],
                                omega_d_dot_hat[1,0]])

        omega_spatial = Jw @ qd                 
        Omega         = R_cur.T @ omega_spatial 

        Omega_hat      = hat(Omega)
        Omega_d_hat    = hat(omega_des)
        Omega_d_dot_hat= hat(omega_des_dot)

        R_t_Rd = R_cur.T @ R_des   
        Rd_t_R = R_des.T @ R_cur   

        f_dot = Omega - (R_t_Rd @ omega_des)

        # The feed-forward body torque u_hat reproduces the geometric controller from the paper.
        term1 = R_t_Rd @ Omega_d_dot_hat @ Rd_t_R
        term2 = (R_t_Rd @ Omega_d_hat - Omega_hat @ R_t_Rd) @ Omega_d_hat @ Rd_t_R
        term3 = R_t_Rd @ Omega_d_hat @ (R_t_Rd @ Omega_hat - Omega_d_hat @ R_t_Rd)
        u_hat = term1 + term2 + term3 \
                - hat(self.Kp_rot * e_rot) - hat(self.Kd_rot * f_dot)

        v_w_body = np.array([u_hat[2,1], u_hat[0,2], u_hat[1,0]])  
        v_w      = R_cur @ v_w_body                                

        J = Robot.jacobian(q)
        J_dot = Robot.jacobian_dot(q, qd)
        Jv = J[:3, :]
        Jw = J[3:, :]

        h = np.zeros((6))
        h[0] = float(eta_1)
        h[1] = float(xi_1)
        h[2] = float(xi_3)
        h[3:6]       = e_rot

        hd = np.zeros((6))
        hd[0] = float(eta_2)
        hd[1] = float(xi_2)
        hd[2] = float(xi_4)
        hd[3:6]      = Omega

        J_H = np.zeros((6, number_of_joints))
        J_H [:3, :] = E @ Jv
        J_H[3:6, :]  = Jw

        J_H_dot = np.zeros((6, number_of_joints))
        J_H_dot[:3, :] = E_dot @ Jv + E @ J_dot[:3, :]

        
        v = np.zeros((6))
        v[0] = reference_signal_acceleration + self.Kd_eta * (reference_signal_velocity - hd[0]) + self.Kp_eta * Angular_error(h[0], reference_signal_pos, Frame)
        v[1] = self.Kd_xi * (-hd[1]) + self.Kp_xi * (-h[1])
        v[2] = self.Kd_xi * (-hd[2]) + self.Kp_xi * (-h[2])
        v[3:6] = v_w


        Task_space_variable = {
            "eta_1": eta_1,
            "eta_2": eta_2,
            "xi_1": xi_1,
            "xi_2": xi_2,
            "xi_3": xi_3,
            "xi_4": xi_4,
            "s_star": s_star,
            "s_star_d": s_star_d,
            "beta": beta,
            "beta_d": beta_d,
            "J_H": J_H,
            "J_H_dot": J_H_dot, 
            "h": h,
            "hd": hd,
            "E": E,
            "e_rot": e_rot,
            "R_err": R_err,
        }

        return v, Task_space_variable

    def reset_s_star(self):
        """
        Clear the stored closest arc-length to force a fresh search next time.

        Returns
        -------
        None
        """
        self.last_s_star = None