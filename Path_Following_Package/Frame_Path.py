from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as Rot

from Path_Following_Package.Path_Generation import Path_Generation_Tool


class Frame_Path(Path_Generation_Tool):

    def __init__(self, points, rotation_matrices, Is_loop, n_continuity, print_equation=False , _samples: int = 4000, frame_method = "parallel"):
        super().__init__(points, rotation_matrices, Is_loop, n_continuity, print_equation)
        self._samples = _samples
        self._build_arc_length_map(self._samples)
        self.frame_method = frame_method
        self._frame_cache: dict[str, np.ndarray] = {}

        self._make_frame()


    def _build_arc_length_map(self, N: int):
        lam = np.linspace(0.0, 1.0, N)
        P = super().P(lam)
        seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seglen)))
        self.total_length = s[-1]
        # s_norm = s / self.total_length

        # self._lam_of_s = interp1d(s_norm, lam, kind="linear", assume_sorted=True)
        # self._s_of_lam = interp1d(lam, s_norm, kind="linear", assume_sorted=True)

        self._lam_of_s = make_interp_spline(s, lam, k = 3)
        self._s_of_lam = make_interp_spline(lam, s, k = 3)

    def Ensure_in_range(self, s):
        if self.Is_loop:
            return (s + self.total_length) % self.total_length
        else:
            return np.clip(s, 0, self.total_length)

    def P(self, s):
        s = self.Ensure_in_range(s)

        lam = self._lam_of_s(s)
        return super().P(lam)

    def R(self, s):
        s = self.Ensure_in_range(s)
        lam = self._lam_of_s(s)
        return super().R(lam)


    def _make_frame(self):
        if (self.frame_method == "parallel"):
            self._make_parallel_transport_frame()
        else:
            raise Exception("Frame method not found")

    def _make_parallel_transport_frame(self):
        s = np.linspace(0, self.total_length, 20000)


        T = self._dP_ds(s, 1)
        M1 = np.zeros_like(T)
        M2 = np.zeros_like(T)

        Initial_guess_reference = np.array([0, 0, 1]) if abs(T[0][2]) < 0.9 else np.array([0, 1, 0])

        M1_0 = np.cross(T[0], Initial_guess_reference)
        M1[0] = M1_0 / np.linalg.norm(M1_0)

        # norm = np.linalg.norm(M1_0)
        # if norm > 1e-8:
        #     M1[0] = M1_0 / norm
        # else:
        #     M1[0] = np.array([1.0, 0.0, 0.0]) 

        M2[0] = np.cross(T[0], M1[0])

        for i in range(1, len(s)):
            proj = M1[i - 1] - np.dot(M1[i - 1], T[i]) * T[i]
            M1[i] = proj / np.linalg.norm(proj)
            M2[i] = np.cross(T[i], M1[i])

        self._frame_cache = {"s": s, "T": T, "M1": M1, "M2": M2}

    def _eval_dP_dLambda(self, lam, order):
        lam = np.asarray(lam, dtype=float)
        return np.column_stack(
            [spl.derivative(order)(lam) for spl in self.positional_spine]
        )

    def _dP_ds(self, s, order=1):
        s   = self.Ensure_in_range(s)
        lam = self._lam_of_s(s)

        if order == 1:
            P_l = self._eval_dP_dLambda(lam, 1)

            # 20250627 Update non-loop path
            norms = np.linalg.norm(P_l, axis=1, keepdims=True)
            # T   = P_l / np.linalg.norm(P_l, axis=1, keepdims=True)
            T = np.divide(P_l, norms, out=np.full_like(P_l, [1.0, 0.0, 0.0]), where=(norms > 1e-8))

            #========
            # return T[0] if T.shape[0] == 1 else T
            return T

        elif order == 2:
            P_l  = self._eval_dP_dLambda(lam, 1)
            P_ll = self._eval_dP_dLambda(lam, 2)
            ds_dλ = np.linalg.norm(P_l, axis=1, keepdims=True)
            
            # 20250627 Update non-loop path
            # P_ss  = (P_ll - np.sum(P_ll*P_l, axis=1, keepdims=True)*P_l/ds_dλ**2) / ds_dλ**2

            epsilon = 1e-8
            safe_denom = np.where(ds_dλ**2 > epsilon, ds_dλ**2, epsilon)
            projection = np.sum(P_ll * P_l, axis=1, keepdims=True) * P_l / safe_denom
            P_ss = (P_ll - projection) / safe_denom
            #======
            # return P_ss[0] if P_ss.shape[0] == 1 else P_ss
            return P_ss
        else:
            raise ValueError("order must be 1 or 2")

    def get_frame(self, s):
        if self.frame_method != "parallel":
            raise Exception("Get frame error")

        s_arr = np.asarray(s, dtype=float)
        scalar_input = s_arr.ndim == 0
        flat_s = np.atleast_1d(s_arr).reshape(-1)

        T = self._dP_ds(flat_s, 1)

        M1_guess = np.column_stack([
            np.interp(flat_s, self._frame_cache["s"], self._frame_cache["M1"][:, k])
            for k in range(3)
        ])

        proj = np.sum(M1_guess * T, axis=1, keepdims=True)
        M1 = M1_guess - proj * T
        M1_norms = np.linalg.norm(M1, axis=1, keepdims=True)
        M1 = np.divide(M1, M1_norms, out=np.copy(M1), where=M1_norms > 1e-8)

        M2 = np.cross(T, M1)
        M2_norms = np.linalg.norm(M2, axis=1, keepdims=True)
        M2 = np.divide(M2, M2_norms, out=np.copy(M2), where=M2_norms > 1e-8)

        if scalar_input:
            return T[0], M1[0], M2[0]

        new_shape = s_arr.shape + (3,)
        return T.reshape(new_shape), M1.reshape(new_shape), M2.reshape(new_shape)

    def visualize_frame(self, L = 1, num_of_frame_plots = 10):
        
        s = np.linspace(0.0,self.total_length, 400, endpoint=not self.Is_loop)
        pos_s = self.P(s)
        if (self.frame_method == "parallel"):
            labels = ("T", "$M_1$", "$M_2$")

        T, A1, A2 = self.get_frame(s)
        frame_vectors = np.hstack((T, A1, A2))
        output_path = Path(__file__).with_name("frame_vectors.csv")
        np.savetxt(
            output_path,
            frame_vectors,
            delimiter=",",
            header="Tx,Ty,Tz,A1x,A1y,A1z,A2x,A2y,A2z",
            comments="",
        )

        cols = ("r", "g", "b")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*pos_s.T, lw=1.8)
        for i in  np.linspace(0, len(s) - 1, num_of_frame_plots, dtype=int):
            p = pos_s[i]
            for v, c in zip((T[i], A1[i], A2[i]), cols):

                # # vec_norm = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-12 else vec
                ax.quiver(*p, *v, color=c, length=L, normalize=True)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect([1, 1, 1])
        for lab, c in zip(labels, cols):
            ax.quiver([], [], [], [], [], [], color=c, label=lab)

        getattr(ax, "set_box_aspect", lambda *_: None)([1, 1, 1])
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"{self.frame_method} frame (intrinsic unit-speed)")

        plt.show()
    
    def num_derivative(self, func, x, h=1e-6):
        return (func(x + h) - func(x - h)) / (2.0 * h)
    
    def find_s_star(self, y, last_s = None, n_init=10, tol=1e-6, n_max=50):
        y = np.asarray(y, dtype=float)
        s_bound = (0.0, self.total_length)
        if (last_s == None):
            s_grid = np.linspace(0, self.total_length, n_init)
        else: 
            s_grid = [last_s]

        best_s, best_distance = None, np.inf

        for s0 in s_grid:
            s = s0
            for _ in range(n_max):
                p  = self.P(s)
                p_d = self.num_derivative(self.P, s)
                # f   = np.dot(p - y, p_d)            
                f = np.dot((p - y).flatten(), p_d.flatten())
                f_d = self.num_derivative(
                        lambda l: np.dot((self.P(l) - y).ravel(), self.num_derivative(self.P, l).ravel()),
                        s
                    )
                if abs(f_d) < 1e-12:        
                    break
            
                s = s - f / f_d

            distance = np.sum((self.P(s) - y)**2)
            if distance < best_distance:
                best_distance, best_s = distance, s

        best_s_before = best_s
        if (self.Is_loop):
            while best_s < 0:
                best_s += s_bound[1] - s_bound[0]
            best_s = np.fmod((best_s - s_bound[0]), (s_bound[1] - s_bound[0])) + s_bound[0]
            # Lambda_new = np.clip(Lambda_new, Lambda_bounds[0], Lambda_bounds[1]) 
            # pass
        else: 
            best_s = np.clip(best_s, s_bound[0], s_bound[1]) 

        if (best_s < 0):
            raise Exception(f"negative s {best_s_before}{best_s}, {s_bound[0]} {s_bound[1]}")

        return best_s, self.P(best_s)
    
    def visualize_closest_point(self, query_pts = [[0, 0, 0]]):

        s = np.linspace(0.0,self.total_length, 400, endpoint=not self.Is_loop)
        pos_s = self.P(s)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*pos_s.T, lw=1.8)

        for q in query_pts:
            s_star, p_star = self.find_s_star(q)
            p_star = np.ravel(p_star)
            # print(s_star)
            ax.scatter(*q,  color='red',  s=60)
            ax.scatter(*p_star, color='blue', s=60)
            ax.plot([q[0], p_star[0]],
                    [q[1], p_star[1]],
                    [q[2], p_star[2]], linestyle='--', color='gray')

        ax.set_title("Closest points on a general 3-D path")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        ax.plot([], [], color='black', label='Path p(s)')
        ax.scatter([], [], color='red',  label='Query point')
        ax.scatter([], [], color='blue', label='Closest point p(s*)')
        ax.legend()
        plt.tight_layout()
        plt.show()
