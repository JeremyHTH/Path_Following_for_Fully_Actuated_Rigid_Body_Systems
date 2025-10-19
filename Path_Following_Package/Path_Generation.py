from __future__ import annotations
import numpy as np
import sympy as sp
from scipy.interpolate import make_interp_spline, PPoly
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

class Path_Generation_Tool:

    def __init__(self, points, rotation_matrices, Is_loop, n_continuity, print_equation=False):
        self.points     = np.array(points, dtype=float)
        self.rotation_matrices   = np.array(rotation_matrices, dtype=float)  
        self.Is_loop    = bool(Is_loop)

        if (not self.Is_loop and n_continuity % 2):
            print("[Info] Adjusted n_continuity to be even for open spline to ensure generated correct gradient at endpoints.")
            n_continuity += 1

        self.n_continuity = n_continuity
        self.spline_degree = n_continuity + 1
        self.print_equation = print_equation

        if self.rotation_matrices.ndim != 3 or self.rotation_matrices.shape[1:] != (3, 3):
            raise ValueError("rotation_matrices must be of shape (N, 3, 3)")

        self.positional_spine = None
        self.rotaional_spine = None

        self.Generate_Path()

        
    def _auto_bc(self, k: int):
        if k % 2:                                  
            return "not-a-knot"
        else:
            raise Exception("Even degree splines is not supported for open splines to ensure correct gradient.")
        
    def _make_spline(self, x, y, k: int):
        if self.Is_loop:
            if len(x) <= k:
                raise ValueError(f"periodic spline: need > k={k} points")
            return make_interp_spline(x, y, k=k, bc_type="periodic")
        if len(x) < k + 1:
            raise ValueError(f"open spline: need ≥ k+1={k+1} points")
        return make_interp_spline(x, y, k=k, bc_type=self._auto_bc(k))

    def _make_quat_spline(self, lam_key, quat_key, k_wanted=4):
        n = len(lam_key)
        k = min(k_wanted, n - 1)      
        quat_key = quat_key.copy().astype(float)

        for i in range(1, n):
            if np.dot(quat_key[i - 1], quat_key[i]) < 0.0:
                quat_key[i] = -quat_key[i]

        if self.Is_loop:
            if not np.allclose(quat_key[0], quat_key[-1]):
                print("[info] Closing quaternion loop: appending first quaternion to end")
                quat_key = np.vstack([quat_key, quat_key[0]])

                if not np.isclose(lam_key[-1], 1.0):
                    lam_key = np.append(lam_key, 1.0)
                else:
                    lam_key = np.append(lam_key, lam_key[-1] + (lam_key[-1] - lam_key[-2]))

                n += 1

        if self.Is_loop:
            bc = "periodic"
        else:                             
            bc = self._auto_bc(k)             

        self.rotaional_spine = [make_interp_spline(lam_key, quat_key[:, j], k=k, bc_type=bc)
                for j in range(4)]

    def _Lambda_wrap(self, Lambda):         
        return np.mod(Lambda, 1.0) if self.Is_loop else np.clip(Lambda, 0.0, 1.0)
    
    def P(self, Lambda):    
        Wrapped_Lambda = self._Lambda_wrap(Lambda)
        return np.column_stack([p(Wrapped_Lambda) for p in self.positional_spine])
    
    def R(self, Lambda):
        Wrapped_Lambda = self._Lambda_wrap(Lambda)
        q = np.column_stack([s(Wrapped_Lambda) for s in self.rotaional_spine])
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return Rot.from_quat(q)

    def Rotation_vector(self, Lambda):     
        Wrapped_Lambda = self._Lambda_wrap(Lambda)      
        return self.R(Wrapped_Lambda).as_rotvec()
    
    def Print_spine(self, spl):
        Lambda = sp.Symbol("λ", real=True)
        pp, deg = PPoly.from_spline(spl), spl.k
        pieces = []
        for k in range(pp.c.shape[1]):
            Lambda0 = pp.x[k]
            poly = sum(pp.c[i, k] * (Lambda - Lambda0) ** (deg - i)
                       for i in range(deg + 1))
            pieces.append((sp.expand(poly), (Lambda >= Lambda0) & (Lambda <= pp.x[k + 1])))
        return sp.Piecewise(*pieces)

    def Generate_Path(self):

        if len(self.points) != len(self.rotation_matrices): 
            print("Length of points and angle does not match")
            return 

        pos_loop = self.Is_loop and not np.allclose(self.points[0], self.points[-1])
        rot_loop = self.Is_loop and not np.allclose(self.rotation_matrices[0], self.rotation_matrices[-1])

        if pos_loop or rot_loop:
            print("[info] Closing loop by appending first element to end")

            self.rotation_matrices = np.concatenate([self.rotation_matrices, self.rotation_matrices[None, 0]], axis=0)
            self.points = np.vstack([self.points, self.points[0]])
            
        Lambda_vals = np.linspace(0.0, 1.0, len(self.points), endpoint=True)

        self.positional_spine = [self._make_spline(Lambda_vals, self.points[:, i], self.spline_degree) for i in range(3)]

        quat_key = Rot.from_matrix(self.rotation_matrices).as_quat()
        for i in range(1, len(quat_key)):
            if np.dot(quat_key[i - 1], quat_key[i]) < 0.0:
                quat_key[i] = -quat_key[i]
        self.quat_key = quat_key
        
        self._make_quat_spline(Lambda_vals, self.quat_key)

        if (self.print_equation):
            print(f"\n--- Symbolic position splines (C^{self.n_continuity}) ---")
            # for name, spl in zip("xyz", (self.spline_x, self.spline_y, self.spline_z)):
            for name, spl in zip("xyz", [i for i in self.positional_spine]):
                print(f"\n{name}(Lambda) = "); sp.pretty_print(self.Print_spine(spl))
            for name, spl in zip("abcd", [i for i in self.rotaional_spine]):
                print(f"\n{name}(Lambda) = "); sp.pretty_print(self.Print_spine(spl))

    def Visualize_Path(self, L = 1.0, num_of_roation_plots = 10):
        Lambda_samp = np.linspace(0.0, 1.0, 400, endpoint= self.Is_loop)
        pos_s  = self.P(Lambda_samp)
        rot_s  = self.R(Lambda_samp)
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot(*pos_s.T, lw=1.8,
                label=rf"$C^{self.n_continuity}$ {'loop' if self.Is_loop else 'open'}")
        ax.scatter(*self.points.T, c="r", s=4)
        for i in np.linspace(0, len(Lambda_samp) - 1, num_of_roation_plots, dtype=int):
            o, Rm = pos_s[i], rot_s[i].as_matrix()
            ax.quiver(o[0], o[1], o[2], *(Rm[:, 0]), color="r", length=L, normalize=True)
            ax.quiver(o[0], o[1], o[2], *(Rm[:, 1]), color="g", length=L, normalize=True)
            ax.quiver(o[0], o[1], o[2], *(Rm[:, 2]), color="b", length=L, normalize=True)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("Position + orientation (C³ loop)")
        getattr(ax, "set_box_aspect", lambda *_: None)([1, 1, 1])

        # Quaternion plots
        Lambda_vals = np.linspace(0.0, 1.0, len(self.quat_key), endpoint=True)
        fig2, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        quat_labels = [r"$q_w(\lambda)$", r"$q_x(\lambda)$", r"$q_y(\lambda)$", r"$q_z(\lambda)$"]
        qw_k, qx_k, qy_k, qz_k = self.quat_key.T
        q_spline = np.column_stack([s(Lambda_samp) for s in self.rotaional_spine])
        q_spline /= np.linalg.norm(q_spline, axis=1, keepdims=True)  # Normalize

        for comp, ax_ in zip(range(4), axs):
            ax_.plot(Lambda_vals, (qw_k, qx_k, qy_k, qz_k)[comp], "o", label="Way-points")
            ax_.plot(Lambda_samp, q_spline[:, comp], "-", label="C³ spline")
            ax_.set_ylabel(quat_labels[comp])
            ax_.grid(True); ax_.legend()

        axs[-1].set_xlabel(r"Lambda  (curve parameter)")
        fig2.suptitle("Quaternion Components (C³ continuous)")
        plt.tight_layout(); 
        
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        xyz_labels = [r"$x(\lambda)$", r"$y(\lambda)$", r"$z(\lambda)$"]
        x, y, z = pos_s.T
        x_key, y_key, z_key = self.points.T  # waypoints

        for comp, key_pts, ax_ in zip([x, y, z], [x_key, y_key, z_key], axs3):
          ax_.plot(Lambda_vals, key_pts, "o", label="Way-points")
          ax_.plot(Lambda_samp, comp, "-", label="C³ spline")
          ax_.grid(True)
          ax_.legend()

        for label, ax_ in zip(xyz_labels, axs3):
            ax_.set_ylabel(label)

        axs3[-1].set_xlabel(r"Lambda  (curve parameter)")
        fig3.suptitle("Position Components (C³ continuous)")
        plt.tight_layout()
        plt.show()