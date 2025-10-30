from __future__ import annotations
import numpy as np
import sympy as sp
from scipy.interpolate import make_interp_spline, PPoly
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   

class Path_Generation_Tool:

    def __init__(self, points, rotation_matrices, Is_loop, n_continuity, print_equation=False):
        """
        Construct the spline representation of a geometric path.

        Parameters
        ----------
        points : array-like
            Waypoints describing the Cartesian curve.
        rotation_matrices : array-like
            Orientation samples aligned with each waypoint.
        Is_loop : bool
            Flag indicating whether the path is periodic.
        n_continuity : int
            Desired continuity order (C^n) for the spline.
        print_equation : bool, optional
            If True, print symbolic spline expressions after construction.

        Returns
        -------
        None
        """
        self.points     = np.array(points, dtype=float)
        self.rotation_matrices   = np.array(rotation_matrices, dtype=float)  
        self.Is_loop    = bool(Is_loop)

        if (not self.Is_loop and n_continuity % 2):
            # Odd degrees break the open-spline boundary conditions, so promote
            # the requested continuity to the next admissible even value.
            n_continuity += 1
            print(f"[Info] Adjusted n_continuity to be even ({n_continuity}) from ({n_continuity - 1}) for open spline to ensure generated correct gradient at endpoints.")

        self.n_continuity = n_continuity
        self.spline_degree = n_continuity + 1
        self.print_equation = print_equation

        if self.rotation_matrices.ndim != 3 or self.rotation_matrices.shape[1:] != (3, 3):
            raise ValueError("rotation_matrices must be of shape (N, 3, 3)")

        self.positional_spine = None
        self.rotaional_spine = None

        self.total_length = 1.0

        self.Generate_Path()

        
    def _auto_bc(self, k: int):
        """
        Select the boundary condition string for open splines.

        Parameters
        ----------
        k : int
            Polynomial degree of the spline segment.

        Returns
        -------
        str
            Boundary condition identifier understood by SciPy.

        Raises
        ------
        Exception
            If an unsupported even-degree spline is requested.
        """
        if k % 2:                                  
            return "not-a-knot"
        else:
            raise Exception("Even degree splines is not supported for open splines to ensure correct gradient.")
        
    def _make_spline(self, x, y, k: int):
        """
        Build a 1D spline interpolant for a path component.

        Parameters
        ----------
        x : array-like
            Normalised curve parameter samples.
        y : array-like
            Coordinate values associated with each sample.
        k : int
            Desired spline degree.

        Returns
        -------
        BSpline
            SciPy spline object interpolating the provided data.
        """
        if self.Is_loop:
            if len(x) <= k:
                raise ValueError(f"periodic spline: need > k={k} points")
            return make_interp_spline(x, y, k=k, bc_type="periodic")
        if len(x) < k + 1:
            raise ValueError(f"open spline: need ≥ k+1={k+1} points")
        # Open splines fall back to SciPy's not-a-knot boundary, configured via _auto_bc.
        return make_interp_spline(x, y, k=k, bc_type=self._auto_bc(k))

    def _make_quat_spline(self, lam_key, quat_key, k):  
        """
        Create four scalar splines whose normalised outputs represent orientations.

        Parameters
        ----------
        lam_key : array-like
            Parameter samples associated with the quaternion keys.
        quat_key : array-like
            Quaternion waypoints (one per parameter sample).
        k : int
            Spline degree to use for each quaternion component.

        Returns
        -------
        None
        """
        n = len(lam_key)
        quat_key = quat_key.copy().astype(float)

        for i in range(1, n):
            # Enforce quaternion sign consistency to avoid discontinuous jumps
            # when neighbouring samples lie on opposite hemispheres.
            if np.dot(quat_key[i - 1], quat_key[i]) < 0.0:
                quat_key[i] = -quat_key[i]

        if self.Is_loop:
            if not np.allclose(quat_key[0], quat_key[-1]):
                print("[Info] Closing quaternion loop: appending first quaternion to end")
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
        # Closed paths loop in parameter space; open paths allow extrapolation so
        # that higher-level code can decide how to clamp or extend the spline.
        return np.mod(Lambda, 1.0) if self.Is_loop else Lambda
    
    def P(self, Lambda):    
        """
        Evaluate the positional spline at one or more parameter values.

        Parameters
        ----------
        Lambda : array-like or float
            Curve parameter(s) to evaluate.

        Returns
        -------
        ndarray
            Cartesian coordinates corresponding to the supplied parameters.
        """
        Wrapped_Lambda = self._Lambda_wrap(Lambda)
        return np.column_stack([p(Wrapped_Lambda) for p in self.positional_spine])
    
    def R(self, Lambda):
        """
        Evaluate the rotational spline at one or more parameter values.

        Parameters
        ----------
        Lambda : array-like or float
            Curve parameter(s) to evaluate.

        Returns
        -------
        Rotation
            SciPy Rotation object encapsulating the orientations.
        """
        Wrapped_Lambda = self._Lambda_wrap(Lambda)
        q = np.column_stack([s(Wrapped_Lambda) for s in self.rotaional_spine])
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        return Rot.from_quat(q)

    def Rotation_vector(self, Lambda):     
        """
        Return rotation vectors corresponding to the spline orientations.

        Parameters
        ----------
        Lambda : array-like or float
            Curve parameter(s) to evaluate.

        Returns
        -------
        ndarray
            Rotation vectors expressed in axis-angle form.
        """
        Wrapped_Lambda = self._Lambda_wrap(Lambda)      
        return self.R(Wrapped_Lambda).as_rotvec()
    
    def Print_spine(self, spl):
        """
        Convert a spline into a SymPy piecewise polynomial for inspection.

        Parameters
        ----------
        spl : BSpline
            SciPy spline instance to convert.

        Returns
        -------
        sympy.Piecewise
            Symbolic representation of the spline.
        """
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
        """
        Assemble the positional and rotational spline data structures.

        Returns
        -------
        None
        """

        if len(self.points) != len(self.rotation_matrices): 
            print("Length of points and angle does not match")
            return 

        pos_loop = self.Is_loop and not np.allclose(self.points[0], self.points[-1])
        rot_loop = self.Is_loop and not np.allclose(self.rotation_matrices[0], self.rotation_matrices[-1])

        if pos_loop or rot_loop:
            # Duplicate the first waypoint/orientation so periodic splines stay C^n.
            print("[Info] Closing loop by appending first element to end")

            self.rotation_matrices = np.concatenate([self.rotation_matrices, self.rotation_matrices[None, 0]], axis=0)
            self.points = np.vstack([self.points, self.points[0]])
            
        Lambda_vals = np.linspace(0.0, 1.0, len(self.points), endpoint=True)

        self.positional_spine = [self._make_spline(Lambda_vals, self.points[:, i], self.spline_degree) for i in range(3)]

        quat_key = Rot.from_matrix(self.rotation_matrices).as_quat()
        for i in range(1, len(quat_key)):
            if np.dot(quat_key[i - 1], quat_key[i]) < 0.0:
                quat_key[i] = -quat_key[i]
        self.quat_key = quat_key
        
        self._make_quat_spline(Lambda_vals, self.quat_key, self.spline_degree)

        if (self.print_equation):
            print(f"\n--- Symbolic position splines (C^{self.n_continuity}) ---")
            for name, spl in zip("xyz", [i for i in self.positional_spine]):
                print(f"\n{name}(Lambda) = "); sp.pretty_print(self.Print_spine(spl))
            for name, spl in zip("abcd", [i for i in self.rotaional_spine]):
                print(f"\n{name}(Lambda) = "); sp.pretty_print(self.Print_spine(spl))

    def Visualize_path(self, L = 1.0, num_of_roation_plots = 10, Show_plots = True):
        """
        Plot the interpolated path along with sampled orientations and spline traces.

        Parameters
        ----------
        L : float, optional
            Length of the orientation triad arrows.
        num_of_roation_plots : int, optional
            Number of orientation glyphs to draw along the path.
        Show_plots : bool, optional
            If True, display the figures immediately; otherwise defer to caller.

        Returns
        -------
        None
        """
        # Sample slightly outside the nominal range so the plot shows how open paths extrapolate.
        Input_sample = np.linspace(0.0 - 10, self.total_length - 10, 400, endpoint= self.Is_loop)
        pos_s  = self.P(Input_sample)
        rot_s  = self.R(Input_sample)
        fig = plt.figure()
        ax  = fig.add_subplot(111, projection="3d")
        ax.plot(*pos_s.T, lw=1.8,
                label=rf"$C^{self.n_continuity}$ {'loop' if self.Is_loop else 'open'}")
        ax.scatter(*self.points.T, c="r", s=4)
        for i in np.linspace(0, len(Input_sample) - 1, num_of_roation_plots, dtype=int):
            o, Rm = pos_s[i], rot_s[i].as_matrix()
            ax.quiver(o[0], o[1], o[2], *(Rm[:, 0]), color="r", length=L, normalize=True)
            ax.quiver(o[0], o[1], o[2], *(Rm[:, 1]), color="g", length=L, normalize=True)
            ax.quiver(o[0], o[1], o[2], *(Rm[:, 2]), color="b", length=L, normalize=True)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("Position and orientation along generated path")
        getattr(ax, "set_box_aspect", lambda *_: None)([1, 1, 1])

        # Quaternion plots
        Lambda_vals = np.linspace(0.0, 1.0, len(self.quat_key), endpoint=True)
        fig2, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        quat_labels = [r"$q_w(\lambda)$", r"$q_x(\lambda)$", r"$q_y(\lambda)$", r"$q_z(\lambda)$"]
        qw_k, qx_k, qy_k, qz_k = self.quat_key.T
        q_spline = np.column_stack([s(Lambda_vals) for s in self.rotaional_spine])
        q_spline /= np.linalg.norm(q_spline, axis=1, keepdims=True)  # Normalize

        for comp, ax_ in zip(range(4), axs):
            ax_.plot(Lambda_vals, (qw_k, qx_k, qy_k, qz_k)[comp], "o", label="Way-points")
            ax_.plot(Lambda_vals, q_spline[:, comp], "-", label="C³ spline")
            ax_.set_ylabel(quat_labels[comp])
            ax_.grid(True); ax_.legend()

        axs[-1].set_xlabel(r"Lambda  (curve parameter)")
        fig2.suptitle("Quaternion Components (C³ continuous)")
        plt.tight_layout(); 
        
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        xyz_labels = [r"$x(\lambda)$", r"$y(\lambda)$", r"$z(\lambda)$"]
        x_key, y_key, z_key = self.points.T  # waypoints
        pos_spline = np.column_stack([p(Lambda_vals) for p in self.positional_spine])

        for comp, ax_ in zip(range(3), axs3):
            ax_.plot(Lambda_vals, (x_key, y_key, z_key)[comp], "o", label="Way-points")
            ax_.plot(Lambda_vals, pos_spline[:, comp], "-", label="C³ spline")
            ax_.set_ylabel(xyz_labels[comp])
            ax_.grid(True); ax_.legend()

        axs3[-1].set_xlabel(r"Lambda  (curve parameter)")
        fig3.suptitle("Position Components (C³ continuous)")
        plt.tight_layout()

        # xyz_labels = [r"$x(\lambda)$", r"$y(\lambda)$", r"$z(\lambda)$"]
        # x, y, z = pos_s.T
        # x_key, y_key, z_key = self.points.T  # waypoints

        # for comp, key_pts, ax_ in zip([x, y, z], [x_key, y_key, z_key], axs3):
        #     ax_.plot(Lambda_vals, key_pts, "o", label="Way-points")
        #     ax_.plot(Input_sample, comp, "-", label="C³ spline")
        #     ax_.grid(True)
        #     ax_.legend()

        # for label, ax_ in zip(xyz_labels, axs3):
        #     ax_.set_ylabel(label)

        # axs3[-1].set_xlabel(r"Lambda  (curve parameter)")
        # fig3.suptitle("Position Components (C³ continuous)")
        # plt.tight_layout()

        if (Show_plots):
            plt.show()
