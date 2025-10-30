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

    def __init__(self, points, rotation_matrices, Is_loop, n_continuity, print_equation=False , _samples: int = 4000, frame_method = "parallel", Frame_log = False):
        """
        Extend the base path with arc-length parametrisation and moving frames.

        Parameters
        ----------
        points : array-like
            Waypoints to interpolate.
        rotation_matrices : array-like
            Orientation samples aligned with each waypoint.
        Is_loop : bool
            Whether the path is periodic.
        n_continuity : int
            Continuity level of the underlying spline.
        print_equation : bool, optional
            If True, print symbolic spline details.
        _samples : int, optional
            Number of samples used to build the arc-length map.
        frame_method : str, optional
            Frame construction strategy (currently only "parallel").
        Frame_log : bool, optional
            If True, dump frame data to CSV during visualisation.

        Returns
        -------
        None
        """
        super().__init__(points, rotation_matrices, Is_loop, n_continuity, print_equation)
        self._samples = _samples
        # Build arc-length parametrisation right away so later queries can use metres.
        self._build_arc_length_map(self._samples)
        self.frame_method = frame_method
        self._frame_cache: dict[str, np.ndarray] = {}
        self.Frame_log = Frame_log

        self._make_frame()


    def _build_arc_length_map(self, N: int):
        """
        Generate spline maps between curve parameter and arc-length.

        Parameters
        ----------
        N : int
            Number of uniform parameter samples used to approximate arc-length.

        Returns
        -------
        None
        """
        # lam = np.linspace(0.0, 1.0, N, endpoint=not self.Is_loop)
        lam = np.linspace(0.0, 1.0, N)
        P = super().P(lam)
        seglen = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seglen)))
        self.total_length = s[-1]

        self._lam_of_s = make_interp_spline(s, lam, k = 3)
        self._s_of_lam = make_interp_spline(lam, s, k = 3)

    def Ensure_in_range(self, s):
        """
        Wrap or expose the curve parameter depending on whether the path loops.

        Parameters
        ----------
        s : array-like or float
            Arc-length value(s) to adjust.

        Returns
        -------
        array-like or float
            Wrapped or unclamped arc-length values.
        """
        if self.Is_loop:
            return (s + self.total_length) % self.total_length
        else:
            # Allow Newton iterations to overshoot slightly on open paths;
            # callers clamp the result when necessary.
            return s
            # return np.clip(s, 0, self.total_length)

    def P(self, s):
        """
        Evaluate the positional spline using arc-length input.

        Parameters
        ----------
        s : array-like or float
            Arc-length value(s).

        Returns
        -------
        ndarray
            Cartesian coordinates at the requested arc-length.
        """
        s = self.Ensure_in_range(s)

        lam = self._lam_of_s(s)
        return super().P(lam)

    def R(self, s):
        """
        Evaluate the rotational spline using arc-length input.

        Parameters
        ----------
        s : array-like or float
            Arc-length value(s).

        Returns
        -------
        Rotation
            SciPy Rotation describing orientation along the path.
        """
        s = self.Ensure_in_range(s)
        lam = self._lam_of_s(s)
        return super().R(lam)


    def _make_frame(self):
        """
        Dispatch to the configured frame construction routine.

        Returns
        -------
        None
        """
        if (self.frame_method == "parallel"):
            self._make_parallel_transport_frame()
        else:
            raise Exception("Frame method not found")

    def _make_parallel_transport_frame(self):
        """
        Build parallel-transport frames along the path for curvature-aware control.

        Returns
        -------
        None
        """
        s = np.linspace(0, self.total_length, 20000, endpoint=not self.Is_loop)


        T = self._dP_ds(s, 1)
        M1 = np.zeros_like(T)
        M2 = np.zeros_like(T)

        # Start the transport with an arbitrary axis that is not collinear
        # with the initial tangent to avoid a zero cross product.
        Initial_guess_reference = np.array([0, 0, 1]) if abs(T[0][2]) < 0.9 else np.array([0, 1, 0])

        M1_0 = np.cross(T[0], Initial_guess_reference)
        M1[0] = M1_0 / np.linalg.norm(M1_0)

        M2[0] = np.cross(T[0], M1[0])

        for i in range(1, len(s)):
            # Project the previous normal into the new normal plane and rebuild the binormal.
            proj = M1[i - 1] - np.dot(M1[i - 1], T[i]) * T[i]
            M1[i] = proj / np.linalg.norm(proj)
            M2[i] = np.cross(T[i], M1[i])

        self._frame_cache = {"s": s, "T": T, "M1": M1, "M2": M2}

    def _eval_dP_dLambda(self, lam, order):
        """
        Evaluate derivatives of the positional spline with respect to λ.

        Parameters
        ----------
        lam : array-like
            Normalised curve parameters.
        order : int
            Derivative order to evaluate.

        Returns
        -------
        ndarray
            Derivative vectors for each requested parameter.
        """
        lam = np.asarray(lam, dtype=float)
        return np.column_stack(
            [spl.derivative(order)(lam) for spl in self.positional_spine]
        )

    def _dP_ds(self, s, order=1):
        """
        Evaluate derivatives of the positional spline with respect to arc-length.

        Parameters
        ----------
        s : array-like
            Arc-length positions.
        order : int, optional
            Derivative order (1 for tangent, 2 for curvature vector).

        Returns
        -------
        ndarray
            Tangent or curvature vectors depending on the requested order.
        """
        s   = self.Ensure_in_range(s)
        lam = self._lam_of_s(s)

        if order == 1:
            P_l = self._eval_dP_dLambda(lam, 1)

            norms = np.linalg.norm(P_l, axis=1, keepdims=True)
            T = np.divide(P_l, norms, out=np.full_like(P_l, [1.0, 0.0, 0.0]), where=(norms > 1e-8))

            return T

        elif order == 2:
            P_l  = self._eval_dP_dLambda(lam, 1)
            P_ll = self._eval_dP_dLambda(lam, 2)
            ds_dλ = np.linalg.norm(P_l, axis=1, keepdims=True)

            epsilon = 1e-8
            safe_denom = np.where(ds_dλ**2 > epsilon, ds_dλ**2, epsilon)
            projection = np.sum(P_ll * P_l, axis=1, keepdims=True) * P_l / safe_denom
            P_ss = (P_ll - projection) / safe_denom

            return P_ss
        else:
            raise ValueError("order must be 1 or 2")

    def get_frame(self, s):
        """
        Retrieve the tangent and normal vectors at specified arc-length positions.

        Parameters
        ----------
        s : array-like or float
            Arc-length value(s) where the frame should be evaluated.

        Returns
        -------
        tuple
            Tangent `T`, first normal `M1`, and second normal `M2`. Shapes follow
            the input (`(3,)` for scalar arc-length or `(..., 3)` for arrays).
        """
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

        # Remove any component of the interpolated normal that drifted into the tangent direction.
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

    def Visualize_frame(self, L = 1, num_of_frame_plots = 10, Show_plots = True):
        
        """
        Visualise the path together with sampled moving frames.

        Parameters
        ----------
        L : float, optional
            Vector length for frame glyphs.
        num_of_frame_plots : int, optional
            Number of frames to draw.
        Show_plots : bool, optional
            Whether to display the plot immediately.

        Returns
        -------
        None
        """
        s = np.linspace(0.0,self.total_length, 400, endpoint=not self.Is_loop)
        pos_s = self.P(s)
        if (self.frame_method == "parallel"):
            labels = ("T", "$M_1$", "$M_2$")

        T, A1, A2 = self.get_frame(s)
        frame_vectors = np.hstack((T, A1, A2))

        if (self.Frame_log):
            output_path = Path.cwd() / "frame_vectors.csv"
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

        if (Show_plots):
            plt.show()
    
    def num_derivative(self, func, x, h=1e-6):
        """
        Compute a centred finite-difference derivative of a scalar function.

        Parameters
        ----------
        func : callable
            Function of a single scalar variable.
        x : float
            Point at which to evaluate the derivative.
        h : float, optional
            Finite-difference step size.

        Returns
        -------
        float
            Approximated first derivative at `x`.
        """
        return (func(x + h) - func(x - h)) / (2.0 * h)
    
    def find_s_star(self, y, last_s = None, n_init=10, tol=1e-6, n_max=50):
        """
        Locate the closest point on the path to an external query position.

        Parameters
        ----------
        y : array-like
            Cartesian position to project onto the path.
        last_s : float or None, optional
            Warm-start arc-length guess; if None, use multiple seeds.
        n_init : int, optional
            Number of uniformly spaced initial guesses when `last_s` is None.
        tol : float, optional
            Convergence tolerance for the Newton iterations (unused but kept for API).
        n_max : int, optional
            Maximum number of Newton iterations per seed.

        Returns
        -------
        tuple
            `(best_s, closest_point)` where `best_s` is the arc-length of the closest
            path point and `closest_point` its Cartesian coordinates.
        """
        y = np.asarray(y, dtype=float)
        s_bound = (0.0, self.total_length)
        if (last_s == None):
            s_grid = np.linspace(0, self.total_length, n_init)
        else: 
            s_grid = [last_s]

        # Run a small multi-start Newton search in arc-length space.
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

        else: 
            # Permit a small buffer so numerical differentiation in downstream code
            # can still sample just outside the physical interval.
            best_s = np.clip(best_s, s_bound[0] - self.total_length * 0.1, s_bound[1] + self.total_length * 0.1) 

        # if (best_s < 0):
        #     raise Exception(f"negative s {best_s_before}{best_s}, {s_bound[0]} {s_bound[1]}")

        return best_s, self.P(best_s)
    
    def Visualize_closest_point(self, query_pts = [[0, 0, 0]], Show_plots = True):

        """
        Display closest-point projections for a collection of queries.

        Parameters
        ----------
        query_pts : list of array-like, optional
            Positions to project onto the path.
        Show_plots : bool, optional
            Whether to display the figure immediately.

        Returns
        -------
        None
        """
        s = np.linspace(0.0,self.total_length, 400, endpoint=not self.Is_loop)
        pos_s = self.P(s)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*pos_s.T, lw=1.8)

        for q in query_pts:
            s_star, p_star = self.find_s_star(q)
            p_star = np.ravel(p_star)
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
        if (Show_plots):
            plt.show()
