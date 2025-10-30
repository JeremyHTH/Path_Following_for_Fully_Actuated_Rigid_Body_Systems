import numpy as np
from numpy.linalg import norm, solve
from tqdm import tqdm
from typing import List, Tuple
class Robot:
    def __init__(self, Number_of_joint, Alpha, a, d, Offset, Mass, Center_of_mass, Inertia, Fv, Fs, Gravity, 
                 joint_types=None, convention="MDH", End_effector_transformation=np.eye(4)):
        """
        Instantiate a serial manipulator model with Denavit-Hartenberg parameters.

        Parameters
        ----------
        Number_of_joint : int
            Number of actuated joints in the chain.
        Alpha, a, d, Offset : array-like
            DH or MDH parameters for each joint.
        Mass : array-like
            Mass of each link.
        Center_of_mass : array-like
            Link centre of mass expressed in the link frame.
        Inertia : array-like
            Link inertia tensors in the link frame (Ixx, Iyy, Izz, Ixy, Iyz, Izx).
        Fv : array-like
            Viscous friction coefficients.
        Fs : array-like
            Coulomb friction coefficients.
        Gravity : array-like
            Gravity vector expressed in base coordinates.
        joint_types : list of str, optional
            Sequence of "R"/"P" describing each joint; defaults to all revolute.
        convention : {"MDH","DH"}, optional
            Denavit-Hartenberg convention to use.
        End_effector_transformation : ndarray, optional
            Constant transform applied after the final joint.

        Returns
        -------
        None
        """
        self.Number_of_joint = int(Number_of_joint)
        self.Alpha = Alpha
        self.a = a
        self.d = d
        self.offset = Offset
        self.Mass = Mass
        self.Center_of_mass = Center_of_mass
        self.Inertia = Inertia
        self.Fv = Fv
        self.Fs = Fs
        self.Gravity = np.array([0.0, 0.0, -9.81])
        self._EPS = 1e-6
        self.End_effector_transformation = End_effector_transformation


        self.convention = convention.upper()
        if self.convention not in ("MDH", "DH"):
            raise ValueError("convention must be either 'MDH' or 'DH'")
        # Default to all revolute joints unless specified otherwise.
        if joint_types is None:
            joint_types = ["R"] * self.Number_of_joint
        if len(joint_types) != self.Number_of_joint:
            raise ValueError("joint_types must have length equal to Number_of_joint")
        self.joint_types = [jt.upper() for jt in joint_types]
        if any(jt not in ("R", "P") for jt in self.joint_types):
            raise ValueError("joint_types entries must be 'R' (revolute) or 'P' (prismatic)")

    def _mdh(self, alpha, a, d, theta):
        """
        Compute a modified DH transform for a single joint.

        Parameters
        ----------
        alpha, a, d, theta : float
            Modified DH parameters for the link.

        Returns
        -------
        ndarray
            4x4 homogeneous transform.
        """
        sa, ca = np.sin(alpha), np.cos(alpha)
        st, ct = np.sin(theta), np.cos(theta)
        return np.array([
            [ ct, -st, 0.0, a],
            [ st*ca, ct*ca, -sa, -d*sa],
            [ st*sa, ct*sa, ca, d*ca],
            [ 0.0, 0.0, 0.0, 1.0]
        ])

    def _dh(self, alpha, a, d, theta):
        """
        Compute a classic DH transform for a single joint.

        Parameters
        ----------
        alpha, a, d, theta : float
            DH parameters for the link.

        Returns
        -------
        ndarray
            4x4 homogeneous transform.
        """
        sa, ca = np.sin(alpha), np.cos(alpha)
        st, ct = np.sin(theta), np.cos(theta)
        return np.array([
            [ ct, -st * ca,  st * sa, a * ct],
            [ st,  ct * ca, -ct * sa, a * st],
            [ 0.0,       sa,       ca,      d],
            [ 0.0,     0.0,      0.0,    1.0]
        ])

    def _joint_transform(self, index, qi):
        """
        Compute the homogeneous transform contributed by the i-th joint.

        Parameters
        ----------
        index : int
            Joint index.
        qi : float
            Joint displacement (angle or prismatic extension).

        Returns
        -------
        ndarray
            4x4 transformation matrix for the joint.
        """
        alpha = self.Alpha[index]
        a = self.a[index]
        d = self.d[index]
        theta = self.offset[index]
        jt = self.joint_types[index]
        if jt == "R":
            theta = theta + qi
        elif jt == "P":
            d = d + qi
        else:
            raise ValueError(f"Unsupported joint type '{jt}' at index {index}")
        # Delegate to the chosen DH convention to obtain the homogeneous transform.
        if self.convention == "MDH":
            return self._mdh(alpha, a, d, theta)
        else:
            return self._dh(alpha, a, d, theta)

    def _rot_vec_from_R(self, R):
        """
        Convert a rotation matrix to its equivalent rotation vector.

        Parameters
        ----------
        R : ndarray
            3x3 rotation matrix.

        Returns
        -------
        ndarray
            3D rotation vector (axis scaled by angle).
        """
        trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(trace)
        if theta < 1e-8:
            return np.zeros(3)
        vec = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        return 0.5 * theta / np.sin(theta) * vec

    def forward_kinematics(self, q):
        """
        Compute the pose of the end-effector for given joint positions.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.

        Returns
        -------
        ndarray
            4x4 homogeneous transform of the end-effector.
        """
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.Number_of_joint:
            raise ValueError(f"Expected joint vector of length {self.Number_of_joint}, got {q.shape[0]}")
        T = np.eye(4)
        for i in range(self.Number_of_joint):
            # Multiply each joint transform to obtain the chain pose.
            T = T @ self._joint_transform(i, q[i])

        T = T @ self.End_effector_transformation
        return T
    
    def forward_kinematics_all(self, q):
        """
        Compute transforms for every joint frame along the chain.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.

        Returns
        -------
        ndarray
            Array of shape (N+2, 4, 4) containing cumulative transforms.
        """
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.Number_of_joint:
            raise ValueError(f"Expected joint vector of length {self.Number_of_joint}, got {q.shape[0]}")
        Result = np.tile(np.eye(4), (self.Number_of_joint + 2, 1, 1))
        T = np.eye(4)
        for i in range(self.Number_of_joint):
            T = T @ self._joint_transform(i, q[i])
            Result[i + 1] = T.copy()

        Result[-1] = T @ self.End_effector_transformation
        return Result

    def jacobian(self, q):
        """
        Compute the geometric Jacobian in the base frame.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.

        Returns
        -------
        ndarray
            6xN Jacobian (linear velocity stacked over angular velocity).
        """
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.Number_of_joint:
            raise ValueError(f"Expected joint vector of length {self.Number_of_joint}, got {q.shape[0]}")
        Transformations = self.forward_kinematics_all(q)

        if (self.convention == "MDH"):
            index_offset = 1
        else:
            index_offset = 0

        origins = [Transformations[i + index_offset][:3, 3] for i in range(self.Number_of_joint)]
        axes = [Transformations[i + index_offset][:3, 2] for i in range(self.Number_of_joint)]

        p_e = Transformations[-1][:3, 3]
        # Spatial Jacobian: top rows linear velocity, bottom rows angular.
        Jv = np.zeros((3, self.Number_of_joint))
        Jw = np.zeros((3, self.Number_of_joint))
        for i in range(self.Number_of_joint):
            axis = axes[i]
            origin = origins[i]
            if self.joint_types[i] == "R":
                Jw[:, i] = axis
                Jv[:, i] = np.cross(axis, p_e - origin)
            else:  # prismatic joint
                Jw[:, i] = 0.0
                Jv[:, i] = axis
        return np.vstack((Jv, Jw))

    def jacobian_dot(self, q, qd):
        """
        Numerically differentiate the Jacobian with respect to time.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.
        qd : ndarray
            Joint velocity vector.

        Returns
        -------
        ndarray
            6xN matrix representing J̇.
        """
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        if q.shape[0] != self.Number_of_joint or qd.shape[0] != self.Number_of_joint:
            raise ValueError(f"Expected joint vectors of length {self.Number_of_joint}")
        J_dot = np.zeros((6, self.Number_of_joint))

        for i in range(self.Number_of_joint):
            # Approximate ∂J/∂q_i numerically and accumulate the time derivative.
            dq = np.zeros(self.Number_of_joint)
            dq[i] = self._EPS
            J_plus = self.jacobian(q + dq)
            J_minus = self.jacobian(q - dq)
            dJ_dqi = (J_plus - J_minus) / (2 * self._EPS)
            J_dot += dJ_dqi * qd[i]

        return J_dot


    def _link_frames(self, q):
        """
        Gather per-link kinematics (pose and Jacobians) for dynamics calculations.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.

        Returns
        -------
        tuple
            Lists of rotation matrices, positions, centre-of-mass positions,
            translational Jacobians, and rotational Jacobians for each link.
        """
        q = np.asarray(q, dtype=float)
        if q.shape[0] != self.Number_of_joint:
            raise ValueError(f"Expected joint vector of length {self.Number_of_joint}, got {q.shape[0]}")
        Transformations = self.forward_kinematics_all(q)

        if self.convention == "MDH":
            index_offset = 1
        else:
            index_offset = 0

        origins = [Transformations[i + index_offset][:3, 3] for i in range(self.Number_of_joint)]
        axes = [Transformations[i + index_offset][:3, 2] for i in range(self.Number_of_joint)]

        R_out, p_out, pc_out, Jv_out, Jw_out = [], [], [], [], []
        for i in range(self.Number_of_joint):
            T_link = Transformations[i + 1]
            R_i = T_link[:3, :3]
            p_i = T_link[:3, 3]
            p_com = p_i + R_i @ self.Center_of_mass[i]
            Jv_i = np.zeros((3, self.Number_of_joint))
            Jw_i = np.zeros((3, self.Number_of_joint))
            for j in range(i + 1):
                axis = axes[j]
                origin = origins[j]
                if self.joint_types[j] == "R":
                    Jw_i[:, j] = axis
                    Jv_i[:, j] = np.cross(axis, p_com - origin)
                else:
                    Jw_i[:, j] = 0.0
                    Jv_i[:, j] = axis
            R_out.append(R_i)
            p_out.append(p_i)
            pc_out.append(p_com)
            Jv_out.append(Jv_i)
            Jw_out.append(Jw_i)
        return R_out, p_out, pc_out, Jv_out, Jw_out

    def _inertia_tensor(self, i):
        """
        Build the inertia tensor matrix for the i-th link.

        Parameters
        ----------
        i : int
            Link index.

        Returns
        -------
        ndarray
            3x3 inertia tensor in the link frame.
        """
        Ixx, Iyy, Izz, Ixy, Iyz, Izx = self.Inertia[i]
        return np.array([[Ixx, Ixy, Izx], [Ixy, Iyy, Iyz], [Izx, Iyz, Izz]])

    def _mass_matrix(self, q):
        # Standard rigid-body inertia: project link inertia and COM terms through each joint Jacobian.
        R, _, _, Jv, Jw = self._link_frames(q)
        M = np.zeros((self.Number_of_joint, self.Number_of_joint))
        for i in range(self.Number_of_joint):
            m_i = self.Mass[i]
            I_i = R[i] @ self._inertia_tensor(i) @ R[i].T
            M += m_i * (Jv[i].T @ Jv[i]) + Jw[i].T @ I_i @ Jw[i]
        return M

    def _gravity_vector(self, q):
        """
        Assemble the joint-space gravity torque vector.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.

        Returns
        -------
        ndarray
            Gravity-induced torques for each joint.
        """
        _, _, pc, Jv, _ = self._link_frames(q)
        g_vec = np.zeros(self.Number_of_joint)
        for i in range(self.Number_of_joint):
            # Project link weights through the translational Jacobian.
            g_vec += Jv[i].T @ (self.Mass[i] * (-self.Gravity))
        return g_vec

    def _friction_split(self, qd):
        """
        Separate viscous and Coulomb friction contributions.

        Parameters
        ----------
        qd : ndarray
            Joint velocity vector.

        Returns
        -------
        tuple
            Diagonal viscous matrix and Coulomb matrix (with sign(qd)).
        """
        F_v = np.diag(self.Fv)
        F_c = np.diag(self.Fs * np.sign(qd))
        return F_v, F_c

    def _coriolis_matrix(self, q, qd):
        """ Compute the Coriolis matrix C(q, q̇) such that c = C(q, q̇) q̇ """
        n = self.Number_of_joint
        M0 = self._mass_matrix(q)
        dM = np.empty((n, n, n))
        for k in range(n):
            # Central differences give ∂M/∂q_k needed for the Christoffel symbols.
            dqk = np.zeros(n); dqk[k] = self._EPS
            Mk_plus = self._mass_matrix(q + dqk)
            Mk_minus = self._mass_matrix(q - dqk)
            dM[k] = (Mk_plus - Mk_minus) / (2.0 * self._EPS)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i, j] += 0.5 * (dM[k, i, j] + dM[j, i, k] - dM[i, j, k]) * qd[k]
        return C

    def dynamics(self, q, qd):
        """
        Compute the manipulator mass matrix, Coriolis matrix, friction, and gravity.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.
        qd : ndarray
            Joint velocity vector.

        Returns
        -------
        tuple
            `(M, C, F_v, F_c, G)` inertia matrix, Coriolis matrix, viscous friction,
            Coulomb friction, and gravity torques.
        """
        M = self._mass_matrix(q)
        C = self._coriolis_matrix(q, qd)
        G = self._gravity_vector(q)
        F_v, F_c = self._friction_split(qd)
        return M, C, F_v, F_c, G

    def forward_dynamics(self, q, qd, tau):
        """
        Evaluate the joint accelerations arising from applied torques.

        Parameters
        ----------
        q : ndarray
            Joint configuration vector.
        qd : ndarray
            Joint velocity vector.
        tau : ndarray
            Applied joint torques/forces.

        Returns
        -------
        ndarray
            Joint accelerations solving M(q) q̈ = τ - C q̇ - friction - G.
        """
        M, C, Fv, Fc, G = self.dynamics(q, qd)
        return np.linalg.solve(M, tau - C @ qd - Fv @ qd - Fc @ np.sign(qd) - G) 
