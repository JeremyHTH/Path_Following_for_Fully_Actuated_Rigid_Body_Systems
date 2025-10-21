import numpy as np

def Angular_error(theta, theta_des, Frame):
    if (Frame.Is_loop):
        Result = None
        error_1 = theta_des - theta
        error_2 = (error_1 + Frame.total_length) % (Frame.total_length)
        if (np.abs(error_1) < np.abs(error_2)):
            Result = error_1
        else:
            Result = error_2
        return Result
    else:
        return theta_des - theta

def skew(v):
    return np.array([[ 0,   -v[2],  v[1]],
                     [ v[2], 0,   -v[0]],
                     [-v[1], v[0],  0]])

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])


def log_SO3(R):
    tr = np.trace(R)
    tr = np.clip(tr, -1.0, 3.0)  
    phi = np.arccos((tr - 1) / 2)

    if phi < 1e-6:
        return np.zeros(3)
    else:
        v = np.array([
            R[2,1] - R[1,2],
            R[0,2] - R[2,0],
            R[1,0] - R[0,1]
        ]) / (2 * np.sin(phi))
        return phi * v

def hat(v):
    return np.array([[    0, -v[2],  v[1]],
                    [ v[2],     0, -v[0]],
                    [-v[1],  v[0],    0]])