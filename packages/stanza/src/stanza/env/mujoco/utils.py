import jax

def _quat_to_angle(quat):
    w0 = quat[0] # cos(theta/2)
    w3 = quat[3] # sin(theta/2)
    angle = 2*jax.numpy.atan2(w3, w0)
    return angle