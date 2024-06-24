import jax

def _quat_to_angle(quat):
    w0 = quat[0] # cos(theta/2)
    w3 = quat[3] # sin(theta/2)
    angle = 2*jax.numpy.atan2(w3, w0)
    return angle

def brax_to_state(sys, data):
    import brax.mjx.pipeline as pipeline
    from brax.base import Contact, Motion, System, Transform
    q, qd = data.qpos, data.qvel
    x = Transform(pos=data.xpos[1:], rot=data.xquat[1:])
    cvel = Motion(vel=data.cvel[1:, 3:], ang=data.cvel[1:, :3])
    offset = data.xpos[1:, :] - data.subtree_com[sys.body_rootid[1:]]
    offset = Transform.create(pos=offset)
    xd = offset.vmap().do(cvel)
    data = pipeline._reformat_contact(sys, data)
    return pipeline.State(q=q, qd=qd, x=x, xd=xd, **data.__dict__)

def brax_render(mj_model, data_seq):
    import brax
    import brax.io.mjcf
    import brax.io.html
    sys = brax.io.mjcf.load_model(mj_model)
    T = data_seq.xpos.shape[0]
    states = jax.vmap(brax_to_state, in_axes=(None, 0))(sys, data_seq)
    states = [jax.tree_map(lambda x: x[i], states) for i in range(T)]
    return brax.io.html.render(sys, states)