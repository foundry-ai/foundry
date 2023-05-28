from stanza import partial, Partial
import jax.numpy as jnp
import jax

def sanitize_color(color):
    # if alpha is not specified, make alpha = 1
    if color.shape[0] == 3:
        return jnp.concatenate((color, jnp.ones((1,))), axis=0)
    return color

def _aa_alpha(dist):
    return jnp.clip(0.1 - dist, 0, 1)

@jax.jit
def _aa(dist, color):
    alpha = _aa_alpha(dist)
    # modify the color alpha
    color = color.at[3].set(color[3]*alpha)
    return color

# Does an anti-aliases
def paint(canvas, sdf):
    coords_y = jnp.linspace(0, canvas.shape[0], num=canvas.shape[0])
    coords_x = jnp.linspace(0, canvas.shape[1], num=canvas.shape[1])
    grid_x, grid_y = jnp.meshgrid(coords_x, coords_y)
    grid = jnp.stack((grid_x, grid_y), axis=-1)
    dist, colors = jax.vmap(jax.vmap(sdf))(grid)
    colors = jax.vmap(jax.vmap(_aa))(dist, colors)
    alpha = jnp.expand_dims(colors[:,:,3],-1)
    if canvas.shape[-1] == 3:
        colors = colors[:,:,:3]
    canvas = alpha*colors + (1 - alpha)*canvas
    return canvas

@jax.jit
def _stacked(x, sdfs):
    d, c = sdfs[0](x)
    for s in sdfs:
        dist, cp = s(x)
        alpha = _aa_alpha(dist)
        c = alpha*cp + (1 - alpha)*c
        d = jnp.minimum(d, dist)
    return d, c

def stack(*sdfs):
    return partial(_stacked, sdfs=sdfs)

@jax.jit
def _transformed(p, fn, loc=None, theta=None, scale=None):
    if loc is not None:
        p = p - loc
    if scale is not None:
        p = p / scale
    if theta is not None:
        c, s = np.cos(-theta), np.sin(-theta)
        M = jnp.array(((c,-s),(s,c)))
        p = M @ p
    return fn(p)

def transform(fn, loc=None, theta=None, scale=None):
    return Partial(partial(_transformed, fn=fn),loc=loc, theta=theta, scale=scale)

@jax.jit
def _circle(x, loc, radius=1, color=None):
    dist = jnp.linalg.norm(x - loc) - radius
    return dist, color

def circle(loc, radius=1, color=jnp.array([0.,0.,0.,1.])):
    color = sanitize_color(color)
    return Partial(_circle, loc=loc, radius=radius, color=color)

@jax.jit
def _line(x, a, b, thickness, color=None):
    xa = x - a
    ba = b - a
    h = jnp.clip(jnp.dot(xa, ba) / jnp.dot(ba, ba), 0., 1.)
    dist = jnp.linalg.norm(xa - h * ba)
    return dist - 1, color

def line(a, b, thickness=1., color=jnp.array([0.,0.,0.,1.])):
    color = sanitize_color(color)
    return Partial(_line, a=a, b=b, thickness=thickness, color=color)