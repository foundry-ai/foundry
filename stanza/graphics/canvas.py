from stanza import partial, Partial
from stanza.dataclasses import dataclass
import jax.numpy as jnp
import jax

from typing import List

class Geometry:
    @property
    def aabb(self):
        raise NotImplementedError()

    def signed_distance(self, x):
        raise NotImplementedError()

@dataclass(jax=True)
class Box:
    top_left: jnp.ndarray
    bottom_right: jnp.ndarray

    @property
    def aabb(self):
        return self
    
    def signed_distance(self, x):
        left = -(x[0] - self.top_left[0])
        right = x[0] - self.bottom_right[0]
        top = -(x[1] - self.top_left[1])
        bot = x[1] - self.bottom_right[1]
        hor = jnp.maximum(left, right)
        ver = jnp.maximum(top, bot)
        return jnp.maximum(hor, ver)
    
    def intersects(self, other):
        return jnp.logical_not(
            jnp.logical_or(
                self.top_left - other.bottom_right >= 0,
                self.bottom_right - other.top_left <= 0
            )
        )

def box(top_left, bottom_right):
    return Box(
        top_left=jnp.array(top_left),
        bottom_right=jnp.array(bottom_right)
    )
rectangle = box

@dataclass(jax=True)
class Polygon:
    vertices: jnp.ndarray

    @property
    def aabb(self):
        return Box(
            top_left=jnp.min(self.vertices, axis=0),
            bottom_right=jnp.max(self.vertices, axis=0)
        )

    def signed_distance(self, x):
        edges = jnp.roll(self.vertices,1,axis=0) - self.vertices
        rot_edges = jnp.stack(
            (-edges[:,1], edges[:,0]), 
            axis=-1
        )
        rot_edges = rot_edges / jnp.linalg.norm(rot_edges, axis=-1, keepdims=True)
        corner_dists = jax.vmap(jnp.dot)(rot_edges, self.vertices)
        dists = rot_edges @ x - corner_dists
        return jnp.max(dists)

def polygon(vertices):
    return Polygon(jnp.array(vertices))

@dataclass(jax=True)
class Circle:
    center: jnp.ndarray
    radius: float

    @property
    def aabb(self):
        return Box(
            top_left=self.center - self.radius,
            bottom_right=self.center + self.radius
        )

    def signed_distance(self, x):
        return jnp.linalg.norm(x - self.center) - self.radius

def circle(center, radius):
    return Circle(jnp.array(center), radius)

@dataclass(jax=True)
class Segment:
    a: jnp.ndarray
    b: jnp.ndarray
    thickness: float

    @property
    def aabb(self):
        return Box(
            top_left=jnp.minimum(self.a, self.b) - self.thickness,
            bottom_right=jnp.maximum(self.a, self.b) + self.thickness
        )

    def signed_distance(self, x):
        xa = x - self.a
        ba = self.b - self.a
        h = jnp.clip(jnp.dot(xa, ba) / jnp.dot(ba, ba), 0., 1.)
        dist = jnp.linalg.norm(xa - h * ba)
        return dist - self.thickness

def segment(a, b, thickness=1.):
    return Segment(jnp.array(a), jnp.array(b), thickness)

# Composed geometries

@dataclass(jax=True)
class Union:
    geometries: List[Geometry]

    @property
    def aabb(self):
        top_lefts = jnp.stack([g.aabb.top_left for g in self.geometries], axis=0)
        bottom_rights = jnp.stack([g.aabb.bottom_right for g in self.geometries], axis=0)
        return Box(
            top_left=jnp.min(top_lefts, axis=0),
            bottom_right=jnp.max(bottom_rights, axis=0)
        )

    def signed_distance(self, x):
        distances = [g.signed_distance(x) for g in self.geometries]
        return jnp.min(distances, axis=0)

def union(*geoemtries):
    return Union(geoemtries)

@dataclass(jax=True)
class VMapUnion:
    geometries: Geometry

    @property
    def aabb(self):
        aabs = jax.vmap(lambda x: x.aabb)(self.geometries)
        return Box(
            top_left=jnp.min(aabs.top_left, axis=0),
            bottom_right=jnp.max(aabs.bottom_right, axis=0)
        )

    def signed_distance(self, x):
        distances = jax.vmap(lambda g: g.signed_distance(x))(self.geometries)
        return jnp.min(distances, axis=0)

def vmap_union(geoemtries):
    return VMapUnion(geoemtries)

def _aa_alpha(dist):
    return jnp.clip(-(2*dist - 1), 0, 1)

@jax.jit
def _aa(dist, color):
    alpha = _aa_alpha(dist)
    # modify the color alpha
    color = color.at[3].set(color[3]*alpha)
    return color

@jax.jit
def _composite(dist, color, background_color):
    alpha = _aa_alpha(dist)
    alpha_a = alpha * color[3]
    alpha_b = background_color[3] if background_color.shape[0] == 4 else 1.
    bc, fc = background_color[:3], color[:3]
    alpha_o = alpha_a + alpha_b * (1-alpha_a)
    color_o  = (alpha_a * fc + alpha_b * (1 - alpha_a) * bc) / alpha_o
    out = jnp.concatenate((color_o, jnp.expand_dims(alpha_o, axis=0)), axis=0) \
        if background_color.shape[0] == 4 else color_o
    return out


class Renderable:
    @property
    def aabb(self):
        raise NotImplementedError()

    def color_distance(self, x, pixel_metric_hessian):
        raise NotImplementedError()
    
    def rasterize(self, canvas, offset=None):
        coords_y = jnp.linspace(0, canvas.shape[0], num=canvas.shape[0])
        coords_x = jnp.linspace(0, canvas.shape[1], num=canvas.shape[1])
        if offset is not None:
            coords_y = coords_y + offset[0]
            coords_x = coords_x + offset[1]
        grid_x, grid_y = jnp.meshgrid(coords_x, coords_y)
        grid = jnp.stack((grid_x, grid_y), axis=-1)
        dists, colors = jax.vmap(jax.vmap(self.color_distance, 
                                          in_axes=(0, None)), 
                                          in_axes=(0,None))(
                                              grid, jnp.eye(2))
        canvas = jax.vmap(jax.vmap(_composite))(dists, colors, canvas)
        return canvas

@jax.jit
def paint(canvas, *renderables):
    for r in renderables:
        canvas = r.rasterize(canvas)
    return canvas


@dataclass(jax=True)
class Fill(Renderable):
    geometry: Geometry
    color: jnp.ndarray # must have 4 channels

    @property
    def aabb(self):
        return self.geometry.aabb

    @jax.jit
    def color_distance(self, x, pixel_metric_hessian):
        dist = self.geometry.signed_distance(x)
        return dist, self.color

def sanitize_color(color):
    color = jnp.array(color)
    # if alpha is not specified, make alpha = 1
    if color.shape[0] == 3:
        return jnp.concatenate((color, jnp.ones((1,))), axis=0)
    return color

def fill(geometry, color=jnp.array([0.,0.,0.,1.])):
    color = sanitize_color(color)
    return Fill(geometry, color)

@dataclass(jax=True)
class Stack(Renderable):
    renderables: List[Renderable]

    @property
    def aabb(self):
        return self.geometry.aabb
    
    def color_distance(self, x, pixel_metric_hessian):
        dist, color = self.renderables[0].color_distance(x, pixel_metric_hessian)
        for g in self.renderables[1:]:
            (s_dist, s_color), grad = jax.value_and_grad(g.color_distance, 
                                            has_aux=True, argnums=0)(x, pixel_metric_hessian)
            # use pixel-scaling for the antialiasing distance
            scaling = jnp.dot(grad, pixel_metric_hessian @ grad)
            aa_dist = s_dist * jnp.sqrt(scaling)
            color = _composite(aa_dist, s_color, color)
            dist = jnp.minimum(dist, s_dist)
        return dist, color

# Does an anti-aliases
def stack(*renderables):
    return Stack(renderables)

@dataclass(jax=True)
class Transformed(Renderable):
    renderable: Renderable

    scale: jnp.ndarray
    rotation: float
    translation: jnp.ndarray

    @property
    def aabb(self):
        return self.renderable.aabb
    
    @jax.jit
    def color_distance(self, x, pixel_metric_hessian):
        def _transform(x, pixel_metric_hessian):
            if self.scale is not None:
                x = x / self.scale
                M_inv = jnp.eye(2) * self.scale
                pixel_metric_hessian = M_inv @ pixel_metric_hessian @ M_inv.T
            if self.translation is not None:
                x = x - self.translation
            if self.rotation is not None:
                c, s = jnp.cos(self.rotation), jnp.sin(self.rotation)
                M = jnp.array(((c,-s),(s,c))) if self.rotation is not None else jnp.eye(2)
                # M_inv = jnp.array(((c,s),(-s,c))) if self.rotation is not None else jnp.eye(2)
                x = M @ x
            return self.renderable.color_distance(x, pixel_metric_hessian)
        # renormalize the distance
        # so that the gradient is norm 1
        (dist, color), grad = jax.value_and_grad(_transform,
                                    argnums=0, has_aux=True)(x, pixel_metric_hessian)
        dist = dist / jnp.linalg.norm(grad)
        return dist, color

def transform(fn, translation=None, rotation=None, scale=None):
    return Transformed(
        renderable=fn,
        translation=jnp.array(translation) if translation is not None else None,
        rotation=jnp.array(rotation) if rotation is not None else None,
        scale=jnp.array(scale) if scale is not None else None
    )