from stanza import struct
from stanza.graphics import sanitize_color
import jax.numpy as jnp
import jax

from typing import List, Protocol

class Geometry:
    @property
    def aabb(self): ...

    def signed_distance(self, x): ...

    def fill(self, color):
        color = sanitize_color(color)
        return Fill(self, color)

    # def transform(self, translation=None, rotation=None, scale=None):
    #     return TransformedGeometry(
    #         geometry=self,
    #         translation=jnp.array(translation) if translation is not None else None,
    #         rotation=jnp.array(rotation) if rotation is not None else None,
    #         scale=jnp.array(scale) if scale is not None else None
    #     )

@struct.dataclass
class Box(Geometry):
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

@struct.dataclass
class Polygon(Geometry):
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

@struct.dataclass
class Circle(Geometry):
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

@struct.dataclass
class Segment(Geometry):
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

@struct.dataclass
class Union(Geometry):
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

@struct.dataclass
class BatchUnion(Geometry):
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

def union_batch(geoemtries):
    return BatchUnion(geoemtries)

def _aa_alpha(dist):
    return jnp.clip(-(2*dist - 1), 0, 1)

def _split_color(color):
    if color.shape[0] == 1:
        return color, jnp.ones(())
    elif color.shape[0] == 2:
        return color[:1], color[1]
    elif color.shape[0] == 3:
        return color, jnp.ones(())
    elif color.shape[0] == 4:
        return color[:3], color[3]
    else:
        raise ValueError(f"Invalid color shape {color.shape}")

@jax.jit
def _composite(dist, color, background_color):
    alpha = _aa_alpha(dist)
    fc, alpha_a = _split_color(color)
    bc, alpha_b = _split_color(background_color)
    if bc.shape[0] == 1:
        fc = jnp.mean(fc, axis=0, keepdims=True)
    alpha_a = alpha_a * alpha
    alpha_o = alpha_a + alpha_b * (1-alpha_a)
    color_o  = (alpha_a * fc + alpha_b * (1 - alpha_a) * bc) / alpha_o
    if background_color.shape[0] == 4 or background_color.shape[0] == 2:
        color_o = jnp.concatenate((color_o, alpha_o[None]), axis=0)
    return color_o

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

    def transform(self, translation=None, rotation=None, scale=None):
        return TransformedRenderable(
            renderable=self,
            translation=jnp.array(translation) if translation is not None else None,
            rotation=jnp.array(rotation) if rotation is not None else None,
            scale=jnp.array(scale) if scale is not None else None
        )

@jax.jit
def paint(canvas, *renderables):
    for r in renderables:
        canvas = r.rasterize(canvas)
    return canvas

@struct.dataclass
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

def fill(geometry, color=jnp.array([0.,0.,0.,1.])):
    color = sanitize_color(color)
    return Fill(geometry, color)

@struct.dataclass
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

@struct.dataclass
class BatchStack:
    renderables: Renderable

    @property
    def aabb(self):
        aabs = jax.vmap(lambda x: x.aabb)(self.renderables)
        return Box(
            top_left=jnp.min(aabs.top_left, axis=0),
            bottom_right=jnp.max(aabs.bottom_right, axis=0)
        )

    def color_distance(self, x, pixel_metric_hessian):
        func = lambda g: jax.value_and_grad(g.color_distance, has_aux=True, argnums=0)(x, pixel_metric_hessian)
        (distances, colors), grads = jax.vmap(func)(self.renderables)
        dist, color = distances[0], colors[0]
        def compose(carry, scan):
            dist, color = carry
            s_dist, s_color, grad = scan
            scaling = jnp.dot(grad, pixel_metric_hessian @ grad)
            aa_dist = s_dist * jnp.sqrt(scaling)
            color = _composite(aa_dist, s_color, color)
            dist = jnp.minimum(dist, s_dist)
            return (dist, color), None
        (dist, color), _ = jax.lax.scan(compose, 
                    (dist, color), 
                    (distances[1:], colors[1:], grads[1:]))
        return dist, color

def stack_batch(renderables):
    return BatchStack(renderables)

@struct.dataclass
class TransformedRenderable(Renderable):
    renderable: Renderable

    scale: jnp.ndarray
    rotation: float
    translation: jnp.ndarray

    @property
    def aabb(self):
        raise NotImplementedError()
    
    @jax.jit
    def color_distance(self, x, pixel_metric_hessian):
        def _transform(x, pixel_metric_hessian):
            if self.scale is not None:
                x = x / self.scale
                M_inv = jnp.diag(self.scale)
                pixel_metric_hessian = M_inv @ pixel_metric_hessian @ M_inv.T
            if self.translation is not None:
                x = x - self.translation
            if self.rotation is not None:
                c, s = jnp.cos(self.rotation), jnp.sin(self.rotation)
                M = jnp.array(((c,-s),(s,c))) if self.rotation is not None else jnp.eye(2)
                M_inv = jnp.array(((c,s),(-s,c))) if self.rotation is not None else jnp.eye(2)
                pixel_metric_hessian = M_inv @ pixel_metric_hessian @ M_inv.T
                x = M @ x
            return self.renderable.color_distance(x, pixel_metric_hessian)
        # renormalize the distance
        # so that the gradient is norm 1
        (dist, color), grad = jax.value_and_grad(_transform,
                                    argnums=0, has_aux=True)(x, pixel_metric_hessian)
        dist = dist / jnp.linalg.norm(grad)
        return dist, color

def transform(r, translation=None, rotation=None, scale=None):
    return r.transform(
        translation=jnp.array(translation) if translation is not None else None,
        rotation=jnp.array(rotation) if rotation is not None else None,
        scale=jnp.array(scale) if scale is not None else None
    )