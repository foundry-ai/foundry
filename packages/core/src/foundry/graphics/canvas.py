"""
A simple functional API for creating and manipulating 2D geometries
and rasterizing them onto a canvas.
"""
from foundry.core.dataclasses import dataclass, field
from foundry.core.typing import Array, ArrayLike
from functools import partial

import foundry.core as F
import foundry.numpy as jnp
import jax
import math

from . import sanitize_color

from typing import Sequence, Self

class Geometry:
    @property
    def aabb(self) -> "Box": ...

    def signed_distance(self, x) -> Array: ...

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

@dataclass
class Box(Geometry):
    top_left: jnp.ndarray
    bottom_right: jnp.ndarray

    @property
    def aabb(self) -> "Box":
        return self

    def signed_distance(self, x) -> Array:
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
        top_left=jnp.array(top_left, dtype=jnp.float32),
        bottom_right=jnp.array(bottom_right, dtype=jnp.float32)
    )
rectangle = box

@dataclass
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
    return Polygon(jnp.array(vertices, dtype=jnp.float32))

import optax

@dataclass
class Circle(Geometry):
    center: Array
    radius: Array

    @property
    def aabb(self):
        return Box(
            top_left=self.center - self.radius,
            bottom_right=self.center + self.radius
        )

    def signed_distance(self, x) -> Array:
        d = optax.safe_norm(x - self.center, 1e-3)
        return d - self.radius

def circle(center, radius):
    return Circle(jnp.array(center, dtype=jnp.float32), radius)

@dataclass
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
    return Segment(jnp.array(a, dtype=jnp.float32), jnp.array(b, dtype=jnp.float32), thickness)

# Composed geometries

@dataclass
class Union(Geometry):
    geometries: Sequence[Geometry]

    @property
    def aabb(self):
        top_lefts = jnp.stack([g.aabb.top_left for g in self.geometries], axis=0)
        bottom_rights = jnp.stack([g.aabb.bottom_right for g in self.geometries], axis=0)
        return Box(
            top_left=jnp.min(top_lefts, axis=0),
            bottom_right=jnp.max(bottom_rights, axis=0)
        )

    def signed_distance(self, x):
        distances = jnp.array([g.signed_distance(x) for g in self.geometries], dtype=jnp.float32)
        return jnp.min(distances, axis=0)

def union(*geoemtries):
    return Union(geoemtries)

@dataclass
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
def _aa_color(dist, color):
    alpha = _aa_alpha(dist)
    fc, alpha_a = _split_color(color)
    alpha_a = alpha_a * alpha
    return jnp.concatenate((fc, alpha_a[None]), axis=0)

@jax.jit
def _composite(colors):
    c, alphas = jax.vmap(_split_color)(colors)
    # trans_part[i] = product of 1 - alphas[j] for j > i
    # i.e. the combined transparency of all the layers above i
    trans_part = jnp.roll((1 - alphas)[::-1], 1).at[0].set(1)
    trans_part = jnp.cumprod(trans_part)[::-1]
    # the contribution of the ith layer
    contrib = trans_part * alphas
    alpha_o = jnp.sum(contrib, axis=0)
    contrib = contrib / jnp.maximum(alpha_o, 1e-4)
    color_o = jnp.sum(c * contrib[:,None], axis=0)
    return jnp.concatenate((color_o, alpha_o[None]), axis=0)

def _composite_pair(color_fg, color_bg):
    c_fg, alpha_fg = _split_color(color_fg)
    c_bg, alpha_bg = _split_color(color_bg)
    if c_bg.shape[-1] == 1:
        c_fg = jnp.mean(c_fg, axis=-1, keepdims=True)
    elif c_fg.shape[-1] == 1 and c_bg.shape[-1] == 3:
        c_fg = c_fg.repeat(3, axis=-1)
    alpha_o = alpha_fg + alpha_bg*(1-alpha_fg)
    color_o = (c_fg*alpha_fg + c_bg*alpha_bg*(1-alpha_fg))/jnp.maximum(alpha_o, 1e-4)
    return jnp.concatenate((color_o, alpha_o[None]), axis=0)

class Renderable:
    @property
    def aabb(self):
        raise NotImplementedError()

    def color_distance(self, x, pixel_metric_hessian) -> tuple[Array, Array]:
        raise NotImplementedError()

    def rasterize(self, canvas, offset=None):
        coords_y = jnp.linspace(0, canvas.shape[0], num=canvas.shape[0])
        coords_x = jnp.linspace(0, canvas.shape[1], num=canvas.shape[1])
        if offset is not None:
            coords_y = coords_y + offset[0]
            coords_x = coords_x + offset[1]
        grid_x, grid_y = jnp.meshgrid(coords_x, coords_y)
        grid = jnp.stack((grid_x, grid_y), axis=-1)
        def render(x, background_color):
            dist, c = self.color_distance(x, jnp.eye(2))
            c = _aa_color(dist, c)
            return _composite_pair(c, background_color)
        canvas = jax.vmap(jax.vmap(render))(grid, canvas)
        return canvas

    def transform(self, translation=None, rotation=None, scale=None):
        return TransformedRenderable(
            renderable=self,
            translation=jnp.array(translation, dtype=jnp.float32) if translation is not None else None,
            rotation=jnp.array(rotation, dtype=jnp.float32) if rotation is not None else None,
            scale=jnp.array(scale, dtype=jnp.float32) if scale is not None else None
        )

@jax.jit
def paint(canvas, *renderables):
    for r in renderables:
        canvas = r.rasterize(canvas)
    return canvas

@dataclass
class Fill(Renderable):
    geometry: Geometry
    color: Array # must have 4 channels

    @property
    def aabb(self):
        return self.geometry.aabb

    @jax.jit
    def color_distance(self, x, pixel_metric_hessian) -> tuple[Array, Array]:
        dist = self.geometry.signed_distance(x)
        return dist, self.color

def fill(geometry, color=jnp.array([0.,0.,0.,1.], dtype=jnp.float32)):
    color = sanitize_color(color)
    return Fill(geometry, color)

@dataclass
class Stack(Renderable):
    renderables: Sequence[Renderable]

    @property
    def aabb(self):
        return self.geometry.aabb

    def color_distance(self, x, pixel_metric_hessian):
        if len(self.renderables) == 0:
            return jnp.inf, jnp.zeros(4)
        dists = []
        s_colors = []
        grads = []
        for g in self.renderables:
            #(s_dist, s_color), grad = jax.value_and_grad(g.color_distance,
            #                                has_aux=True, argnums=0)(x, pixel_metric_hessian)
            s_dist, s_color = g.color_distance(x, pixel_metric_hessian)
            dists.append(s_dist)
            s_colors.append(s_color)
            #grads.append(grad)
        dists, s_colors, grads = (
            jnp.array(dists, dtype=jnp.float32),
            jnp.array(s_colors, dtype=jnp.float32),
            jnp.array(grads, dtype=jnp.float32)
        )
        grad = jnp.array([0., 1.], dtype=jnp.float32)
        scalings = jnp.sqrt(jnp.dot(grad, pixel_metric_hessian @ grad))
        #scalings = jax.vmap(lambda grad: jnp.sqrt(jnp.dot(grad, pixel_metric_hessian @ grad)))(grads)
        aa_dist = dists * scalings
        colors = jax.vmap(_aa_color)(aa_dist, s_colors)
        color = _composite(colors)
        dist = jnp.min(dists)
        return dist, color

# Does an anti-aliases
def stack(*renderables):
    return Stack(renderables)

@dataclass
class BatchStack(Renderable):
    renderables: Renderable

    @property
    def aabb(self):
        aabs = jax.vmap(lambda x: x.aabb)(self.renderables)
        return Box(
            top_left=jnp.min(aabs.top_left, axis=0),
            bottom_right=jnp.max(aabs.bottom_right, axis=0)
        )

    @F.jit
    def color_distance(self, x, pixel_metric_hessian) -> tuple[Array, Array]:
        #func = lambda g: jax.value_and_grad(g.color_distance, has_aux=True, argnums=0)(x, pixel_metric_hessian)
        #(dists, colors), grads = jax.vmap(func)(self.renderables)
        dists, colors = jax.vmap(lambda g: g.color_distance(x, pixel_metric_hessian))(self.renderables)
        grad = jnp.array([0., 1.], dtype=jnp.float32)
        #scalings = jax.vmap(lambda grad: jnp.sqrt(jnp.dot(grad, pixel_metric_hessian @ grad)))(grads)
        scalings = jnp.sqrt(jnp.dot(grad, pixel_metric_hessian @ grad))
        aa_dist = dists * scalings
        colors = jax.vmap(_aa_color)(aa_dist, colors)
        color = _composite(colors)
        dist = jnp.min(dists)
        return dist, color

def stack_batch(renderables):
    return BatchStack(renderables)

@dataclass
class TransformedRenderable(Renderable):
    renderable: Renderable

    scale: ArrayLike
    rotation: ArrayLike
    translation: ArrayLike

    @property
    def aabb(self):
        raise NotImplementedError()

    @F.jit
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
                M = jnp.array(((c,-s),(s,c)), dtype=jnp.float32) if self.rotation is not None else jnp.eye(2)
                M_inv = jnp.array(((c,s),(-s,c)), dtype=jnp.float32) if self.rotation is not None else jnp.eye(2)
                pixel_metric_hessian = M @ pixel_metric_hessian @ M.T
                x = M_inv @ x
            return self.renderable.color_distance(x, pixel_metric_hessian)
        # renormalize the distance
        # so that the gradient is norm 1
        #(dist, color), grad = jax.value_and_grad(_transform,
        #                            argnums=0, has_aux=True)(x, pixel_metric_hessian)
        #dist = dist / jnp.linalg.norm(grad)
        dist, color = _transform(x, pixel_metric_hessian)
        return dist, color

def transform(r, translation=None, rotation=None, scale=None):
    return r.transform(
        translation=jnp.array(translation, dtype=jnp.float32) if translation is not None else None,
        rotation=jnp.array(rotation, dtype=jnp.float32) if rotation is not None else None,
        scale=jnp.array(scale, dtype=jnp.float32) if scale is not None else None
    )


# generic graphics utilities

def batch(*objs):
    return jax.tree_map(lambda *x: jnp.stack(x, axis=0), *objs)

# colors
class colors:
    Red = (1., 0., 0.)
    Green = (0., 1., 0.)
    Blue = (0., 0., 1.)
    LightGreen = (0.565, 0.933, 0.565)
    LightSlateGray = (0.467, 0.533, 0.60)
