"""Implements utilities for building planar environments in Mujoco.
Builds the XML.
"""
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from stanza import struct, canvas
from stanza.envs import Environment, RenderConfig
from functools import cached_property

@struct.dataclass
class Geometry:
    mass: float = 1.
    pos: tuple[float, float] = (0., 0.)
    color: tuple[float, float, float] = canvas.colors.LightSlateGray

    @property
    def _geom_args(self):
        mass = f' mass="{self.mass:.4}"'
        pos = f' pos="{self.pos[0]:.4} {self.pos[1]:.4} 0.5"'
        color = f' rgba="{self.color[0]:.4} {self.color[1]:.4} {self.color[2]:.4} 1"'
        return f"{mass}{pos}{color}"

@struct.dataclass
class Circle(Geometry):
    radius: float = 1.
    com: tuple[float, float] = (0., 0.)

    @property
    def _geom_args(self):
        mass = f' mass="{self.mass:.4}"' \
            if self.mass is not None else ''
        pos = f' pos="{self.pos[0]:.4} {self.pos[1]:.4} {self.radius}"' \
            if self.pos is not None else ''
        color = f' rgba="{self.color[0]:.4} {self.color[1]:.4} {self.color[2]:.4} 1"' \
            if self.color is not None else ''
        return f"{mass}{pos}{color}"
    
    def to_canvas(self, color=None):
        color = color or self.color or canvas.colors.LightGray
        pos = jnp.array([self.pos[0], -self.pos[1]])
        return canvas.fill(canvas.circle(
            center=pos, radius=self.radius
        ), color=color)

    def to_xml(self):
        return f'<geom type="sphere" size="{self.radius:.4}"{self._geom_args}/>'
        # TODO: Use cylinder rather than sphere. Currently mjx does not support
        # return f'<geom type="cylinder" size="{self.radius:.4} 0.5"{self._geom_args}/>'

@struct.dataclass
class Box(Geometry):
    half_size: tuple[float, float] = (0.5, 0.5)
    com: tuple[float, float] = (0., 0.)

    def to_canvas(self, color=None):
        color = color or self.color or canvas.colors.LightGray
        x, y = jnp.array([self.pos[0], -self.pos[1]])
        hx, hy = self.half_size
        return canvas.fill(canvas.box(
            (x - hx, y - hy), (x + hx, y + hy)
        ), color=color)


    def to_xml(self):
        return f'<geom type="box" size="{self.half_size[0]:.4} {self.half_size[1]:.4} 0.5"{self._geom_args}/>'

@struct.dataclass
class Body:
    name: str
    geom: list[Geometry]
    pos: tuple[float, float] = (0., 0.)
    rot: float = 0.
    vel_damping: float = 0.01
    rot_damping: float = 0.01
    hinge: bool = True
    custom_com: tuple[float, float] = None

    @property
    def com(self):
        if self.custom_com is not None:
            return self.custom_com
        com = jnp.sum(jnp.array([jnp.array(geom.com) * geom.mass for geom in self.geom]), axis=0)
        return com

    @com.setter
    def com(self, value):
        self.custom_com = value

    def to_xml(self):
        geoms = "\n\t\t".join(geom.to_xml() for geom in self.geom)
        name = f' name="{self.name}"' if self.name else ''
        pos = f' pos="{self.pos[0]:.4} {self.pos[1]:.4} 0."' if self.pos else ''
        args = f'{name}{pos}'
        # Allow rotation in z plane and sliding in x,y
        damping = self.vel_damping
        rot_damping = self.rot_damping
        if self.hinge:
            com = self.com
            return f'''\t<body{args}>
            \t\t<joint type="slide" axis="1 0 0" damping="{damping}" stiffness="0" ref="{self.pos[0]:.4}"/>
            \t\t<joint type="slide" axis="0 1 0" damping="{damping}" stiffness="0" ref="{self.pos[1]:.4}"/>
            \t\t<joint type="hinge" axis="0 0 1" damping="{rot_damping}" stiffness="0" ref="{self.rot:.4}" pos="{com[0]} {com[1]} {com[2]}"/>
            \t\t{geoms}
            \t</body>'''
        else:
            return f'''\t<body{args}>
            \t\t<joint type="slide" axis="1 0 0" damping="{damping}" stiffness="0" ref="{self.pos[0]:.4}"/>
            \t\t<joint type="slide" axis="0 1 0" damping="{damping}" stiffness="0" ref="{self.pos[1]:.4}"/>
            \t\t{geoms}
            \t</body>'''

@struct.dataclass
class BodyState:
    pos: jax.Array = jnp.zeros((2,))
    vel: jax.Array = jnp.zeros((2,))
    rot: jax.Array = jnp.zeros(())
    rot_vel: jax.Array = jnp.zeros(())

TEMPLATE = """
<mujoco>
<option timestep="{dt}"/>
<worldbody>
{bodies}
{geoms}
    # The boundary and support planes
    <geom pos="-{world_half_x:.4} 0 0" size="{world_x:.4} {world_y:.4} 0.1"  xyaxes="0 1 0 0 0 1" type="plane"/>
    <geom pos="{world_half_x:.4} 0 0" size="{world_x:.4} {world_y:.4} 0.1"   xyaxes="0 0 1 0 1 0" type="plane"/>
    <geom pos="0 -{world_half_y:.4} 0" size="{world_x:.4} {world_y:.4} 0.1"  xyaxes="0 0 1 1 0 0" type="plane"/>
    <geom pos="0 {world_half_y:.4} 0" size="{world_x:.4} {world_y:.4} 0.1"   xyaxes="1 0 0 0 0 1" type="plane"/>
</worldbody>
</mujoco>
"""

class WorldBuilder:
    def __init__(self, world_half_x, world_half_y, dt=0.005):
        self.bodies = []
        self.geometries = []
        self.dt = dt
        self.world_half_x = float(world_half_x)
        self.world_half_y = float(world_half_y)
    
    def add_body(self, body: Body):
        self.bodies.append(body)

    def add_geom(self, geom: Geometry):
        self.geometries.append(geom)

    def extract_state(self, mjx_data: mjx.Data):
        state = {}
        pos = mjx_data.xpos[1:,:2]
        vel = mjx_data.cvel[1:,3:5]
        w0 = mjx_data.xquat[1:,0] # cos(theta/2)
        w3 = mjx_data.xquat[1:,3] # sin(theta/2)
        angle = 2*jax.numpy.atan2(w3, w0)
        avel = mjx_data.cvel[1:,2]
        for b, pos, vel, angle, avel in zip(self.bodies, pos, vel, angle, avel):
            if b.name is not None:
                state[b.name] = BodyState(pos, vel, angle, avel)
        return state
    
    def renderable(self, state: dict[str, BodyState], colors : dict[str, tuple[float, float, float]] = {}):
        bodies = []
        for body in self.bodies:
            if body.name not in state: continue
            body_state = state[body.name]
            pos = body_state.pos
            rot = body_state.rot
            geoms = []
            for geom in body.geom:
                color = colors.get(body.name, None)
                geoms.append(geom.to_canvas(color=color))
            geoms = canvas.stack(*geoms)
            bodies.append(canvas.transform(
                geoms, translation=jnp.array([pos[0], -pos[1]]), rotation=rot
            ))
        return canvas.stack(*bodies)
    
    def to_xml(self):
        bodies = "\n".join(body.to_xml() for body in self.bodies)
        geoms = "\n".join(geom.to_xml() for geom in self.geometries)
        return TEMPLATE.format(dt=self.dt,
            world_half_x=self.world_half_x, world_x=2*self.world_half_x,
            world_half_y=self.world_half_y, world_y=2*self.world_half_y,
            bodies=bodies, geoms=geoms
        ).strip().replace("\t", "    ")
    
    def load_model(self):
        xml = self.to_xml()
        model = mujoco.MjModel.from_xml_string(xml)
        return mjx.put_model(model)

    def load_mj_model(self):
        xml = self.to_xml()
        model = mujoco.MjModel.from_xml_string(xml)
        return model