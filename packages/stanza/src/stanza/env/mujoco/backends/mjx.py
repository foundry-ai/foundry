from mujoco import mjx
from stanza.env.mujoco.core import SystemState

from ..core import Simulator, SystemState, SystemData

class MjxSimulator(Simulator[mjx.Data]):
    def __init__(self, model):
        self.model = mjx.put_model(model)
    
    def step(self, state, action, rng_key):
        if action is not None:
            state = state.replace(ctrl=action)
        state = mjx.step(self.model, state)
        return state
    
    def full_state(self, state: SystemState) -> mjx.Data:
        data = mjx.make_data(self.model)
        data = data.replace(time=state.time, 
                     qpos=state.qpos,
                     qvel=state.qvel,
                     act=state.act)
        data = mjx.forward(self.model, data)
        return data
    
    def reduce_state(self, state: mjx.Data) -> SystemState:
        return SystemState(
            time=state.time,
            qpos=state.qpos,
            qvel=state.qvel,
            act=state.act
        )
    
    def data(self, state: mjx.Data) -> SystemData:
        return SystemData(
            time=state.time,
            qpos=state.qpos,
            qvel=state.qvel,
            act=state.act,
            qacc=state.qacc,
            act_dot=state.act_dot,
            xpos=state.xpos,
            xquat=state.xquat,
            actuator_velocity=state.actuator_velocity,
            cvel=state.cvel
        )