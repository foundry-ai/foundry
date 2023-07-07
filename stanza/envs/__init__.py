from stanza.env import Environment
from stanza.env.builders import create, register_lazy

register_lazy('pusht', '.pusht')
register_lazy('pendulum', '.pendulum')
register_lazy('linear', '.linear')
register_lazy('quadrotor', '.quadrotor')
register_lazy('gym', '.gymnasium')
register_lazy('gymnax', '.gymnax')
register_lazy('brax', '.brax')
register_lazy('robosuite', '.robosuite')