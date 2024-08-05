import jax
import jax.numpy as jnp
import numpy as np

import logging
import os.path
import io
import time
import IPython.display
import asyncio

from IPython.display import display

from ipyevents import Event
from ipywidgets import Image, HBox, HTML, Button, Label, IntSlider
from ipywebrtc.webrtc import ImageStream
from PIL import Image as PILImage
from threading import Thread

from stanza.data import PyTreeData
from stanza.data.sequence import SequenceData, Step
from stanza.random import PRNGSequence
from stanza.env import ImageRender
from stanza.util.ipython import STYLE

logger = logging.getLogger(__name__)

class DemonstrationCollector:
    def __init__(self, env, interactive_policy, 
                demonstrations, save, width, height, fps=30):
        self.demonstrations = demonstrations
        self._save_fn = save

        # if len(jax.devices()) > 1:
        #     for p in range(len(jax.devices()), 0, -1):
        #         if 256 % p == 0:
        #             break
        #     mesh = jax.sharding.Mesh(jax.devices()[:p], ('x',))
        #     sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x',))
        # else:
        sharding = None

        self._step_fn = jax.jit(env.step)
        self._reset_fn = jax.jit(env.reset)
        self._reduce_state = jax.jit(env.reduce_state)
        self._render_fn = jax.jit(
            lambda x: env.render(env.full_state(x), ImageRender(width, height)),
            out_shardings=sharding
        )
        self._policy = interactive_policy
        self.interface = StreamingInterface(width, height)

        # precompile the reset and step functions
        s = self._reset_fn(jax.random.key(42))
        self._sample_input = env.sample_action(jax.random.key(42))
        self._step_fn(s, self._sample_input, jax.random.key(42))
        self._render_fn(self._reduce_state(s))

        self.env = env
        self.fps = fps
        self.width = width
        self.height = height
        self.rng = PRNGSequence(42)

        # The currently collecting demonstration
        self.curr_demonstration = None
        self.curr_state = None
        self.collect_task = None

        # The reseted state, if we want to collect a new demonstration
        self.reset_state = None

        self.demonstration_slider = IntSlider(min=0, max=len(demonstrations))
        self.step_slider = IntSlider()
        self.demonstration_slider.observe(lambda _: self._demo_changed(), names='value')
        self.step_slider.observe(lambda _: self._step_changed(), names='value')

        self.interface.set_key_callback('c', self._toggle_collection)
        self.interface.set_key_callback('r', self._reset_state)

        self.save_button = Button(description='Save')
        self.save_button.on_click(self._do_save)
        self.reset_button = Button(description='Reset (r)')
        self.reset_button.on_click(lambda _: self._reset_state())
        self.collect_button = Button(description='Collect (c)')
        self.collect_button.on_click(lambda _: self._toggle_collection())
        self.delete_button = Button(description='Delete')
        self.delete_button.on_click(lambda _: self._delete_demonstration())
        self.trim_button = Button(description='Trim')
        self.trim_button.on_click(lambda _: self._trim_demonstration())

        self.countdown = Label(value='')

        self.loop = asyncio.new_event_loop()
        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        self.loop_thread = Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        self._visualize()
    
    def save(self):
        if self.demonstrations is None:
            return
        self._save_fn(self.demonstrations)

    def _ipython_display_(self):
        display(HBox([self.demonstration_slider, self.step_slider]))
        display(self.interface)
        display(HBox([self.delete_button, self.reset_button, self.collect_button, self.trim_button, self.save_button]))
        display(self.countdown)
    
    def _visualize(self):
        d = self.demonstration_slider.value
        T = self.step_slider.value
        if (len(self.demonstrations) > 0) and \
                (d < len(self.demonstrations)):
            dem = self.demonstrations[d]
            T = min(T, len(dem) - 1)
            state = self.demonstrations[d][T].reduced_state
        elif self.curr_state is not None and self.curr_demonstration is not None:
            T = min(T, len(self.curr_demonstration))
            if T < len(self.curr_demonstration):
                state = self.curr_demonstration[self.step_slider.value].state
            else:
                state = self._reduce_state(self.curr_state)
        else:
            return
        image = self._render_fn(state)
        self.interface.update(image)
    
    def _do_save(self, change):
        self.save()
    
    def _reset_state(self):
        if self.collect_task is not None:
            return
        r = next(self.rng)
        self.curr_demonstration = []
        self.curr_state = self._reset_fn(r)
        self.demonstration_slider.max = len(self.demonstrations)
        self.demonstration_slider.value = len(self.demonstrations)
        self.step_slider.value = 0
        self.step_slider.max = 0
        self._visualize()

    def _step_changed(self):
        self._visualize()
    
    def _demo_changed(self):
        d = self.demonstration_slider.value
        if d < len(self.demonstrations):
            self.demonstration_slider.max = max(0, len(self.demonstrations) - 1)
            if self.curr_demonstration is not None:
                self._stop_collection()
                return
        if d == len(self.demonstrations) and self.curr_demonstration is not None:
            max_T = len(self.curr_demonstration)
        else:
            max_T = len(self.demonstrations[d]) - 1
        self.step_slider.max = max_T
        self.step_slider.value = min(max_T, self.step_slider.value)
        self._visualize()
    
    async def _collect(self):
        t = time.time()
        if self.curr_demonstration is None or self.curr_state is None:
            return
        state = self.curr_state
        while True:
            elapsed = time.time() - t
            action = self._policy(self.interface.mouse_pos())
            self.curr_demonstration.append(Step(self._reduce_state(state), None, action))
            state = self._step_fn(state, action, next(self.rng))
            self.curr_state = state
            self.step_slider.max = len(self.curr_demonstration)
            self.step_slider.value = len(self.curr_demonstration)
            self._visualize()
            await asyncio.sleep(max(1/self.fps - elapsed, 0))
            t = time.time()
    
    def _toggle_collection(self):
        if self.collect_task is None: self._start_collection()
        else: self._stop_collection()
        
    def _start_collection(self):
        self.collect_task = asyncio.run_coroutine_threadsafe(self._collect(), self.loop)
        self.collect_button.description = 'Stop (c)'
    
    def _stop_collection(self):
        if self.collect_task is not None:
            self.collect_task.cancel()
        self.collect_task = None
        self.collect_button.description = 'Collect (c)'
        if self.curr_demonstration is not None:
            if len(self.curr_demonstration) > 0:
                steps = jax.tree.map(lambda *xs: jnp.stack(xs), *self.curr_demonstration)
                self.demonstrations = self.demonstrations.append(
                    SequenceData.from_trajectory(PyTreeData(steps))
                )
            self.curr_demonstration = None
            self.curr_state = None
            self._demo_changed()

    def _trim_demonstration(self):
        if self.curr_demonstration is not None:
            return
        if len(self.demonstrations) > 0:
            d = min(self.demonstration_slider.value, len(self.demonstrations) - 1)
            demonstration = self.demonstrations[d]
            T = self.step_slider.value
            trimmed = jax.tree_map(lambda x: x[:T + 1], demonstration)
            self.demonstrations[d] = trimmed
            self.step_slider.max = T

    def _delete_demonstration(self):
        if self.curr_demonstration is not None:
            return
        if len(self.demonstrations) > 0:
            d = min(self.demonstration_slider.value, len(self.demonstrations) - 1)
            del self.demonstrations[d]
            self.demonstration_slider.max = len(self.demonstrations) - 1
            self._demo_changed()


class StreamingInterface:
    def __init__(self, width, height):
        self.stream = None

        self._mouse_pos = jnp.zeros((2,))
        self._key_events = []

        self._key_press_callbacks = {}

        self.array = 0.8*jnp.ones((width, height, 3))
        self.image = Image(value=self._to_bytes(self.array))
        self.stream = ImageStream(image=self.image)
        self.event = Event(source=self.stream, watched_events=['click', 'keydown', 'keyup', 'mousemove'])
        self.event.on_dom_event(self._handle_event)
    
    def update(self, array):
        assert array.shape[0] == self.array.shape[0]
        assert array.shape[1] == self.array.shape[1]
        self.array = array
        self.image.value = self._to_bytes(array)
    
    def set_key_callback(self, key, callback):
        self._key_press_callbacks[key] = callback

    def mouse_pos(self):
        return self._mouse_pos
    
    def key_events(self):
        events = self._key_events
        self._key_events = []
        return events
    
    def _ipython_display_(self):
        IPython.display.display(HBox([self.stream, HTML(STYLE)]))

    def _handle_event(self, event):
        if event['type'] == 'mousemove':
            mouse_pos = jax.numpy.array([event['offsetX'], event['offsetY']])
            limit = jax.numpy.array(self.array.shape[:2])
            mouse_pos = jax.numpy.minimum(mouse_pos, limit) / limit
            mouse_pos = 2*mouse_pos - 1
            self._mouse_pos = jax.numpy.array([mouse_pos[0], -mouse_pos[1]])
        elif event['type'] == 'keydown':
            key = event['key']
            if key in self._key_press_callbacks:
                self._key_press_callbacks[key]()
            else:
                self._key_events.append(key)

    def _to_bytes(self, array):
        array = np.array(array)
        array = np.nan_to_num(array, copy=False, 
                            nan=0, posinf=0, neginf=0)
        if array.ndim == 2:
            array = np.expand_dims(array, -1)
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = (array*255).clip(0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        img = PILImage.fromarray(array)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()