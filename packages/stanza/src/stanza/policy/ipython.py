import jax
import jax.numpy as jnp
import numpy as np

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
from stanza.data.sequence import SequenceInfo, SequenceData
from stanza.random import PRNGSequence
from stanza.env import SequenceRender
from stanza.util.ipython import STYLE

class DemonstrationCollector:
    def __init__(self, path, env, interactive_policy, width, height, fps=30):
        self.path = path
        
        if len(jax.devices()) > 1:
            for p in range(len(jax.devices()), 0, -1):
                if 256 % p == 0:
                    break
            mesh = jax.sharding.Mesh(jax.devices()[:p], ('x',))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x',))
        else:
            sharding = None

        self._step_fn = jax.jit(env.step)
        self._reset_fn = jax.jit(env.reset)
        self._render_fn = jax.jit(env.render, out_shardings=sharding)
        self._policy = interactive_policy
        self.interface = StreamingInterface(width, height)

        # precompile the reset and step functions
        s = self._reset_fn(jax.random.PRNGKey(42))
        self._sample_input = interactive_policy(self.interface.mouse_pos())
        self._step_fn(s, self._sample_input)
        self._render_fn(SequenceRender(width, height), s)

        self.env = env
        self.demonstrations = []
        self.fps = fps
        self.width = width
        self.height = height
        self.rng = PRNGSequence(42)

        # The currently collecting demonstration
        self.col_demonstration = None
        self.collect_task = None

        # The reseted state, if we want to collect a new demonstration
        self.reset_state = None

        self.demonstration_slider = IntSlider(min=0, max=max(0,len(self.demonstrations) - 1))
        self.step_slider = IntSlider()
        self.demonstration_slider.observe(lambda _: self._demo_changed(), names='value')
        self.step_slider.observe(self._step_changed, names='value')

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

        if os.path.exists(self.path):
            self.load(self.path)
    
    def save(self, path):
        elements = jax.tree_util.tree_map(
            lambda *x: jnp.concatenate(x, axis=0), *self.demonstrations
        )
        def length(demo):
            return jax.tree_util.tree_flatten(demo)[0][0].shape[0]
        id = jnp.arange(len(self.demonstrations))
        lengths = jnp.array([length(d) for d in self.demonstrations])
        end_idx = jnp.cumsum(lengths)
        start = end_idx - lengths
        infos = SequenceInfo(
            id=id,
            info=None,
            start_idx=start,
            end_idx=end_idx,
            length=lengths
        )
        data = SequenceData(PyTreeData(elements), PyTreeData(infos))
        data.save(path)

    def load(self, path):
        data = SequenceData.load(path)
        infos = data.infos.as_pytree()
        elements = data.elements.as_pytree()
        for t in range(len(infos.id)):
            start = infos.start_idx[t]
            end = infos.end_idx[t]
            demo = jax.tree_map(lambda x: x[start:end], elements)
            self.demonstrations.append(demo)
        self.demonstration_slider.max = max(0,len(self.demonstrations) - 1)
        self._demo_changed()

    def _ipython_display_(self):
        display(HBox([self.demonstration_slider, self.step_slider]))
        display(self.interface)
        display(HBox([self.delete_button, self.reset_button, self.collect_button, self.trim_button, self.save_button]))
        display(self.countdown)
    
    def _visualize(self):
        if self.col_demonstration is None and len(self.demonstrations) > 0:
            d = min(self.demonstration_slider.value, len(self.demonstrations) - 1)
            T = self.step_slider.value
            demonstration = self.demonstrations[d]
            state = jax.tree_map(lambda x: x[T], demonstration[0])
        elif self.col_demonstration is not None:
            state = self.col_demonstration[-1][0]
        image = self._render_fn(SequenceRender(self.width, self.height), state)
        self.interface.update(image)
    
    def _do_save(self, change):
        self.save(self.path)
    
    def _reset_state(self):
        if self.collect_task is not None:
            return
        r = next(self.rng)
        self.col_demonstration = [(self._reset_fn(r), self._sample_input)]
        self._visualize()
        self.demonstration_slider.max = len(self.demonstrations)
        self.demonstration_slider.value = len(self.demonstrations)
        self.step_slider.value = 0
        self.step_slider.max = 0

    def _step_changed(self, change):
        self._visualize()
    
    def _demo_changed(self):
        d = self.demonstration_slider.value
        if self.col_demonstration is not None and d < len(self.demonstrations):
            self._stop_collection()
        if self.col_demonstration is None and len(self.demonstrations) > 0:
            max_T = jax.tree_util.tree_flatten(self.demonstrations[d])[0][0].shape[0]
            self.step_slider.max = max_T
            self.step_slider.value = min(max_T, self.step_slider.value)
        self._visualize()
    
    async def _collect(self):
        t = time.time()
        if self.col_demonstration is None:
            return
        state = self.col_demonstration[-1][0]
        self.demonstration_slider.value = len(self.demonstrations) 
        self.demonstration_slider.max = len(self.demonstrations) 
        while True:
            elapsed = time.time() - t
            action = self._policy(self.interface.mouse_pos())
            self.col_demonstration.append((state, action))
            state = self._step_fn(state, action)
            self.step_slider.max = len(self.col_demonstration)
            self.step_slider.value = len(self.col_demonstration)
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
        if self.col_demonstration is not None and len(self.col_demonstration) > 1:
            states, actions = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *self.col_demonstration)
            # remove the dummy action as the start, last state at the end
            actions = jax.tree_map(lambda x: x[1:], actions)
            states = jax.tree_map(lambda x: x[:-1], states)
            self.demonstrations.append((states, actions))
        self.demonstration_slider.max = max(0,len(self.demonstrations) - 1)
        self.col_demonstration = None

    def _trim_demonstration(self):
        if self.col_demonstration is not None:
            return
        if len(self.demonstrations) > 0:
            d = min(self.demonstration_slider.value, len(self.demonstrations) - 1)
            demonstration = self.demonstrations[d]
            T = self.step_slider.value
            trimmed = jax.tree_map(lambda x: x[:T + 1], demonstration)
            self.demonstrations[d] = trimmed
            self.step_slider.max = T

    def _delete_demonstration(self):
        if self.col_demonstration is not None:
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