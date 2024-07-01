import time
import numpy as np
import tempfile
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ffmpegio
import uuid
import rich.jupyter
import rich.live
import io
import os

from PIL import Image as PILImage

import jax
import jax.numpy as jnp

from pathlib import Path
from ipywebrtc.webrtc import ImageStream
from ipywidgets import Video, Image, HBox, HTML, Output
from ipyevents import Event
from typing import Iterable

STYLE = """<style>
.cell-output-ipywidget-background {
    background-color: transparent !important;
}
.jp-OutputArea-output {
    background-color: transparent;
}
pre {
    color: var(--vscode-editor-foreground);
    margin: 0;
}
.jupyter-widget > video::-webkit-media-controls {
  display: none;
}
</style>"""

def as_image(array):
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
    id = uuid.uuid4()
    path = Path("/tmp") / "notebook" / (str(id) + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return HBox([Image.from_file(path), HTML(STYLE)])

def as_video(array, fps=28):
    array = np.array(array)
    array = np.nan_to_num(array, copy=False, 
                          nan=0, posinf=0, neginf=0)
    if array.dtype == np.float32 or array.dtype == np.float64:
        array = (array*255).clip(0, 255).astype(np.uint8)
    f = tempfile.mktemp() + ".mp4"
    id = uuid.uuid4()
    path = Path("/tmp") / "notebook" / (str(id) + ".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpegio.video.write(path, fps, array)
    return HBox([Video.from_file(path), HTML(STYLE)])

def _rich_live_refresh(self):
    with self._lock:
        self._live_render.set_renderable(self.renderable)
        from IPython.display import TextDisplayObject
        def _render_to_text():
            loopback = io.StringIO()
            self.loopback_console.file = loopback
            with self.loopback_console:
                self.loopback_console.print(self._live_render.renderable)
            value = loopback.getvalue()
            return value
        self.jupyter_display.update({"text/plain": _render_to_text()}, raw=True)

def _rich_live_start(self, refresh: bool = False):
    from rich.live import _RefreshThread
    from IPython.display import display
    with self._lock:
        if self._started:
            return
        loopback = io.StringIO()
        # render to an offscreen console:
        loopback_console = rich.console.Console(
            force_jupyter=False,
            force_terminal=False,
            force_interactive=False,
            no_color=False,
            color_system="standard",
            file=loopback
        )
        self.loopback_console = loopback_console
        self.jupyter_display = display(display_id=True)

        self._started = True
        if refresh:
            try:
                self.refresh()
            except Exception:
                self.stop()
                raise
        if self.auto_refresh:
            self._refresh_thread = _RefreshThread(self, self.refresh_per_second)
            self._refresh_thread.start()

def _rich_live_stop(self):
    with self._lock:
        if not self._started:
            return
        self.console.clear_live()
        self._started = False
        if self.auto_refresh and self._refresh_thread is not None:
            self._refresh_thread.stop()
            self._refresh_thread = None
        self.refresh()

def setup_rich_notebook_hook():
    if rich.get_console().is_jupyter:
        # reconfigure not to use the jupyter console
        rich.reconfigure(
            color_system="standard",
            force_terminal=False,
            force_jupyter=False,
            force_interactive=True,
            no_color=False
        )
        rich.live.Live.refresh = _rich_live_refresh
        rich.live.Live.stop = _rich_live_stop
        rich.live.Live.start = _rich_live_start