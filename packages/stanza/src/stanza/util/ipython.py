import time
import numpy as np
import tempfile
import ffmpegio
import uuid
import rich.jupyter
import rich.terminal_theme
import rich.live
import IPython
import io
from PIL import Image as PILImage

import jax
import jax.numpy as jnp

from pathlib import Path
from ipywebrtc.webrtc import ImageStream
from ipywidgets import Video, Image, HBox, HTML, Output
from ipyevents import Event
from rich.segment import Segment
from rich.control import Control
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
video::-webkit-media-controls {
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
    print(array.shape)
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

# Some code to set up rich to display properly
# (even in vscode) and to interact better with logging output

# Hook into rich to display in jupyter
_JUPYTER_HTML_FORMAT = """\
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">{code}</pre>
"""

def _jupyter_render_segments(segments: Iterable[Segment]) -> str:
    def escape(text: str) -> str:
        """Escape html."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    fragments = []
    append_fragment = fragments.append
    theme = rich.terminal_theme.MONOKAI
    for text, style, control in Segment.simplify(segments):
        if control:
            continue
        text = escape(text)
        if style:
            rule = style.get_html_style(theme)
            text = f'<span style="{rule}">{text}</span>' if rule else text
            if style.link:
                text = f'<a href="{style.link}" target="_blank">{text}</a>'
        append_fragment(text)

    code = "".join(fragments)
    html = _JUPYTER_HTML_FORMAT.format(code=code)

    return html

_current_cell_id = 0

_output_cell_id = None
_current_output = None
_current_box = None

from IPython.core.interactiveshell import InteractiveShell

_orig_runcell = InteractiveShell.run_cell
def wrapped_run_cell(self, *args, **kwargs):
    global _current_cell_id
    _current_cell_id = _current_cell_id + 1
    return _orig_runcell(self, *args, **kwargs)
InteractiveShell.run_cell = wrapped_run_cell

def _add_to_rich_display(renderable):
    global _current_cell_id
    global _output_cell_id
    global _current_output
    global _current_box
    if _output_cell_id != _current_cell_id or _current_output is None:
        _output_cell_id = _current_cell_id
        _current_output = Output()
        _current_output.append_display_data(renderable)
        _current_box = HBox([HTML(STYLE), _current_output])
        IPython.display.display(_current_box)
    else:
        _current_output.append_display_data(renderable)

_direct_display = False

def _rich_jupyter_display(segments: Iterable[Segment], text: str):
    html = _jupyter_render_segments(segments)
    jupyter_renderable = rich.jupyter.JupyterRenderable(html, text)
    if _direct_display:
        IPython.display.display(jupyter_renderable)
    else:
        _add_to_rich_display(jupyter_renderable)

def display(*objs, **kwargs):
    global _current_output
    # interrupt current output
    _current_output = None
    return IPython.display.display(*objs, **kwargs)

def _rich_live_refresh(self):
    with self._lock:
        self._live_render.set_renderable(self.renderable)
        if self.console.is_jupyter:  # pragma: no cover
            try:
                from IPython.display import display
                from ipywidgets import Output
            except ImportError:
                import warnings
                warnings.warn('install "ipywidgets" for Jupyter support')
            else:
                if self.ipy_widget is None:
                    self.ipy_widget = Output()
                    # _add_to_rich_display(self.ipy_widget)
                    display(HBox([HTML(STYLE), self.ipy_widget]))
                    # Force the next output to create a new display
                    global _current_output
                    _current_output = None

                with self.ipy_widget:
                    self.ipy_widget.clear_output(wait=True)
                    global _direct_display
                    _direct_display = True
                    self.console.print(self._live_render.renderable)
                    _direct_display = False
        elif self.console.is_terminal and not self.console.is_dumb_terminal:
            with self.console:
                self.console.print(rich.Control())
        elif (
            not self._started and not self.transient
        ):  # if it is finished allow files or dumb-terminals to see final result
            with self.console:
                self.console.print(rich.Control())

rich_live_stop_orig = rich.live.Live.stop
def _rich_live_stop(self):
    rich_live_stop_orig(self)
    self.refresh()

def setup_rich_notebook_hook():
    if rich.get_console().is_jupyter:
        rich.jupyter.display = _rich_jupyter_display
        rich.live.Live.refresh = _rich_live_refresh
        rich.live.Live.stop = _rich_live_stop