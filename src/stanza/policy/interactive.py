from stanza import struct

import jax

KEY_MAP = {
    chr(ord('a') + i): i for i in range(26)
}

@struct.dataclass
class InteractiveInput:
    # Mouse position
    mouse_pos: jax.Array
    mouse_click_pos: jax.Array
    mouse_clicked: jax.Array

    # Keyboard state (26,) dimensions
    keyboard_state: jax.Array