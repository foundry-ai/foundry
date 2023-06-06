import glfw
import numpy as np
import OpenGL.GL as gl

def opengl_error_check():
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print("OPENGL_ERROR: ", gl.gluErrorString(error))

_INIT = False
_KEY_CODES = {
    'w': glfw.KEY_W,
    'a': glfw.KEY_A,
    's': glfw.KEY_S,
    'd': glfw.KEY_D,
    'q': glfw.KEY_Q,
    'e': glfw.KEY_E
}

class Window:
    def __init__(self, title=None):
        global _INIT
        if not _INIT:
            glfw.init()
            _INIT = True
        self.title = title or ""
        self.window = None

    def _init(self, width, height):
        self.window = glfw.create_window(width, height, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        # setup the texture
        gl.glEnable(gl.GL_TEXTURE_2D)
        self.texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexEnvf(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, 
            gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,
            gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
                        width, height, 0, gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE, np.zeros((height, width, 3)))

    @property
    def close_requested(self):
        if not self.window:
            return False
        return glfw.window_should_close(self.window)
    
    def key_pressed(self, key):
        code = _KEY_CODES[key]
        return glfw.get_key(self.window, code) == glfw.PRESS

    def show(self, buffer):
        buffer = np.array(buffer[...,:3])
        if buffer.dtype != np.uint8:
            buffer = (255*buffer).astype(np.uint8)
        if not self.window:
            self._init(buffer.shape[1], buffer.shape[0])
        glfw.make_context_current(self.window)
        # blit the buffer to the window
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, buffer.shape[1],
                           buffer.shape[0], gl.GL_RGB, 
                           gl.GL_UNSIGNED_BYTE, buffer)
        gl.glBegin(gl.GL_QUADS)
        gl.glColor3f(1, 0, 0)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex3f(-1.0,-1.0, 0.0)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex3f( 1.0,-1.0, 0.0)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex3f( 1.0, 1.0, 0.0)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex3f(-1.0, 1.0, 0.0)
        gl.glEnd()
        glfw.poll_events()
        glfw.swap_buffers(self.window)
    
    def close(self):
        glfw.terminate()