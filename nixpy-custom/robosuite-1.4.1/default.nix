{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let numpy = dependencies.numpy;
    mujoco = dependencies.mujoco;
    numba = dependencies.numba;
    pillow = dependencies.pillow;
    pynput = dependencies.pynput;
    scipy = dependencies.scipy;
    termcolor = dependencies.termcolor;
    setuptools = build-system.setuptools;

    fetchPypi = python.pkgs.fetchPypi;
in
# robosuite, minus opencv-python dependency
buildPythonPackage rec {
    pname = "robosuite";
    version = "1.4.1";
    format= "setuptools";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-4gmw94IbuEsr83Y662T6lCrZ1YoVGe6/s5+7aCMeu0I=";
    };

    # patch to remove opencv-python dependency
    # and make the dependency optional instead in the source
    postPatch = ''
        substituteInPlace "setup.py" \
            --replace-fail '"opencv-python",' ' '
        substituteInPlace "robosuite/renderers/nvisii/nvisii_renderer.py" \
            --replace-fail 'import cv2' ' '
        substituteInPlace "robosuite/renderers/nvisii/nvisii_renderer.py" \
            --replace-fail 'if video_mode:' 'if video_mode:
                    import cv2'
        substituteInPlace "robosuite/renderers/nvisii/nvisii_renderer.py" \
            --replace-fail 'if self.video_mode:' 'if self.video_mode:
                    import cv2'

        substituteInPlace "robosuite/utils/opencv_renderer.py" \
            --replace-fail 'import cv2' ' '
        substituteInPlace "robosuite/utils/opencv_renderer.py" \
            --replace-fail 'def render(self):' 'def render(self):
                import cv2'
        substituteInPlace "robosuite/utils/opencv_renderer.py" \
            --replace-fail 'def close(self):' 'def close(self):
                import cv2'
        
        substituteInPlace "robosuite/renderers/context/egl_context.py" \
            --replace-fail 'from mujoco.egl import egl_ext as EGL' \
            'from OpenGL import EGL'
        substituteInPlace "robosuite/renderers/context/egl_context.py" \
            --replace-fail 'def create_initialized_egl_device_display(device_id=0):' \
            'def create_initialized_egl_device_display(device_id=0):
            from mujoco.egl import egl_ext as EGL'
    '';

    dependencies = [mujoco numba numpy pillow pynput scipy termcolor];
    build-system = [setuptools];

    doCheck = false;
}
