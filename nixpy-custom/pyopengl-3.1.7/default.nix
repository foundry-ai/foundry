{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    fetchPypi = python.pkgs.fetchPypi;
    lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
in
# robosuite, minus opencv-python dependency
buildPythonPackage rec {
    pname = "PyOpenGL";
    version = "3.1.7";
    format= "setuptools";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-7vMaOIjmmE/U2ObJlhsYTJgTyoJgTTf+PagOsACnbIY=";
    };
    dependencies = [nixpkgs.libGL nixpkgs.libGLU nixpkgs.libglvnd nixpkgs.libglut nixpkgs.gle];
    build-system = [setuptools];

    patchPhase =
    let
      ext = stdenv.hostPlatform.extensions.sharedLibrary;
    in
    lib.optionalString (!stdenv.isDarwin) ''
        # Theses lines are patching the name of dynamic libraries
        # so pyopengl can find them at runtime.
        substituteInPlace OpenGL/platform/glx.py \
            --replace '"OpenGL",' '"${nixpkgs.libGL}/lib/libOpenGL${ext}",' \
            --replace '"GL",' '"${nixpkgs.libGL}/lib/libGL${ext}",' \
            --replace '"GLU",' '"${nixpkgs.libGLU}/lib/libGLU${ext}",' \
            --replace '"GLX",' '"${nixpkgs.libglvnd}/lib/libGLX${ext}",' \
            --replace '"glut",' '"${nixpkgs.libglut}/lib/libglut${ext}",' \
            --replace '"GLESv1_CM",' '"${nixpkgs.libGL}/lib/libGLESv1_CM${ext}",' \
            --replace '"GLESv2",' '"${nixpkgs.libGL}/lib/libGLESv2${ext}",' \
            --replace '"gle",' '"${nixpkgs.gle}/lib/libgle${ext}",' \
            --replace "'EGL'" "'${nixpkgs.libGL}/lib/libEGL${ext}'"
        substituteInPlace OpenGL/platform/egl.py \
            --replace "('OpenGL','GL')" "('${nixpkgs.libGL}/lib/libOpenGL${ext}', '${nixpkgs.libGL}/lib/libGL${ext}')" \
            --replace "'GLU'," "'${nixpkgs.libGLU}/lib/libGLU${ext}'," \
            --replace "'glut'," "'${nixpkgs.libglut}/lib/libglut${ext}'," \
            --replace "'GLESv1_CM'," "'${nixpkgs.libGL}/lib/libGLESv1_CM${ext}'," \
            --replace "'GLESv2'," "'${nixpkgs.libGL}/lib/libGLESv2${ext}'," \
            --replace "'gle'," '"${nixpkgs.gle}/lib/libgle${ext}",' \
            --replace "'EGL'," "'${nixpkgs.libGL}/lib/libEGL${ext}',"
        substituteInPlace OpenGL/platform/darwin.py \
            --replace "'OpenGL'," "'${nixpkgs.libGL}/lib/libGL${ext}'," \
            --replace "'GLUT'," "'${nixpkgs.libglut}/lib/libglut${ext}',"
        ''
        + ''
        # https://github.com/NixOS/nixpkgs/issues/76822
        # pyopengl introduced a new "robust" way of loading libraries in 3.1.4.
        # The later patch of the filepath does not work anymore because
        # pyopengl takes the "name" (for us: the path) and tries to add a
        # few suffix during its loading phase.
        # The following patch put back the "name" (i.e. the path) in the
        # list of possible files.
        substituteInPlace OpenGL/platform/ctypesloader.py \
            --replace "filenames_to_try = [base_name]" "filenames_to_try = [name]"
        '';

    # Need to fix test runner
    # Tests have many dependencies
    # Extension types could not be found.
    # Should run test suite from $out/${python.sitePackages}
    doCheck = false; # does not affect pythonImportsCheck
    # OpenGL looks for libraries during import, making this a somewhat decent test of the flaky patching above.
    pythonImportsCheck = ["OpenGL" "OpenGL.EGL"];
}