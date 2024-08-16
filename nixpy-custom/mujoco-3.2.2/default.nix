{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    mujoco = (import ./mujoco.nix) { nixpkgs = nixpkgs; };
    perl = nixpkgs.perl;
    fetchPypi = python.pkgs.fetchPypi;

    setuptools = build-system.setuptools;
    cmake = nixpkgs.cmake;
    pybind11 = build-system.pybind11;

    etils = dependencies.etils;
    glfw = dependencies.glfw;
    numpy = dependencies.numpy;
    pyopengl = dependencies.pyopengl;
    absl-py = dependencies.absl-py;

    ApplicationServices = nixpkgs.darwin.apple_sdk.frameworks.ApplicationServices;
    Cocoa = nixpkgs.darwin.apple_sdk.frameworks.Cocoa;
    pythonVersionMajorMinor =
        with lib.versions;
        "${major python.pythonVersion}${minor python.pythonVersion}";
    cpuPname = stdenv.hostPlatform.parsed.cpu.name;
    cpuName = if cpuPname == "x86_64" then "x86" else
              if cpuPname == "aarch64" then "arm64" else
              if cpuPname == "ppc64le" then "ppc64" else
              cpuPname;
    platform = with stdenv.hostPlatform.parsed; 
            if stdenv.isDarwin then "macosx-11.0-${cpuName}" 
            else "${kernel.name}-${cpu.name}";
in
buildPythonPackage rec {
    pname = "mujoco";
    inherit (mujoco) version;

    pyproject = true;

    # We do not fetch from the repository because the PyPi tarball is
    # impurely build via
    # <https://github.com/google-deepmind/mujoco/blob/main/python/make_sdist.sh>
    # in the project's CI.
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-HDN6KA2JYDZqad/MybsX1oV/VvXPshb0mi0UBZq47Qs=";
    };
    # patches = [ 
    #     ./mujoco-system-deps-dont-fetch.patch 
    #     ./external-glfw.patch
    # ];

    mujocoBuildInputs = [
    ] ++ lib.optionals stdenv.isDarwin [
        ApplicationServices
        Cocoa
    ];
    preConfigure = ''
        ${perl}/bin/perl -0777 -i -pe "s/GIT_REPO\n.*\n.*GIT_TAG\n.*\n//gm" mujoco/CMakeLists.txt
        ${perl}/bin/perl -0777 -i -pe "s/(FetchContent_Declare\(\n.*lodepng\n.*)(GIT_REPO.*\n.*GIT_TAG.*\n)(.*\))/\1\3/gm" mujoco/simulate/CMakeLists.txt
        substituteInPlace mujoco/simulate/cmake/SimulateDependencies.cmake \
            --replace-fail "if(NOT SIMULATE_STANDALONE)" "if (NO)"
        build=build/temp.${platform}-cpython-${pythonVersionMajorMinor}/
        mkdir -p $build/_deps
        ln -s ${mujoco.pin.abseil-cpp} $build/_deps/abseil-cpp-src
        ln -s ${mujoco.pin.eigen3} $build/_deps/eigen-src
        ln -s ${mujoco.pin.lodepng} $build/_deps/lodepng-src
    '';

    nativeBuildInputs = [
        perl
        cmake
        setuptools
        nixpkgs.git
    ];

    dontUseCmakeConfigure = true;

    buildInputs = [
        mujoco
        # use nixpkgs' pybind11 since
        # this handles some cmake stuff better
        python.pkgs.pybind11
    ] ++ mujocoBuildInputs;
    propagatedBuildInputs = [
        absl-py
        etils
        glfw
        numpy
        pyopengl
    ];

    pythonImportsCheck = [ "${pname}" ];

    env.MUJOCO_PATH = "${mujoco}";
    env.MUJOCO_PLUGIN_PATH = "${mujoco}/lib";
    env.MUJOCO_CMAKE_ARGS = lib.concatStringsSep " " [
        "-DMUJOCO_SIMULATE_USE_SYSTEM_GLFW=ON"
        "-DMUJOCO_PYTHON_USE_SYSTEM_PYBIND11=ON"
    ];
}
