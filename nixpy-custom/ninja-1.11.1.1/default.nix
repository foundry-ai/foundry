{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : buildPythonPackage {
    pname = "ninja";
    version = "1.11.1.1";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/37/2c/d717d13a413d6f7579cdaa1e28e6e2c98de95461549b08d311c8a5bf4c51/ninja-1.11.1.1.tar.gz";
        hash="sha256-nXk7CN2FfjjQtv/p5rcUXXxIWkLc/qBJBcoM22AXzDw=";
    };
    format = "pyproject";
    build-system = [
        build-system.scikit-build 
        build-system.setuptools 
        build-system.setuptools-scm
    ];
    nativeBuildInputs = [nixpkgs.cmake nixpkgs.ninja];
    dontUseCmakeConfigure = true;
    doCheck = false;
}