{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : buildPythonPackage {
    name = "jaxlib";
    version = "0.4.30";
    src = fetchurl {
        url="https://github.com/google/jax/archive/refs/tags/jaxlib-v0.4.30.tar.gz";
        hash="sha256-DvljXHNNm7tE/Mh99PHDzM4c/P0kNXLIDTb834Jv4eY=";
    };
    doCheck = false;
}
