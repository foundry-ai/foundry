{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let scikit-build-core = build-system.scikit-build-core;
    fetchPypi = python.pkgs.fetchPypi;
in
buildPythonPackage rec {
    pname = "cmake";
    version = "3.30.2";
    format = "pyproject";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-VNupjBLGt3vYa0Urccf387BAJwgfNFHhjN8tkm5GleU=";
    };
    build-system = [scikit-build-core];
    nativeBuildInputs = [nixpkgs.cmake nixpkgs.ninja];
    dependencies = [];
    # remove the check for macOS version (it's broken)
    postPatch = ''
        substituteInPlace CMakeLists.txt \
         --replace-fail 'message(FATAL_ERROR "Unsupported macOS deployment target: ''${CMAKE_OSX_DEPLOYMENT_TARGET} is less than 10.10")' \
                        'set(binary_archive "macos10_10_binary")'
    '';
    dontUseCmakeConfigure = true;
}