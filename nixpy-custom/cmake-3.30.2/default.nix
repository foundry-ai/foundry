{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    setuptools = build-system.setuptools;
    cmake = import ./system/system.nix { nixpkgs = nixpkgs; };
in
buildPythonPackage rec {
  pname = "cmake";
  inherit (cmake) version;
  format = "pyproject";
  src = ./stub;
  postUnpack = ''
    substituteInPlace "$sourceRoot/pyproject.toml" \
      --subst-var version

    substituteInPlace "$sourceRoot/cmake/__init__.py" \
      --subst-var version \
      --subst-var-by CMAKE_BIN_DIR "${cmake}/bin"
  '';
  inherit (cmake) setupHooks;
  nativeBuildInputs = [ setuptools ];
  pythonImportsCheck = [ "cmake" ];
}