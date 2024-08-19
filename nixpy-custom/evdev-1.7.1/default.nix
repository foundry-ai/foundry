{
  nixpkgs,
  buildPythonPackage,
  build-system,
  dependencies,
  python,
  fetchurl
}:
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    setuptools = build-system.setuptools;
    fetchPypi = python.pkgs.fetchPypi;
    pythonOlder = python.pkgs.pythonOlder;
    linuxHeaders = nixpkgs.linuxHeaders;
    glfw3 = nixpkgs.glfw;
    in
buildPythonPackage rec {
  pname = "evdev";
  version = "1.7.1";
  pyproject = true;
  disabled = pythonOlder "3.7";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-DHLDcL2inYV+GI2TEBnDJlGpweqXfAjI2TmxztFjf94=";
  };
  patchPhase = ''
    substituteInPlace setup.py \
      --replace-fail /usr/include ${linuxHeaders}/include
  '';
  build-system = [ setuptools ];
  buildInputs = [ linuxHeaders ];
  pythonImportsCheck = [ "evdev" ];
}