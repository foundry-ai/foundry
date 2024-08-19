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
    glfw3 = nixpkgs.glfw;
    in
buildPythonPackage rec {
  pname = "glfw";
  version = "2.7.0";
  format = "setuptools";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-DiCa04+oxb5nylkNexdTPZWtHrV9Cj8HuYEx22m3kAA=";
  };
  # patch to use nix glfw library
  postPatch = ''
    substituteInPlace glfw/library.py --replace "_get_library_search_paths()," "[ '${glfw3}/lib' ],"
  '';

  build-system = [ setuptools ];
  propagatedBuildInputs = [ glfw3 ];

  pythonImportsCheck = [ "glfw" ];
}
