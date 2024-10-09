{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let
  pname = "jax-cuda12-prjt";
  version = "0.4.34";
  fetchPypi = python.pkgs.fetchPypi;
  setuptools = build-system.setuptools;
  ml-dtypes = dependencies.ml-dtypes;
  numpy = dependencies.numpy;
  scipy = dependencies.scipy;
  opt-einsum = dependencies.opt-einsum;
  jaxlib = (import ../jaxlib-${version}/common.nix {
    inherit python dependencies build-system;
    pkgs = nixpkgs;
  });
in
buildPythonPackage {
    inherit pname version;
    format = "wheel";
    src = jaxlib.cuda-prjt-wheel;
    build-system = [setuptools];
    dependencies = [numpy scipy ml-dtypes];
}