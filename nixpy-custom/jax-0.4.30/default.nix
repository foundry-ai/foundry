{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let
  pname = "jax";
  version = "0.4.30";
  fetchPypi = python.pkgs.fetchPypi;
  setuptools = build-system.setuptools;
  ml-dtypes = dependencies.ml-dtypes;
  numpy = dependencies.numpy;
  scipy = dependencies.scipy;
  opt-einsum = dependencies.opt-einsum;
  jaxlib = dependencies.jaxlib;
in
buildPythonPackage {
    inherit pname version;
    format = "pyproject";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-lNdLWy2w2AZyth2D8fY+v5nSq3OY7BKyygydHpev5Xc=";
    };
    build-system = [setuptools];
    dependencies = [numpy scipy opt-einsum ml-dtypes jaxlib];
}