{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let
  pname = "jax";
  version = "0.4.28";
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
        hash = "sha256-3PCkSv8uFxPworNpKBzVt52MGPwQGJBcQSWJfLBrN+k=";
    };
    build-system = [setuptools];
    dependencies = [numpy scipy opt-einsum ml-dtypes jaxlib];
}