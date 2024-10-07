{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    numpy = dependencies.numpy;
    scipy = dependencies.scipy;
    ml-dtypes = dependencies.ml-dtypes;

    pybind11 = build-system.pybind11;
    cython = build-system.cython;
    setuptools = build-system.setuptools;
    build = build-system.build;
    wheel = build-system.wheel;
    jaxlib = (import ./common.nix { 
      pkgs = nixpkgs;
      python = python; 
      dependencies = dependencies;
      build-system = build-system;
    });
  in
buildPythonPackage {
  inherit (jaxlib) pname version;
  format = "wheel";
  src = jaxlib.jaxlib-wheel;

  dependencies = [
    ml-dtypes
    numpy
    scipy
  ];

  pythonImportsCheck = [
    "jaxlib"
    # `import jaxlib` loads surprisingly little. These imports are actually bugs that appeared in the 0.4.11 upgrade.
    "jaxlib.cpu_feature_guard"
    "jaxlib.xla_client"
  ];

  passthru = {
    build = jaxlib.jaxlib-wheel;
  };
}