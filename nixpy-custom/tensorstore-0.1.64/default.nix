{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
    let pname = "tensorstore";
        version = "0.1.64";
        lib = nixpkgs.lib;
        stdenv = nixpkgs.stdenv;
        setuptools = build-system.setuptools;
        numpy_dep = dependencies.numpy;
        ml-dtypes = dependencies.ml-dtypes;
in
buildPythonPackage rec {
    inherit pname version;
    format = "pyproject";

    src = ./stub;
    build-system = [setuptools];
    dependencies = [numpy_dep ml-dtypes];

    postUnpack = ''
        substituteInPlace "$sourceRoot/pyproject.toml" \
        --subst-var version
    '';
}
