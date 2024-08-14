{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    setuptools-rust = build-system.setuptools-rust;
    tomli = build-system.tomli;
    fetchPypi = python.pkgs.fetchPypi;
in
buildPythonPackage rec {
    pname = "maturin";
    version = "1.7.0";
    format="pyproject";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-G6UnfdeDLcYYHWmgBRgrl7NSCUWCUFhIT/2SlvLvtZw=";
    };
    build-system = [setuptools setuptools-rust];
    dependencies = [tomli];
}
