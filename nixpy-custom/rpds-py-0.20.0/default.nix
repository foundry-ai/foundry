{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    fetchPypi = python.pkgs.fetchPypi;
    pythonOlder = python.pkgs.pythonOlder;
    pytestCheckHook = python.pkgs.pytestCheckHook;
    rustPlatform = nixpkgs.rustPlatform;
    cargo = nixpkgs.cargo;
    rustc = nixpkgs.rustc;
    libiconv = nixpkgs.libiconv;
in
buildPythonPackage rec {
    pname = "rpds-py";
    version = "0.20.0";
    format = "pyproject";

    disabled = pythonOlder "3.8";

    src = fetchPypi {
        pname = "rpds_py";
        inherit version;
        hash = "sha256-1yohCCT6z9r4dozy18oloELDAyCzAg3i+gRkCSDU4SE=";
    };

    cargoDeps = rustPlatform.fetchCargoTarball {
        inherit src;
        name = "${pname}-${version}";
        hash = "sha256-5vbR2EbrAPJ8pb78tj/+r9nOWgQDT5aO/LUQI4kAGjU=";
    };

    nativeBuildInputs = [
        rustPlatform.cargoSetupHook
        rustPlatform.maturinBuildHook
        cargo
        rustc
    ];
    buildInputs = lib.optionals stdenv.hostPlatform.isDarwin [ libiconv ];
    nativeCheckInputs = [ pytestCheckHook ];
    pythonImportsCheck = [ "rpds" ];
}