{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
    let setuptools = build-system.setuptools;
        numpy = dependencies.numpy;
        pytestCheckHook = python.pkgs.pytestCheckHook;
        fetchFromGitHub = nixpkgs.fetchFromGitHub;
        lib = nixpkgs.lib;
        stdenv = nixpkgs.stdenv;
        libcxx = nixpkgs.libcxx;
    in
        buildPythonPackage rec {
    name = "ml-dtypes";
    version = "0.4.0";
    # src = fetchurl {
    #     url="https://github.com/jax-ml/ml_dtypes/archive/b157c19cc98da40a754109993e02d7eab3d75358.zip";
    #     hash="sha256-7pfKaYnNLWw8ZHaXWX6EprdQru4/OdApfuepJiNvoEg=";
    # };
    src = fetchFromGitHub {
        owner = "jax-ml";
        repo = "ml_dtypes";
        rev = "refs/tags/v${version}";
        hash = "sha256-3qZ1lS1IdSXNLRNE9tyuO9qauVBDlECZvmmwaOffD30=";
        # Since this upstream patch (https://github.com/jax-ml/ml_dtypes/commit/1bfd097e794413b0d465fa34f2eff0f3828ff521),
        # the attempts to use the nixpkgs packaged eigen dependency have failed.
        # Hence, we rely on the bundled eigen library.
        fetchSubmodules = true;
    };
    format = "pyproject";

    postPatch = ''
    substituteInPlace pyproject.toml \
      --replace "numpy~=1.21.2" "numpy" \
      --replace "numpy~=1.23.3" "numpy" \
      --replace "numpy~=1.26.0" "numpy" \
      --replace "numpy==2.0.0rc1" "numpy" \
      --replace "setuptools~=68.1.0" "setuptools"
    '';

    env.NIX_CFLAGS_COMPILE = lib.optionalString stdenv.isDarwin "-I${lib.getDev libcxx}/include/c++/v1";

    nativeBuildInputs = [ setuptools ];
    propagatedBuildInputs = [ numpy ];
}
