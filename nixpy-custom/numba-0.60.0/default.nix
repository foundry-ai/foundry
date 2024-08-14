{buildPythonPackage, build-system, dependencies, 
    nixpkgs, python, fetchurl} :
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    pythonOlder = python.pkgs.pythonOlder;
    pythonAtLeast = python.pkgs.pythonAtLeast;
    numpy = dependencies.numpy;
    llvmlite = dependencies.llvmlite;
    setuptools = build-system.setuptools;

    libcxx = nixpkgs.libcxx;

    fetchFromGitHub = nixpkgs.fetchFromGitHub;
    cudaSupport = false;
in
buildPythonPackage rec {
  version = "0.60.0";
  pname = "numba";
  pyproject = true;

  disabled = pythonOlder "3.8" || pythonAtLeast "3.13";

  src = fetchFromGitHub {
    owner = "numba";
    repo = "numba";
    rev = "refs/tags/${version}";
    hash = "sha256-hUL281wHLA7wo8umzBNhiGJikyIF2loCzjLECuC+pO0=";
  };

  postPatch = ''
    substituteInPlace setup.py \
      --replace-fail 'min_numpy_build_version = "2.0.0rc1"' \
            'min_numpy_build_version = "1.22"'
    '';

  env.NIX_CFLAGS_COMPILE = lib.optionalString stdenv.isDarwin "-I${lib.getDev libcxx}/include/c++/v1";

  build-system = [
    setuptools
    numpy
  ];

  dependencies = [
    numpy
    llvmlite
    setuptools
  ];
  doCheck = false;
}