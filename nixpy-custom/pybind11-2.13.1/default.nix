{ buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl }:
let lib = nixpkgs.lib;
    fetchFromGitHub = nixpkgs.fetchFromGitHub;
    cmake = build-system.cmake;
    ninja = build-system.ninja;
    setuptools = build-system.setuptools;
in
buildPythonPackage rec {
  pname = "pybind11";
  version = "2.13.1";
  format = "pyproject";

  src = fetchFromGitHub {
    owner = "pybind";
    repo = "pybind11";
    rev = "v${version}";
    hash = "sha256-sQUq39CmgsDEMfluKMrrnC5fio//pgExcyqJAE00UjU=";
  };

  build-system = [
    cmake
    ninja
    setuptools
  ];

  buildInputs = lib.optionals (pythonOlder "3.9") [ libxcrypt ];
  propagatedNativeBuildInputs = [ setupHook ];

  stdenv = stdenv';

  dontUseCmakeBuildDir = true;

  # Don't build tests if not needed, read the doInstallCheck value at runtime
  preConfigure = ''
    if [ -n "$doInstallCheck" ]; then
      cmakeFlagsArray+=("-DBUILD_TESTING=ON")
    fi
  '';

  cmakeFlags = [
    "-DBoost_INCLUDE_DIR=${lib.getDev boost}/include"
    "-DEIGEN3_INCLUDE_DIR=${lib.getDev eigen}/include/eigen3"
  ] ++ lib.optionals (python.isPy3k && !stdenv.cc.isClang) [ "-DPYBIND11_CXX_STANDARD=-std=c++17" ];

  postBuild = ''
    # build tests
    make -j $NIX_BUILD_CORES
  '';

  postInstall = ''
    make install
    # Symlink the CMake-installed headers to the location expected by setuptools
    mkdir -p $out/include/${python.libPrefix}
    ln -sf $out/include/pybind11 $out/include/${python.libPrefix}/pybind11
  '';
  doCheck = false;
  hardeningDisable = lib.optional stdenv.hostPlatform.isMusl "fortify";

  meta = with lib; {
    homepage = "https://github.com/pybind/pybind11";
    changelog = "https://github.com/pybind/pybind11/blob/${src.rev}/docs/changelog.rst";
    description = "Seamless operability between C++11 and Python";
    mainProgram = "pybind11-config";
    longDescription = ''
      Pybind11 is a lightweight header-only library that exposes
      C++ types in Python and vice versa, mainly to create Python
      bindings of existing C++ code.
    '';
    license = licenses.bsd3;
    maintainers = with maintainers; [
      yuriaisaka
      dotlambda
    ];
  };
}