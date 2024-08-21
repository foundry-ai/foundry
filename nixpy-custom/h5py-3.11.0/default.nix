{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
    let lib = nixpkgs.lib;
        hdf5 = nixpkgs.hdf5;
        pkgconfig = build-system.pkgconfig;
        setuptools = build-system.setuptools;
        numpy = build-system.numpy;
        cython = build-system.cython;
        stdenv = nixpkgs.stdenv;
        fetchFromGitHub = nixpkgs.fetchFromGitHub;
        pythonOlder = python.pkgs.pythonOlder;
        pythonRelaxDepsHook = python.pkgs.pythonRelaxDepsHook;
        mpi = hdf5.mpi;
        mpiSupport = hdf5.mpiSupport;
    in
buildPythonPackage rec {
  version = "3.11.0";
  pname = "h5py";
  format = "pyproject";

  disabled = pythonOlder "3.7";

  src = fetchFromGitHub {
    owner = "h5py";
    repo = "h5py";
    rev = "d9396a4ee16ecefd2800f83746ebeb6ee4c4930d";
    hash = "sha256-CvPIG9UH1KBTw+7n1n+Q4hQH0SCX00zw16bCS846x3k=";
  };

  # avoid strict pinning of numpy, can't be replaced with pythonRelaxDepsHook,
  # see: https://github.com/NixOS/nixpkgs/issues/327941
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail "numpy >=2.0.0, <3" "numpy"
    substituteInPlace setup.py \
      --replace-fail "mpi4py ==3.1.1" "mi4py >=3.1.1"
    substituteInPlace setup.py \
      --replace-fail "mpi4py ==3.1.4" "mi4py >=3.1.4"
    substituteInPlace setup.py \
      --replace-fail "mpi4py ==3.1.6" "mi4py >=3.1.6"
  '';
  pythonRelaxDeps = [ "mpi4py" ];

  HDF5_DIR = "${hdf5}";
  HDF5_MPI = if mpiSupport then "ON" else "OFF";

  postConfigure = ''
    # Needed to run the tests reliably. See:
    # https://bitbucket.org/mpi4py/mpi4py/issues/87/multiple-test-errors-with-openmpi-30
    ${lib.optionalString mpiSupport "export OMPI_MCA_rmaps_base_oversubscribe=yes"}
  '';

  preBuild = lib.optionalString mpiSupport "export CC=${lib.getDev mpi}/bin/mpicc";

  nativeBuildInputs = [
    pythonRelaxDepsHook
    cython
    pkgconfig
    setuptools
  ];

  buildInputs = [ hdf5 ];
  propagatedBuildInputs = [ numpy ];

  # https://github.com/NixOS/nixpkgs/issues/255262
  preCheck = ''
    cd $out
  '';
  pythonImportsCheck = [ "h5py" ];
}