{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
    let lib = nixpkgs.lib;
        hdf5 = nixpkgs.hdf5;
        pkgconfig = build-system.pkgconfig;
        setuptools = build-system.setuptools;
        numpy = build-system.numpy;
        cython = python.pkgs.cython;
        stdenv = nixpkgs.stdenv;
        fetchPypi = python.pkgs.fetchPypi;
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

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-e36PeAcqLt7IfJg28l80ID/UkqRHVwmhi0F6M8+yH6k=";
  };

  patches = [
    # Unlock an overly strict locking of mpi4py version (seems not to be necessary).
    # See also: https://github.com/h5py/h5py/pull/2418/files#r1589372479
    ./mpi4py-requirement.patch
  ];

  # avoid strict pinning of numpy, can't be replaced with pythonRelaxDepsHook,
  # see: https://github.com/NixOS/nixpkgs/issues/327941
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail "numpy >=2.0.0rc1" "numpy"
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