{buildPythonPackage, build-system, dependencies, 
    nixpkgs, python, fetchurl} : 
    let setuptools = build-system.setuptools;
        cython = build-system.cython;
        fetchPypi = python.pkgs.fetchPypi;
        lib = nixpkgs.lib;
        licenses = nixpkgs.licenses;
        pythonOlder = python.pkgs.pythonOlder;
        in
buildPythonPackage rec {
  pname = "msgpack";
  version = "1.0.8";
  format = "setuptools";
  disabled = pythonOlder "3.6";
  src = fetchPypi {
    inherit pname version;
    hash = "sha256-lcArDifnBuSNDlQm0XEMp44PBijW6J1bWluRpfEidPM=";
  };
  nativeBuildInputs = [ setuptools cython ];
  pythonImportsCheck = [ "msgpack" ];
}