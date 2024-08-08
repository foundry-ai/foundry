{
  nixpkgs,
  buildPythonPackage,
  build-system,
  dependencies,
  python,
  fetchurl
}:
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    in
buildPythonPackage rec {
  pname = "kiwisolver";
  version = "1.4.5";
  format = "setuptools";

  src = python.pkgs.fetchPypi {
    inherit pname version;
    hash = "sha256-5X5WOlf7IqFC2jTziswvwaXIZLwpyhUXqIq8lj5g1uw=";
  };

  env.NIX_CFLAGS_COMPILE = lib.optionalString stdenv.isDarwin "-I${lib.getDev nixpkgs.libcxx}/include/c++/v1";

  nativeBuildInputs = [ build-system.setuptools-scm ];

  buildInputs = [ build-system.cppy ];

  pythonImportsCheck = [ "kiwisolver" ];

  meta = with lib; {
    description = "Implementation of the Cassowary constraint solver";
    homepage = "https://github.com/nucleic/kiwi";
    license = licenses.bsd3;
    maintainers = [ ];
  };
}