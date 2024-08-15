{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let cython = build-system.cython;
    setuptools = build-system.setuptools;
    numpy = dependencies.numpy;
in
 buildPythonPackage {
      pname = "shapely";
      version = "2.0.5";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ad/99/c47247f4d688bbb5346df5ff1de5d9792b6d95cbbb2fd7b71f45901c1878/shapely-2.0.5.tar.gz";
        hash="sha256-v/I2a8eGv6bLNT1rR9BEPFcMMndmEuUn7ke232P8/jI=";
      };
      preCheck = ''
        cd $out
      '';
      build-system = [cython numpy setuptools];
      dependencies = [numpy];
      buildInputs = [nixpkgs.geos];
      nativeBuildInputs = [cython numpy setuptools nixpkgs.geos];
}