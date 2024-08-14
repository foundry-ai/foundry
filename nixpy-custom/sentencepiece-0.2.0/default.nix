{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
in
 buildPythonPackage {
      pname = "sentencepiece";
      version = "0.2.0";
      format="setuptools";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c9/d2/b9c7ca067c26d8ff085d252c89b5f69609ca93fb85a00ede95f4857865d4/sentencepiece-0.2.0.tar.gz";
        hash="sha256-pSwZFx2q8uaX3Gy+Z2hOD6NBsSSJZvauu1Qd5lTRWEM=";
      };
      build-system = [setuptools];
      dependencies = [];
      nativeBuildInputs = [nixpkgs.cmake];
      dontUseCmakeConfigure = true;
}