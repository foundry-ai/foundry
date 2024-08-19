{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let ninja = (import ./system/system.nix {nixpkgs = nixpkgs; });
    lib = nixpkgs.lib;
    setuptools = build-system.setuptools;
in
buildPythonPackage rec {
  pname = "ninja";
  version = "1.12.1.1";
  # inherit (ninja) version;
  format = "pyproject";

  src = ./stub;

  postUnpack = ''
    substituteInPlace "$sourceRoot/pyproject.toml" \
      --subst-var version

    substituteInPlace "$sourceRoot/ninja/__init__.py" \
      --subst-var version \
      --subst-var-by BIN_DIR "${ninja}/bin"
  '';

  nativeBuildInputs = [ setuptools ];
  propagatedBuildInputs = [ ninja ];

  preBuild = ''
    cp "${ninja.src}/misc/ninja_syntax.py" ninja/ninja_syntax.py
  '';

  pythonImportsCheck = [
    "ninja"
    "ninja.ninja_syntax"
  ];

  meta = with lib; {
    description = "Small build system with a focus on speed";
    mainProgram = "ninja";
    longDescription = ''
      This is a stub of the ninja package on PyPI that uses the ninja program
      provided by nixpkgs instead of downloading ninja from the web.
    '';
    homepage = "https://github.com/scikit-build/ninja-python-distributions";
    license = licenses.asl20;
  };
}