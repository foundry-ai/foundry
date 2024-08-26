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
    ffmpeg = nixpkgs.ffmpeg-headless;

    setuptools = build-system.setuptools;
    wheel = build-system.wheel;
    fetchPypi = python.pkgs.fetchPypi;

    ffmpegio-core = dependencies.ffmpegio-core;
    numpy = dependencies.numpy;
in
buildPythonPackage rec {
  pname = "ffmpegio";
  version = "0.10.0.post0";
  format = "pyproject";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-1ocuTX4otvN7Uqc55lL5ZQtrm/WgjPmWE+WbFFLgDWU=";
  };
  build-system = [ setuptools wheel ];

  dependencies = [
    ffmpegio-core numpy
  ];
  propagatedBuildInputs = [ ffmpeg ];

  postInstall = ''
    mkdir -p $out/bin
    ln -s ${ffmpeg}/bin/ffmpeg $out/bin/ffmpeg
  '';


  pythonImportsCheck = [ "ffmpegio" ];
}
