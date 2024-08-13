{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    wheel = build-system.wheel;
    pyobjc-core = dependencies.pyobjc-core;
    Foundation = nixpkgs.swiftPackages.Foundation;
    AVFoundation = nixpkgs.darwin.apple_sdk.frameworks.AVFoundation;
    GameKit = nixpkgs.darwin.apple_sdk.frameworks.GameKit;
    MetalPerformanceShaders = nixpkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders;
    Cocoa = nixpkgs.darwin.apple_sdk.frameworks.Cocoa;
    isDarwin = nixpkgs.stdenv.isDarwin;
in
buildPythonPackage {
    pname = "pyobjc-framework-cocoa";
    version = "9.2";
    format="pyproject";

    disabled = !isDarwin;
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/38/91/c54fdffda6d7cfad67ff617f19001163658d50bc72376d1584e691cf4895/pyobjc-framework-Cocoa-9.2.tar.gz";
        hash="sha256-79eAgIctjI3mwrl+Dk6smdYgOl0WN6oTXQcdRk6y21M=";
    };
    build-system = [setuptools wheel];
    dependencies = [pyobjc-core];

    preBuild=''                                                                                                               
            export CC=/usr/bin/clang
            export CXX=/usr/bin/clang
    '';
    buildInputs = [
        Foundation AVFoundation GameKit
        MetalPerformanceShaders Cocoa
        nixpkgs.libffi
    ];
}