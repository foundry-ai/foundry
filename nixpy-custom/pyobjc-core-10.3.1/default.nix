{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    wheel = build-system.wheel;
    Foundation = nixpkgs.swiftPackages.Foundation;
    AVFoundation = nixpkgs.darwin.apple_sdk.frameworks.AVFoundation;
    GameKit = nixpkgs.darwin.apple_sdk.frameworks.GameKit;
    MetalPerformanceShaders = nixpkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders;
    isDarwin = nixpkgs.stdenv.isDarwin;
in
buildPythonPackage {
    pname = "pyobjc-core";
    version = "10.3.1";
    format="pyproject";

    disabled = !isDarwin;

    src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/40/a38d78627bd882d86c447db5a195ff307001ae02c1892962c656f2fd6b83/pyobjc_core-10.3.1.tar.gz";
        hash="sha256-sgSoDMwHD5qz+K9COjolpv14fiKFCNAMTDD4rFOLpyA=";
    };
    build-system = [setuptools wheel];

    # ugh, just use the system clang...
    preBuild=''                                                                                                               
            export CC=/usr/bin/clang
            export CXX=/usr/bin/clang
    '';
    buildInputs = [
        Foundation AVFoundation GameKit
        MetalPerformanceShaders
        nixpkgs.libffi
    ];
    doCheck = false;
}