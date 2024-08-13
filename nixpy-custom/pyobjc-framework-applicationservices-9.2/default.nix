{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    wheel = build-system.wheel;
    pyobjc-core = dependencies.pyobjc-core;
    pyobjc-framework-cocoa = dependencies.pyobjc-framework-cocoa;
    pyobjc-framework-quartz = dependencies.pyobjc-framework-quartz;
    Foundation = nixpkgs.swiftPackages.Foundation;
    AVFoundation = nixpkgs.darwin.apple_sdk.frameworks.AVFoundation;
    GameKit = nixpkgs.darwin.apple_sdk.frameworks.GameKit;
    MetalPerformanceShaders = nixpkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders;
    ApplicationServices = nixpkgs.darwin.apple_sdk.frameworks.ApplicationServices;
    isDarwin = nixpkgs.stdenv.isDarwin;
in
buildPythonPackage {
    pname = "pyobjc-framework-applicationservices";
    version = "9.2";
    format="pyproject";

    disabled = !isDarwin;

    src = fetchurl {
        url="https://files.pythonhosted.org/packages/1f/0d/55aa1ed3b641675992991c1f353d076ddbcb779baccb297292531740dd51/pyobjc-framework-ApplicationServices-9.2.tar.gz";
        hash="sha256-VoyV3RiZtJ+Ir51KLaHpebNp+AjGhUz1LAA13wmiUoo=";
    };
    build-system = [setuptools wheel];
    dependencies =  [
        pyobjc-core
        pyobjc-framework-cocoa
        pyobjc-framework-quartz
    ];
    preBuild=''                                                                                                               
            export CC=/usr/bin/clang
            export CXX=/usr/bin/clang
    '';
    buildInputs = [
        Foundation AVFoundation GameKit
        MetalPerformanceShaders ApplicationServices
        nixpkgs.libffi
    ];
}