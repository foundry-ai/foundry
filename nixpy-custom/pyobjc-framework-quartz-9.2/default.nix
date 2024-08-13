{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    wheel = build-system.wheel;
    pyobjc-core = dependencies.pyobjc-core;
    pyobjc-framework-cocoa = dependencies.pyobjc-framework-cocoa;
    Foundation = nixpkgs.swiftPackages.Foundation;
    AVFoundation = nixpkgs.darwin.apple_sdk.frameworks.AVFoundation;
    GameKit = nixpkgs.darwin.apple_sdk.frameworks.GameKit;
    MetalPerformanceShaders = nixpkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders;
    Quartz = nixpkgs.darwin.apple_sdk.frameworks.Quartz;
    isDarwin = nixpkgs.stdenv.isDarwin;
in
buildPythonPackage {
    pname = "pyobjc-framework-quartz";
    version = "9.2";
    format="pyproject";

    disabled = !isDarwin;

    src = fetchurl {
        url="https://files.pythonhosted.org/packages/49/52/a56bbd76bba721f49fa07d34ac962414b95eb49a9b941fe4d3761f3e6934/pyobjc-framework-Quartz-9.2.tar.gz";
        hash="sha256-9YYYO5ue9/Fl8ERKe3FO2WXXn26SYXyq+GkVDc/Vpys=";
    };
    build-system = [setuptools wheel];
    dependencies = [pyobjc-core pyobjc-framework-cocoa];

    buildInputs = [
        Foundation AVFoundation GameKit
        MetalPerformanceShaders Quartz
        nixpkgs.libffi
    ];
}