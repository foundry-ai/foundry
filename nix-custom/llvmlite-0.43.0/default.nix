{buildPythonPackage, env, nixpkgs, python, fetchurl} : buildPythonPackage {
    pname = "llvmlite";
    version = "0.43.0";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/9f/3d/f513755f285db51ab363a53e898b85562e950f79a2e6767a364530c2f645/llvmlite-0.43.0.tar.gz";
        hash="sha256-ritbXD72c1SCT7dVF8jbX76TvALNlnHzxiJxYmvAQdU=";
    };
    build-system = [];
    preConfigure = ''
        export LLVM_CONFIG=${nixpkgs.llvmPackages_14.llvm.dev}/bin/llvm-config
    '';
    nativeBuildInputs = [
        nixpkgs.llvmPackages_14.llvm
    ];
    doCheck = false;
}