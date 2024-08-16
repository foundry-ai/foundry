{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 

let
    ninjaSrc = fetchurl {
        url = "https://github.com/Kitware/ninja/archive/v1.11.1.g95dee.kitware.jobserver-1.tar.gz";
        hash = "sha256-e6hFUfWzFbQnDcfFGt713/g6IVSjZlpsl0QkXBIt0Ns=";
    };
in
buildPythonPackage {
    pname = "ninja";
    version = "1.11.1.1";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/37/2c/d717d13a413d6f7579cdaa1e28e6e2c98de95461549b08d311c8a5bf4c51/ninja-1.11.1.1.tar.gz";
        hash="sha256-nXk7CN2FfjjQtv/p5rcUXXxIWkLc/qBJBcoM22AXzDw=";
    };
    format = "pyproject";
    build-system = [
        build-system.scikit-build 
        build-system.setuptools 
        build-system.setuptools-scm
    ];
    nativeBuildInputs = [nixpkgs.cmake nixpkgs.ninja];
    # set the source directory for ninja
    preConfigure = ''
        mkdir -p Ninja-src
        tar xf  ${ninjaSrc} --strip-components=1 -C Ninja-src
        ls -la Ninja-src
    '';
    postPatch = ''
        substituteInPlace CMakeLists.txt --replace-fail \
            "DOWNLOAD_DIR ''${ARCHIVE_DOWNLOAD_DIR}" ""
        substituteInPlace CMakeLists.txt --replace-fail \
            'URL ''${''${src_archive}_url}' ' '
        substituteInPlace CMakeLists.txt --replace-fail \
            'URL_HASH SHA256=''${''${src_archive}_sha256}' ' '
        # Make not build in source
        substituteInPlace CMakeLists.txt --replace-fail \
            "BUILD_IN_SOURCE 1" "BUILD_IN_SOURCE 0"
    '';
    dontUseCmakeConfigure = true;
    doCheck = false;
}