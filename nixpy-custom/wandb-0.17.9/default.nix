{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
    let pname = "wandb";
        version = "0.17.9";

        lib = nixpkgs.lib;
        git = nixpkgs.git;
        substituteAll = nixpkgs.substituteAll;

        hatchling = build-system.hatchling;
        typing-extensions = build-system.typing-extensions;

        click = dependencies.click;
        gitpython = dependencies.gitpython;
        requests = dependencies.requests;
        psutil = dependencies.psutil;
        sentry-sdk = dependencies.sentry-sdk;
        docker-pycreds = dependencies.docker-pycreds;
        setuptools = dependencies.setuptools;
        protobuf = dependencies.protobuf;
        pyyaml = dependencies.pyyaml;
        setproctitle = dependencies.setproctitle;
        platformdirs = dependencies.platformdirs;

        src = nixpkgs.fetchFromGitHub {
            owner="wandb";
            repo="wandb";
            rev = "refs/tags/v${version}";
            hash = "sha256-GHHM3PAGhSCEddxfLGU/1PWqM4WGMf0mQIKwX5ZVIls=";
        };
        goModule = nixpkgs.buildGoModule {
            name = "wandb-core";
            src = "${src}/core";
            nativeBuildInputs = [git];
            vendorHash = null;
        };
    in
buildPythonPackage {
    inherit pname version;
    format = "pyproject";
    src = src;
    patches = [
        # Replace git paths
        # (substituteAll {
        #     src = ./hardcode-git-path.patch;
        #     git = "${lib.getBin git}/bin/git";
        # })
    ];
    doCheck = false;
    build-system = [hatchling typing-extensions];
    buildInputs = [goModule];
    dependencies = [
        click gitpython requests
        psutil sentry-sdk docker-pycreds
        setuptools protobuf pyyaml
        setproctitle platformdirs
    ];
    pythonImportsCheck = [ "wandb" ];
}
