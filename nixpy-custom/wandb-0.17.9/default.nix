{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
    let pname = "wandb";
        version = "0.17.9";

        lib = nixpkgs.lib;
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
        commitSha = "a4d1db2687b9f18e8cd45e96972c619241edbddd";
        wandbCore = nixpkgs.buildGoModule {
            pname = "wandb-core";
            inherit version;
            src = src + "/core";
            vendorHash = null;
            nativeBuildInputs = [ nixpkgs.git ];
            ldflags = [
                "-X main.commit='${commitSha}'"
            ];
            subPackages = [ "cmd/wandb-core" ];
        };
        nvidiaGpuStats = nixpkgs.rustPlatform.buildRustPackage {
            pname = "nvidia_gpu_stats";
            version = "0.1.2";
            src = src + "/nvidia_gpu_stats";
            cargoHash = "sha256-Vy60HMTspQvfWu2hcNjoBwEzi6YwdE3a3trKorDHxVI=";
        };
        nvidiaGpuStatsHatch = ./nvidia-gpu-stats-hatch.py;
        coreHatch = ./core-hatch.py;
    in
buildPythonPackage {
    inherit pname version;
    format = "pyproject";
    src = src;
    postPatch = ''
        cp ${nvidiaGpuStatsHatch} nvidia_gpu_stats/hatch.py
        cp ${coreHatch} core/hatch.py
        substituteInPlace nvidia_gpu_stats/hatch.py --replace "@binary_path@" "${nvidiaGpuStats}/bin/nvidia_gpu_stats"
        substituteInPlace core/hatch.py --replace "@binary_path@" "${wandbCore}/bin/wandb-core"
    '';

    doCheck = false;
    build-system = [hatchling typing-extensions];
    nativeBuildInputs = [nixpkgs.go nixpkgs.cargo];
    dependencies = [
        click gitpython requests
        psutil sentry-sdk docker-pycreds
        setuptools protobuf pyyaml
        setproctitle platformdirs
    ];
    pythonImportsCheck = [ "wandb" ];
    # wandb needs to access shell PATH
    dontWrapPythonPrograms = true;

    # For the go build.
    preConfigure = (''
      export GOCACHE=$TMPDIR/go-cache
      export GOPATH="$TMPDIR/go"
      export GOPROXY=off
      export GOSUMDB=off
      # currently pie is only enabled by default in pkgsMusl
      # this will respect the `hardening{Disable,Enable}` flags if set
      if [[ $NIX_HARDENING_ENABLE =~ "pie" ]]; then
        export GOFLAGS="-buildmode=pie $GOFLAGS"
      fi
    '');
}
