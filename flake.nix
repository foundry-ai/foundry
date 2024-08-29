{
    inputs = {
        nixpkgs.url = "github:inductive-research/nixpkgs";
    };
    outputs = { self, nixpkgs }:
        let
        # The systems supported for this flake
        supportedSystems = [
            "x86_64-linux" # 64-bit Intel/AMD Linux
            "aarch64-linux" # 64-bit ARM Linux
            "x86_64-darwin" # 64-bit Intel macOS
            "aarch64-darwin" # 64-bit ARM macOS
            "powerpc64le-linux"
        ];

        forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
            pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
        });
        in {
            devShells = forEachSupportedSystem ({ pkgs }: 
                let nixpy-custom = import ./nixpy-custom;
                    py = pkgs.python310; 
                    requirements = (import ./requirements.nix) {
                        buildPythonPackage = py.pkgs.buildPythonPackage;
                        fetchurl = pkgs.fetchurl;
                        nixpkgs = pkgs;
                        python = py;
                        nixpy-custom = nixpy-custom;
                    };
                    allPackages = (with requirements.env; foundry-meta.dependencies ++ 
                                (builtins.foldl' (x: y: x ++ y.dependencies) [] foundry-meta.dependencies)
                    );
                    externalPackages = builtins.filter (x: !(builtins.elem x (with requirements.env; [
                        foundry-meta foundry-models
                        foundry-core foundry-systems
                        policy-eval image-classifier 
                        language-model cond-diffusion-toy
                    ]))) allPackages;
                    pythonEnv = py.withPackages(
                        ps: 
                        with requirements.env; externalPackages
                    );
                    driversHook = (import ./drivers.nix { nixpkgs = pkgs; });
                    hook = ''
                        export TMPDIR=/tmp/$USER-foundry-tmp
                        mkdir -p $TMPDIR
                        FOUNDRY=$(pwd)/packages/foundry/src
                        COND_DIFFUSION=$(pwd)/projects/cond-diffusion/src
                        IMAGE_CLASSIFIER=$(pwd)/projects/image-classifier/src
                        export PYTHONPATH=$FOUNDRY:$COND_DIFFUSION:$IMAGE_CLASSIFIER:$PYTHONPATH
                        export PATH=$(pwd)/scripts:$PATH
                        ${driversHook}

                        export FOUNDRY_PATH=$PATH
                    '';
                in {
                externalPackages = externalPackages;
                default = pkgs.mkShell {
                    packages = with pkgs; [ pythonEnv fish pkgs.glxinfo ffmpeg-headless];
                    # add a PYTHON_PATH to the current directory
                    shellHook = hook + ''
                        export SHELL=$(which fish)
                        exec fish
                    '';
                };
                job = pkgs.mkShell {
                    packages = with pkgs; [ pythonEnv ffmpeg-headless];
                    # add a PYTHON_PATH to the current directory
                    shellHook = hook;
                };
            });
            legacyPackages = forEachSupportedSystem ({ pkgs }:
                let nixpy-custom = import ./nixpy-custom;
                    py = pkgs.python310; 
                    requirements = (import ./requirements.nix) {
                        buildPythonPackage = py.pkgs.buildPythonPackage;
                        fetchurl = pkgs.fetchurl;
                        nixpkgs = pkgs;
                        python = py;
                        nixpy-custom = nixpy-custom;
                    };
                in requirements.env
            );
            packages = forEachSupportedSystem ({ pkgs }:
                let nixpy-custom = import ./nixpy-custom;
                    py = pkgs.python310; 
                    requirements = (import ./requirements.nix) {
                        buildPythonPackage = py.pkgs.buildPythonPackage;
                        fetchurl = pkgs.fetchurl;
                        nixpkgs = pkgs;
                        python = py;
                        nixpy-custom = nixpy-custom;
                    };
                in {
                default = requirements.env.foundry-meta;
            });
        };
}