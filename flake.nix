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
                        FOUNDRY_CORE=$(pwd)/packages/core
                        FOUNDRY_SYSTEMS=$(pwd)/packages/systems
                        FOUNDRY_MODELS=$(pwd)/packages/models

                        POLICY_EVAL=$(pwd)/projects/policy-eval
                        IMAGE_CLASSIFIER=$(pwd)/projects/image-classifier

                        export PYTHONPATH=$FOUNDRY_CORE/src:$FOUNDRY_SYSTEMS/src:$FOUNDRY_MODELS/src
                        export PYTHONPATH=:$POLICY_EVAL/src:$IMAGE_CLASSIFIER/src:$PYTHONPATH

                        export PATH=$(pwd)/scripts:$POLICY_EVAL/scripts:$PATH:$IMAGE_CLASSIFIER/scripts

                        ${driversHook}

                        export FOUNDRY_PATH=$PATH
                    '';
                in {
                externalPackages = externalPackages;
                default = 
                let
                    fishPrompt = pkgs.writeText "prompt.fish" "
                    # Copy the current `fish_prompt` function as `_old_fish_prompt`.
                    functions -c fish_prompt _old_fish_prompt

                    function fish_prompt
                        # Run the user's prompt first; it might depend on (pipe)status.
                        set -l prompt (_old_fish_prompt)

                        set_color blue;
                        printf '[%s] ' 'foundry'
                        set_color normal;

                        string join -- \\n $prompt # handle multi-line prompts
                    end
                    ";
                in
                pkgs.mkShell {
                    packages = with pkgs; [ pythonEnv fish pkgs.glxinfo ffmpeg-headless];
                    # add a PYTHON_PATH to the current directory
                    shellHook = hook + ''
                        export SHELL=$(which fish)
                        exec fish -C "source ${fishPrompt}"
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
