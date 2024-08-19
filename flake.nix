{
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
                    pythonEnv = py.withPackages(
                        ps: 
                        with requirements.env; [stanza-meta]
                    );
                in {
                default = pkgs.mkShell {
                    packages = with pkgs; [ pythonEnv fish ];
                    # add a PYTHON_PATH to the current directory
                    shellHook = ''
                    export TMPDIR=/tmp/$USER-stanza-tmp
                    mkdir -p $TMPDIR
                    exec fish
                    '';
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
        };
}
