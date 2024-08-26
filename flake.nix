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
                    allPackages = (with requirements.env; stanza-meta.dependencies ++ 
                                (builtins.foldl' (x: y: x ++ y.dependencies) [] stanza-meta.dependencies)
                    );
                    externalPackages = builtins.filter (x: !(builtins.elem x (with requirements.env; [
                        stanza-meta stanza-models stanza cond-diffusion 
                        image-classifier language-model cond-diffusion-toy
                    ]))) allPackages;
                    pythonEnv = py.withPackages(
                        ps: 
                        with requirements.env; externalPackages
                    );
                    lib = pkgs.lib;
                    libglvnd = pkgs.libglvnd;
                    mesa = pkgs.mesa;
                    glxindirect = pkgs.runCommand "mesa_glxindirect" { } (''
                        mkdir -p $out/lib
                        ln -s ${mesa.drivers}/lib/libGLX_mesa.so.0 $out/lib/libGLX_indirect.so.0
                    '');

                    driverPath = lib.makeLibraryPath [mesa.drivers];
                    libvdpauPath = lib.makeSearchPathOutput "lib" "lib/vdpau" [pkgs.libvdpau-va-gl];
                    libglvndPath = lib.makeLibraryPath [libglvnd];

                    hook = ''
                        export TMPDIR=/tmp/$USER-stanza-tmp
                        mkdir -p $TMPDIR
                        STANZA=$(pwd)/packages/stanza/src
                        COND_DIFFUSION=$(pwd)/projects/cond-diffusion/src
                        IMAGE_CLASSIFIER=$(pwd)/projects/image-classifier/src

                        export PYTHONPATH=$STANZA:$COND_DIFFUSION:$IMAGE_CLASSIFIER:$PYTHONPATH
                        export PATH=$(pwd)/scripts:$PATH

                        # set up the cuda drivers
                        mkdir -p .drivers
                        if [ ! -L .drivers/libcuda.so ]; then
                            ln -s /usr/lib64/libcuda.so .drivers/libcuda.so
                        fi
                        if [ ! -L .drivers/libcuda.so.1 ]; then
                            ln -s /usr/lib64/libcuda.so.1 .drivers/libcuda.so.1
                        fi
                        export LIBGL_DRIVERS_PATH=${lib.makeSearchPathOutput "lib" "lib/dri" [mesa.drivers]}
                        export LIBVA_DRIVERS_PATH=${lib.makeSearchPathOutput "out" "lib/dri" [mesa.drivers]}
                        export __EGL_VENDOR_LIBRARY_FILENAMES=${mesa.drivers}/share/glvnd/egl_vendor.d/50_mesa.json
                        export LD_LIBRARY_PATH=${driverPath}:${libvdpauPath}:${glxindirect}/lib:${libglvndPath}:$(pwd)/.drivers
                    '';
                in {
                externalPackages = externalPackages;
                default = pkgs.mkShell {
                    packages = with pkgs; [ pythonEnv fish ffmpeg-headless pkgs.glxinfo ];
                    # add a PYTHON_PATH to the current directory
                    shellHook = hook + ''
                        export SHELL=$(which fish)
                        exec fish
                    '';
                };
                job = pkgs.mkShell {
                    packages = with pkgs; [ pythonEnv ];
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
		    default = requirements.env.stanza-meta;
		}
            );
        };
}
