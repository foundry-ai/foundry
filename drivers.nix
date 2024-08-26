{nixpkgs}:
let 
    lib = nixpkgs.lib;

    # The hashes for the driver files
    nvidiaHashes = {
        "powerpc64le-linux" = {
            "535.86.10" = "sha256-UDieX7LzPTG25Mq3BbQm3tNMkEewiYm5CSVfPgtndaI=";
        };
    };

    makeNvidiaPackages = { version }: rec {
        nvidiaDrivers = (nixpkgs.linuxPackages.nvidia_x11.override { disable32Bit = true; }).overrideAttrs
        (oldAttrs: rec {
            pname = "nvidia";
            name = "nvidia-x11-${version}-drivers";
            inherit version;
            src = nixpkgs.fetchurl {
                url = "https://us.download.nvidia.com/tesla/${version}/NVIDIA-Linux-ppc64le-${version}.run";
                sha256 = nvidiaHashes."${nixpkgs.system}"."${version}";
            };
            useGLVND = true;
            nativeBuildInputs = oldAttrs.nativeBuildInputs or [] ++ [nixpkgs.zstd];

            meta = with lib; {
                platforms = [nixpkgs.system];
            };
        });
        nvidiaLibsOnly = nvidiaDrivers.override {
            libsOnly = true;
            kernel = null;
        };
    };
    driverConfig = lib.importJSON ./.drivers/config.json;
in if ((!lib.pathExists ./.drivers/config.json) || (driverConfig.nvidia_version == "none")) then ""
else let
    nvidiaPackages = makeNvidiaPackages { version = driverConfig.nvidia_version; };
    nvidiaLibsOnly = nvidiaPackages.nvidiaLibsOnly;
    libglvnd = nixpkgs.libglvnd;
in 
''
    NVIDIA_JSON=(${nvidiaLibsOnly}/share/glvnd/egl_vendor.d/*nvidia.json)
    export LD_LIBRARY_PATH=${lib.makeLibraryPath ([ 
        libglvnd nvidiaLibsOnly 
    ])}
    export __EGL_VENDOR_LIBRARY_FILENAMES=''${NVIDIA_JSON[*]}
''