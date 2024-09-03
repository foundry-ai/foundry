{nixpkgs}:
let 
    lib = nixpkgs.lib;

    # The hashes for the driver files
    nvidiaHashes = {
        "powerpc64le-linux" = {
            "535.86.10" = "sha256-UDieX7LzPTG25Mq3BbQm3tNMkEewiYm5CSVfPgtndaI=";
        };
        "x86_64-linux" = {
            "535.183.01" = "sha256-9nB6+92pQH48vC5RKOYLy82/AvrimVjHL6+11AXouIM=";
            "555.42.02" = "sha256-k7cI3ZDlKp4mT46jMkLaIrc2YUx1lh1wj/J4SVSHWyk=";
        };
    };
    makeNvidiaUrl = { platform, version }: 
        if platform == "x86_64-linux" then 
            "https://us.download.nvidia.com/XFree86/Linux-x86_64/${version}/NVIDIA-Linux-x86_64-${version}.run"
        else if nixpkgs.system == "powerpc64le-linux" then 
            "https://us.download.nvidia.com/tesla/${version}/NVIDIA-Linux-ppc64le-${version}.run"
        else throw "Unsupported platform";

    makeNvidiaPackages = { version }: rec {
        nvidiaDrivers = (nixpkgs.linuxPackages.nvidia_x11.override { disable32Bit = true; }).overrideAttrs
        (oldAttrs: rec {
            pname = "nvidia";
            name = "nvidia-x11-${version}-drivers";
            inherit version;
            src = let
                url = makeNvidiaUrl { platform = nixpkgs.system; version = version; };
                sha256 = nvidiaHashes."${nixpkgs.system}"."${version}";
            in nixpkgs.fetchurl { inherit url sha256; };
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
    driverConfig = lib.importJSON ./.driver_config.json;
in if ((!lib.pathExists ./.driver_config.json) || (driverConfig.nvidia_version == "none")) then ""
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
    export DRIVER="nvidia"
    export __EGL_VENDOR_LIBRARY_FILENAMES=''${NVIDIA_JSON[*]}
''
