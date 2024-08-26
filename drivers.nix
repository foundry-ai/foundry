{nixpkgs}:
let 

lib = nixpkgs.lib;
driverConfig = lib.importJSON ./.drivers/config.json;

    cudaLib = ./.drivers/${driverConfig.cuda.path};
    cudaDrivers = lib.optionals (builtins.hasAttr "cuda" driverConfig) [
            nixpkgs.runCommand "cuda-drivers" {} (''
            mkdir -p $out $out
            cp ${cudaLib} $out/lib/libcuda.so
            ln -s $out/lib/libcuda.so $out/lib/libcuda.so.1
        '')
    ];
    glDrivers = builtins.map (
        driver: nixpkgs.runCommand "driver-${driver.name}"
    ) driverConfig.drivers;

drivers = [];
driverLibPaths = lib.makeLibraryPath drivers;
vendorPaths = lib.strings.concatStringsSep ":" (map (driver: "${driver}/vendor.json") glDrivers)
in
''
    export LD_LIBRARY_PATH=${driverLibPaths}
    export __EGL_VENDOR_LIBRARY_FILENAMES=${vendorPaths}
''