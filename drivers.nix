{nixpkgs}:
let 

lib = nixpkgs.lib;
driverConfig = lib.importJSON ./.drivers/config.json;
cudaPath = ./.drivers/${driverConfig.cuda.path};
cudaDrivers = lib.optionals (builtins.hasAttr "cuda" driverConfig) [
    (nixpkgs.runCommand "cuda-drivers" {} (''
        mkdir -p $out
        cp -r ${cudaPath} $out/lib
    ''))
];
glDrivers = builtins.map (
    driver:  let 
        driverPath = ./.drivers/${driver.path};
    in
    nixpkgs.runCommand "${driver.name}-driver" {} (''
        mkdir -p $out $out/egl
        cp -r ${driverPath} $out/lib
        echo "{
            \"file_format_version\" : \"1.0.0\",
            \"ICD\" : {
                \"library_path\" : \"$out/lib/${driver.egl}\"
            }
        }" > $out/egl/vendor.json
    '')
) driverConfig.drivers;

drivers = cudaDrivers ++ glDrivers;
driverLibPaths = lib.makeLibraryPath drivers;

vendorPaths = lib.strings.concatStringsSep ":" (
    map (driver: "${driver}/egl/vendor.json") glDrivers
);
in
''
    export LD_LIBRARY_PATH=${driverLibPaths}
    export __EGL_VENDOR_LIBRARY_FILENAMES=${vendorPaths}
''