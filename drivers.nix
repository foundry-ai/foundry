{nixpkgs}:
let 

lib = nixpkgs.lib;
driverConfig = lib.importJSON ./.drivers/config.json;

    cudaLib = ./.drivers/${driverConfig.cuda.path};
    cudaDrivers = nixpkgs.runCommand "cuda-drivers" {} (''
        mkdir -p $out $out/lib
        cp ${cudaLib} $out/lib/libcuda.so
        ln -s $out/lib/libcuda.so $out/lib/libcuda.so.1
    '');

in
''
    export LD_LIBRARY_PATH=${cudaDrivers}/lib
''