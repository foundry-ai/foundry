{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
        let lib = nixpkgs.lib;
            stdenv = nixpkgs.stdenv;
            blas = nixpkgs.blas;
            sage = nixpkgs.sage;
            lapack = nixpkgs.lapack;
            cfg = nixpkgs.writeTextFile {
                name = "site.cfg";
                text = lib.generators.toINI { } {
                    ${blas.implementation} = {
                        include_dirs = "${lib.getDev blas}/include:${lib.getDev lapack}/include";
                        library_dirs = "${blas}/lib:${lapack}/lib";
                        runtime_library_dirs = "${blas}/lib:${lapack}/lib";
                        libraries = "lapack,lapacke,blas,cblas";
                    };
                    lapack = {
                        include_dirs = "${lib.getDev lapack}/include";
                        library_dirs = "${lapack}/lib";
                        runtime_library_dirs = "${lapack}/lib";
                    };
                    blas = {
                        include_dirs = "${lib.getDev blas}/include";
                        library_dirs = "${blas}/lib";
                        runtime_library_dirs = "${blas}/lib";
                    };
                };
            }; in
    buildPythonPackage rec {
        pname = "numpy";
        version = "2.0.1";
        pyproject = true;
        src = fetchurl {
            url="https://files.pythonhosted.org/packages/1c/8a/0db635b225d2aa2984e405dc14bd2b0c324a0c312ea1bc9d283f2b83b038/numpy-2.0.1.tar.gz";
            hash="sha256-SFuHI1eWQQw1GaaZz+H6qwl+UJ6Q67BdzQmNsq6H57M=";
        };
        patches = lib.optionals python.hasDistutilsCxxPatch [ ./numpy-distutils-C++.patch ];
        nativeBuildInputs = [
            build-system.meson-python
            build-system.cython
            nixpkgs.gfortran
            nixpkgs.pkg-config
        ] ++ lib.optionals (stdenv.isDarwin) [ nixpkgs.xcbuild.xcrun ]
          ++ lib.optionals (!stdenv.buildPlatform.canExecute stdenv.hostPlatform) [ nixpkgs.mesonEmulatorHook ];

        buildInputs = [
            blas
            lapack
        ];
        # Causes `error: argument unused during compilation: '-fno-strict-overflow'` due to `-Werror`.
        hardeningDisable = lib.optionals stdenv.cc.isClang [ "strictoverflow" ];

        # we default openblas to build with 64 threads
        # if a machine has more than 64 threads, it will segfault
        # see https://github.com/OpenMathLib/OpenBLAS/issues/2993
        preConfigure = ''
            sed -i 's/-faltivec//' numpy/distutils/system_info.py
            export OMP_NUM_THREADS=$((NIX_BUILD_CORES > 64 ? 64 : NIX_BUILD_CORES))
        '';

        # HACK: copy mesonEmulatorHook's flags to the variable used by meson-python
        postConfigure = ''
            mesonFlags="$mesonFlags ''${mesonFlagsArray[@]}"
        '';

        preBuild = ''
            ln -s ${cfg} site.cfg
        '';
        enableParallelBuilding = true;
        doCheck = false;
    }