{buildPythonPackage, env, nixpkgs, python, fetchurl} : 
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
        version = "1.26.4";
        src = fetchurl {
            url = "https://files.pythonhosted.org/packages/65/6e/09db70a523a96d25e115e71cc56a6f9031e7b8cd166c1ac8438307c14058/numpy-1.26.4.tar.gz";
            hash = "sha256-KgKrqe0S5KxOs+qUIcQgMBoMZGDZgw10qd+H76SRIBA=";
        };
        patches = lib.optionals python.hasDistutilsCxxPatch [ ./numpy-distutils-C++.patch ];
        nativeBuildInputs = [
            env.meson-python
            env.cython
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