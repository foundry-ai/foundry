{ pkgs, python, dependencies, build-system }:
let lib = pkgs.lib;
    numpy = dependencies.numpy;
    scipy = dependencies.scipy;
    ml-dtypes = dependencies.ml-dtypes;

    pybind11 = build-system.pybind11;
    cython = build-system.cython;
    setuptools = build-system.setuptools;
    build = build-system.build;
    wheel = build-system.wheel;

    effectiveStdenv = pkgs.overrideCC 
      pkgs.stdenv pkgs.llvmPackages_18.clang;
      stdenv = throw "Use effectiveStdenv instead";

    buildBazelPackage = pkgs.buildBazelPackage.override { 
      stdenv = effectiveStdenv; 
    };
    symlinkJoin = pkgs.symlinkJoin;

    pname = "jaxlib";
    version = "0.4.34";
    arch =
      if effectiveStdenv.hostPlatform.isLinux && effectiveStdenv.hostPlatform.linuxArch == "arm64" then
        "aarch64"
      else if effectiveStdenv.hostPlatform.isLinux && effectiveStdenv.hostPlatform.linuxArch == "powerpc" then
        "ppc64le"
      else effectiveStdenv.hostPlatform.linuxArch;
    platformTag =
      if effectiveStdenv.hostPlatform.isLinux then
        "manylinux2014_${arch}"
      else if effectiveStdenv.system == "x86_64-darwin" then
        "macosx_10_9_${arch}"
      else if effectiveStdenv.system == "aarch64-darwin" then
        "macosx_11_0_${arch}"
      else throw "Unsupported target platform: ${effectiveStdenv.hostPlatform}";
in rec {
  inherit pname version effectiveStdenv buildBazelPackage;
  cudnnMerged = symlinkJoin {
    name = "cudnn-merged";
    paths = with pkgs.cudaPackages; [
      (lib.getDev cudnn)
      (lib.getLib cudnn)
    ];
  };
  # These are necessary at build time and run time.
  cuda_libs_joined = symlinkJoin {
    name = "cuda-joined";
    paths = with pkgs.cudaPackages; [
      (lib.getLib cuda_cudart) # libcudart.so
      (lib.getLib cuda_cupti) # libcupti.so
      (lib.getLib libcublas) # libcublas.so
      (lib.getLib libcufft) # libcufft.so
      (lib.getLib libcurand) # libcurand.so
      (lib.getLib libcusolver) # libcusolver.so
      (lib.getLib libcusparse) # libcusparse.so
    ];
  };
  # These are only necessary at build time.
  cuda_build_deps_joined = symlinkJoin {
    name = "cuda-build-deps-joined";
    paths = with pkgs.cudaPackages; [
      cuda_libs_joined

      # Binaries
      (lib.getBin cuda_nvcc) # nvcc

      # Archives
      (lib.getOutput "static" cuda_cudart) # libcudart_static.a

      # Headers
      (lib.getDev cuda_cccl) # block_load.cuh
      (lib.getDev cuda_cudart) # cuda.h
      (lib.getDev cuda_cupti) # cupti.h
      (lib.getDev cuda_nvcc) # See https://github.com/google/jax/issues/19811
      (lib.getDev cuda_nvml_dev) # nvml.h
      (lib.getDev cuda_nvtx) # nvToolsExt.h
      (lib.getDev libcublas) # cublas_api.h
      (lib.getDev libcufft) # cufft.h
      (lib.getDev libcurand) # curand.h
      (lib.getDev libcusolver) # cusolver_common.h
      (lib.getDev libcusparse) # cusparse.h
    ];
  };
  tf_system_libs = [];

  boringssl = effectiveStdenv.mkDerivation {
    pname = "boringssl-src";
    version = "unstable";
    src = pkgs.fetchurl {
      url = "https://github.com/google/boringssl/archive/b9232f9e27e5668bc0414879dcdedb2a59ea75f2.tar.gz";
      hash = "sha256-U0+mWL2EX9l0tQsQ9ETTkt/Q2TdoxKUbYSY/032FHEA=";
    };
    dontBuild = true;
    installPhase = ''
      cp -r . $out
    '';
  };

  python-version = with lib.versions; "${major python.pythonVersion}.${minor python.pythonVersion}";
  python-toolchain = ./python_init_toolchains.bzl;
  python-hermetic = effectiveStdenv.mkDerivation {
    pname = "bazel-python-hermetic";
    version = "unstable";
    nativeBuildInputs = [ python ];
    src = ./python-wrapper;
    dontBuild = true;
    postPatch = ''
        substituteInPlace bin/python \
            --subst-var-by PYTHON_ROOT "${python}"
        substituteInPlace bin/python3 \
            --subst-var-by PYTHON_ROOT "${python}"
        patchShebangs .
        cp -r ${python}/lib lib
        cp -r ${python}/include include
    '';
    installPhase = ''
        mkdir $out
        tar -czf $out/distribution.tar.gz .
        SHA256=($(sha256sum $out/distribution.tar.gz))
        echo $SHA256 > $out/sha256.txt
    '';
  };
  rules_python = effectiveStdenv.mkDerivation {
    pname = "python_urles";
    version = "unstable";
    src = pkgs.fetchurl { 
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.34.0/rules_python-0.34.0.tar.gz";
        hash = "sha256-d4quqz5s/VbWgcifXBDXrWv40vGnLeneVbIwgbLTFhg=";
    };
    dontBuild = true;
    nativeBuildInputs = [ pkgs.coreutils python ];
    postPatch = ''
        patchShebangs .
        substituteInPlace python/private/common/providers.bzl \
            --replace-fail "#!/usr/bin/env" "#!${pkgs.coreutils}/bin/env"
    '';
    installPhase = ''
      cp -r . $out
    '';
  };
  xla = effectiveStdenv.mkDerivation {
    pname = "xla-src";
    version = "unstable";

    src = pkgs.fetchFromGitHub {
      owner = "openxla";
      repo = "xla";
      # Update this according to https://github.com/google/jax/blob/jaxlib-v${version}/third_party/xla/workspace.bzl.
      rev = "cd6e808c59f53b40a99df1f1b860db9a3e598bff";
      hash = "sha256-QiQgoegUQxmss/ynohF/N2hO6ua2TF3nwBMPi9KvIiU=";
    };

    dontBuild = true;

    # This is necessary for patchShebangs to know the right path to use.
    nativeBuildInputs = [ python ];

    # Main culprits we're targeting are third_party/tsl/third_party/gpus/crosstool/clang/bin/*.tpl
    postPatch = ''
      patchShebangs .
      cp ${python-toolchain} third_party/py/python_init_toolchains.bzl
      PYTHON_SHA256=$(cat ${python-hermetic}/sha256.txt)
      substituteInPlace third_party/py/python_init_toolchains.bzl \
        --subst-var-by PYTHON_VERSION ${python.version} \
        --subst-var-by PYTHON_TAR_PATH ${python-hermetic}/distribution.tar.gz \
        --subst-var-by PYTHON_SHA256 $PYTHON_SHA256
    '';
    installPhase = ''
      cp -r . $out
    '';
  };
  backend_cc_joined = symlinkJoin {
    name = "jaxlib-cc-joined";
    paths = [
      effectiveStdenv.cc
      # for ar, dwp, nm, objcopy, objdump, strip
      pkgs.binutils.bintools 
    ];
  };
  jaxlib-sources = effectiveStdenv.mkDerivation {
    name = "jaxlib-sources";
    version = version;
    src = pkgs.fetchFromGitHub {
        owner = "google";
        repo = "jax";
        # use the jax instead of jaxlib tag because it is more reliable
        rev = "refs/tags/jax-v${version}";
        hash = "sha256-f49YECYVkb5NpG/5GSSVW3D3J0Lruq2gI62iiXSOHkw=";
    };
    postPatch = ''
      rm -f .bazelversion
      substituteInPlace .bazelrc \
        --replace-fail 'build:cuda_clang --action_env=CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-18/bin/clang"' \
                       ' '
      mkdir -p toolchain
      cp -r ${./toolchain_BUILD} toolchain/BUILD

      # Get the include paths from the c++ compiler
      echo "Detecting options:"
      clang_output=$(${effectiveStdenv.cc}/bin/c++ -v -E - < /dev/null 2>&1)
      clang_search=''${clang_output#*"<...> search starts here:"}
      clang_search=''${clang_search%"End of search list."*}
      set -f
      printf -v default_includes '"%s", ' $clang_search
      set +f

      substituteInPlace toolchain/BUILD \
        --subst-var-by CC_PATH ${backend_cc_joined}/bin/cc \
        --subst-var-by AR_PATH ${backend_cc_joined}/bin/ar \
        --subst-var-by LD_PATH ${backend_cc_joined}/bin/ld \
        --subst-var-by OBJCOPY_PATH ${backend_cc_joined}/bin/objcopy \
        --subst-var-by OBJDUMP_PATH ${backend_cc_joined}/bin/objdump \
        --subst-var-by DEFAULT_INCLUDE_PATHS "$default_includes" \
        --subst-var-by ARCH ${arch}
    '';
    preConfigure = ''
      cat <<CFG > ./.jax_configure.bazelrc
      # use our custom-toolchain
      build --strategy=Genrule=standalone
      build --override_repository=rules_python=${rules_python}
      build --override_repository=xla=${xla}
      # build --override_repository=boringssl=${boringssl}
      build -c opt
      build --distinct_host_configuration=false
    '' +
      lib.optionalString (effectiveStdenv.hostPlatform.avxSupport && effectiveStdenv.hostPlatform.isUnix)
    ''
      build --config=avx_posix
    '' + ''
      CFG
    '';
        # ''
        #   build --config=cuda
        #   build --config=cuda_clang
        #   build --repo_env LOCAL_CUDA_PATH="${cuda_build_deps_joined}"
        #   build --repo_env LOCAL_CUDNN_PATH="${cudnnMerged}"
        #   build --repo_env LOCAL_NCCL_PATH="${lib.getDev nccl}"
        #   build --repo_env LOCAL_CUDA_COMPUTE_CAPABILITIES="${builtins.concatStringsSep "," cudaFlags.realArches}"
        # '' +
    installPhase = ''
      cp -r . $out
    '';

    dontBuild = true;
  };
  jaxlib-wheel-build = buildBazelPackage rec {
    name = "jaxlib-wheel-${pname}-${version}";
    bazel = pkgs.bazel_6;
    src = jaxlib-sources;
    nativeBuildInputs = [
      pkgs.git
      pkgs.which
      python
      cython
      setuptools
      wheel
      build
      backend_cc_joined
    ];

    buildInputs = [
        pkgs.curl
        pkgs.double-conversion
        pkgs.openssl
        numpy
        pybind11
        scipy
    ];

    preConfigure = ''
      mkdir dummy-ldconfig
      echo "#!${effectiveStdenv.shell}" > dummy-ldconfig/ldconfig
      chmod +x dummy-ldconfig/ldconfig
      export PATH="$PWD/dummy-ldconfig:$PATH"
    '';

    bazelRunTarget = "//jaxlib/tools:build_wheel";
    runTargetFlags = [
      "--output_path=$out"
      "--cpu=${arch}"
      # This has no impact whatsoever...
      "--jaxlib_git_hash='12345678'"
    ];
    bazelRunFlags = [ 
      "--verbose_failures" 
      "--crosstool_top=//toolchain:cc_nix_toolchains"
    ];
    removeRulesCC = false;
    dontAddBazelOpts = true;
    hardeningDisable = ["all"];

    # The version is automatically set to ".dev" if this variable is not set.
    # https://github.com/google/jax/commit/e01f2617b85c5bdffc5ffb60b3d8d8ca9519a1f3
    env = {
      JAXLIB_RELEASE = "1";
      TF_SYSTEM_LIBS = lib.concatStringsSep "," tf_system_libs;
      HERMETIC_PYTHON_VERSION = python-version;
      GCC_HOST_COMPILER_PREFIX = "${backend_cc_joined}/bin";
      GCC_HOST_COMPILER_PATH = "${backend_cc_joined}/bin/cc";
    };

    # We intentionally overfetch so we can share the fetch derivation across all the different configurations
    fetchAttrs = {
      # we have to force @mkl_dnn_v1 since it's not needed on darwin
      bazelTargets = [
        bazelRunTarget
        "@mkl_dnn_v1//:mkl_dnn"
      ];
      bazelFlags = [
        "--config=avx_posix"
        "--config=mkl_open_source_only"
      ];
      sha256 = ({
        x86_64-linux = "sha256-bkYbcpOknA5Ar7knUp0pWUeofv/wcjjwO5Z5gsuK+8E=";
        aarch64-linux = "";
      }).${effectiveStdenv.system} or (throw "jaxlib: unsupported system: ${effectiveStdenv.system}");

      # Non-reproducible fetch https://github.com/NixOS/nixpkgs/issues/321920#issuecomment-2184940546
      preInstall = ''
        cat << \EOF > "$bazelOut/external/go_sdk/versions.json"
        []
        EOF
      '';
    };

    buildAttrs = {
      outputs = [ "out" ];
    };
  };
  jaxlib-wheel = let
    cp = "cp${builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion}";
  in "${jaxlib-wheel-build}/jaxlib-${version}-${cp}-${cp}-${platformTag}.whl";
}