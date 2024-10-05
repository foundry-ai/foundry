{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    cudaPackages = nixpkgs.cudaPackages;
    bazel_6 = nixpkgs.bazel_6; addDriverRunPath = nixpkgs.addDriverRunPath;
    autoAddDriverRunpath = nixpkgs.autoAddDriverRunpath;
    curl = nixpkgs.curl;

    coreutils = nixpkgs.coreutils;
    double-conversion = nixpkgs.double-conversion;
    giflib = nixpkgs.giflib;
    jsoncpp = nixpkgs.jsoncpp;
    libjpeg_turbo = nixpkgs.libjpeg_turbo;
    openssl = nixpkgs.openssl;
    flatbuffers = nixpkgs.flatbuffers;
    protobuf = nixpkgs.protobuf;
    binutils = nixpkgs.binutils;
    zlib = nixpkgs.zlib;
    git = nixpkgs.git;
    which = nixpkgs.which;
    symlinkJoin = nixpkgs.symlinkJoin;
    snappy = nixpkgs.snappy;
    nsync = nixpkgs.nsync;

    IOKit = nixpkgs.darwin.apple_sdk.frameworks.IOKit;
    cctools = nixpkgs.cctools;

    numpy = dependencies.numpy;
    scipy = dependencies.scipy;
    ml-dtypes = dependencies.ml-dtypes;

    pybind11 = build-system.pybind11;
    cython = build-system.cython;
    setuptools = build-system.setuptools;
    build = build-system.build;
    wheel = build-system.wheel;

    fetchFromGitHub = nixpkgs.fetchFromGitHub;
    fetchurl = nixpkgs.fetchurl;

    cudaSupport = stdenv.isLinux;
    mklSupport = false;
in
let
  inherit (cudaPackages)
    cudaFlags
    cudaVersion
    nccl
    ;

  pname = "jaxlib";
  version = "0.4.33";

  # It's necessary to consistently use backendStdenv when building with CUDA
  # support, otherwise we get libstdc++ errors downstream
  stdenv = throw "Use effectiveStdenv instead";

  #baseStdenv = if cudaSupport then cudaPackages.backendStdenv
  #             else nixpkgs.gcc12Stdenv;
  baseStdenv = nixpkgs.gcc12Stdenv;

  # Use the raw, unwrapped clang.
  # We will handle all flags ourselves
  effectiveStdenv = nixpkgs.overrideCC 
    baseStdenv nixpkgs.llvmPackages_18.clang;

  buildBazelPackage = nixpkgs.buildBazelPackage.override { 
    stdenv = effectiveStdenv; 
  };

  meta = with lib; {
    description = "Source-built JAX backend. JAX is Autograd and XLA, brought together for high-performance machine learning research";
    homepage = "https://github.com/google/jax";
    license = licenses.asl20;
    maintainers = with maintainers; [ ndl ];

    # Make this platforms.unix once Darwin is supported.
    # The top-level jaxlib now falls back to jaxlib-bin on unsupported platforms.
    # aarch64-darwin is broken because of https://github.com/bazelbuild/rules_cc/pull/136
    # however even with that fix applied, it doesn't work for everyone:
    # https://github.com/NixOS/nixpkgs/pull/184395#issuecomment-1207287129
    platforms = platforms.linux;
  };

  # Bazel wants a merged cudnn at configuration time
  cudnnMerged = symlinkJoin {
    name = "cudnn-merged";
    paths = with cudaPackages; [
      (lib.getDev cudnn)
      (lib.getLib cudnn)
    ];
  };

  # These are necessary at build time and run time.
  cuda_libs_joined = symlinkJoin {
    name = "cuda-joined";
    paths = with cudaPackages; [
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
    paths = with cudaPackages; [
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


  # Copy-paste from TF derivation.
  # Most of these are not really used in jaxlib compilation but it's simpler to keep it
  # 'as is' so that it's more compatible with TF derivation.
  tf_system_libs = [
    "absl_py"
    "astor_archive"
    "astunparse_archive"
    # Not packaged in nixpkgs
    # "com_github_googleapis_googleapis"
    # "com_github_googlecloudplatform_google_cloud_cpp"
    # Issue with transitive dependencies after https://github.com/grpc/grpc/commit/f1d14f7f0b661bd200b7f269ef55dec870e7c108
    # "com_github_grpc_grpc"
    # ERROR: /build/output/external/bazel_tools/tools/proto/BUILD:25:6: no such target '@com_google_protobuf//:cc_toolchain':
    # target 'cc_toolchain' not declared in package '' defined by /build/output/external/com_google_protobuf/BUILD.bazel
    # "com_google_protobuf"
    # Fails with the error: external/org_tensorflow/tensorflow/core/profiler/utils/tf_op_utils.cc:46:49: error: no matching function for call to 're2::RE2::FullMatch(absl::lts_2020_02_25::string_view&, re2::RE2&)'
    # "com_googlesource_code_re2"
    "curl"
    "cython"
    "dill_archive"
    "double_conversion"
    "flatbuffers"
    "functools32_archive"
    "gast_archive"
    "gif"
    "hwloc"
    "icu"
    "jsoncpp_git"
    "libjpeg_turbo"
    "lmdb"
    "nasm"
    "opt_einsum_archive"
    "org_sqlite"
    "pasta"
    "png"
    # ERROR: /build/output/external/pybind11/BUILD.bazel: no such target '@pybind11//:osx':
    # target 'osx' not declared in package '' defined by /build/output/external/pybind11/BUILD.bazel
    # "pybind11"
    "six_archive"
    "snappy"
    "tblib_archive"
    "termcolor_archive"
    "typing_extensions_archive"
    "wrapt"
    "zlib"
  ];

  arch =
    # KeyError: ('Linux', 'arm64')
    if effectiveStdenv.hostPlatform.isLinux && effectiveStdenv.hostPlatform.linuxArch == "arm64" then
      "aarch64"
    else if effectiveStdenv.hostPlatform.isLinux && effectiveStdenv.hostPlatform.linuxArch == "powerpc" then
      "ppc64le"
    else
      effectiveStdenv.hostPlatform.linuxArch;

  boringssl = effectiveStdenv.mkDerivation {
    pname = "boringssl-src";
    version = "unstable";
    src = fetchurl {
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
  python-wrapper = ./python-wrapper;
  python-hermetic = effectiveStdenv.mkDerivation {
    pname = "bazel-python-hermetic";
    version = "unstable";
    nativeBuildInputs = [ python ];
    src = python-wrapper;
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
    src = fetchurl { 
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.34.0/rules_python-0.34.0.tar.gz";
        hash = "sha256-d4quqz5s/VbWgcifXBDXrWv40vGnLeneVbIwgbLTFhg=";
    };
    dontBukld = true;
    nativeBuildInputs = [ coreutils python ];
    postPatch = ''
        patchShebangs .
        substituteInPlace python/private/common/providers.bzl \
            --replace-fail "#!/usr/bin/env" "#!${coreutils}/bin/env"
    '';
    installPhase = ''
      cp -r . $out
    '';
  };

  xla = effectiveStdenv.mkDerivation {
    pname = "xla-src";
    version = "unstable";

    src = fetchFromGitHub {
      owner = "openxla";
      repo = "xla";
      # Update this according to https://github.com/google/jax/blob/jaxlib-v${version}/third_party/xla/workspace.bzl.
      rev = "720b2c53346660e95abbed7cf3309a8b85e979f9";
      hash = "sha256-9+YmFAYbOLDw5K3J14CqnwNxMoiU1/iIKC2gs2PlBBA=";
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
  cc-toolchain = ./toolchain;

  backend_cc_joined = symlinkJoin {
    name = "cuda-cc-joined";
    paths = [
      nixpkgs.llvmPackages_18.clang
      # for ar, dwp, nm, objcopy, objdump, strip
      binutils.bintools 
    ];
  };
  jaxlib-sources = effectiveStdenv.mkDerivation {
    name = "jaxlib-sources";
    version = version;
    src = fetchFromGitHub {
        owner = "google";
        repo = "jax";
        # use the jax instead of jaxlib tag because it is more reliable
        rev = "refs/tags/jax-v${version}";
        hash = "sha256-A1EWILuR5/dZdGoUe5lQBb96lkgtmquOxmkde46WS60=";
    };
    postPatch = ''
      rm -f .bazelversion
      substituteInPlace .bazelrc \
        --replace-fail 'build:cuda_clang --action_env=CLANG_CUDA_COMPILER_PATH="/usr/lib/llvm-18/bin/clang"' \
                       ' '
                      #  'build:cuda_clang --action_env=CLANG_CUDA_COMPILER_PATH="${effectiveStdenv.cc}/bin/cc"' 
      cp -r ${cc-toolchain} toolchain
      substituteInPlace toolchain/nix_toolchain_config.bzl \
        --subst-var-by CC_PATH ${backend_cc_joined}/bin/cc \
        --subst-var-by AR_PATH ${backend_cc_joined}/bin/ar \
        --subst-var-by LD_PATH ${backend_cc_joined}/bin/ld \
        --subst-var-by ARCH ${arch}
    '';
    preConfigure =
      # Dummy ldconfig to work around "Can't open cache file /nix/store/<hash>-glibc-2.38-44/etc/ld.so.cache" error
      ''
        mkdir dummy-ldconfig
        echo "#!${effectiveStdenv.shell}" > dummy-ldconfig/ldconfig
        chmod +x dummy-ldconfig/ldconfig
        export PATH="$PWD/dummy-ldconfig:$PATH"

      '' + ''
          cat <<CFG > ./.jax_configure.bazelrc
          # use our custom-toolchain
          build --extra_toolchains=//toolchain:cc_nix_toolchain

          build --strategy=Genrule=standalone
          build --override_repository=rules_python=${rules_python}
          build --override_repository=xla=${xla}
          # build --override_repository=boringssl=${boringssl}
          build -c opt

          build --repo_env PYTHON_BIN_PATH="${python}/bin/python"
          build --python_path="${python}/bin/python"

          build --action_env=PYENV_ROOT
          build --distinct_host_configuration=false
          build --define PROTOBUF_INCLUDE_PATH="${nixpkgs.protobuf}/include"
        '' + lib.optionalString effectiveStdenv.cc.isClang ''
          # bazel depends on the compiler frontend automatically selecting these flags based on file
          # extension but our clang doesn't.
          # https://github.com/NixOS/nixpkgs/issues/150655
          build --cxxopt=-x 
          build --cxxopt=c++
          build --host_cxxopt=-x 
          build --host_cxxopt=c++
        '' + lib.optionalString (!effectiveStdenv.cc.isClang) ''

        '' + lib.optionalString effectiveStdenv.hostPlatform.isPower64 ''
         build --cxxopt="-U__LONG_DOUBLE_IEEE128__"
         build --host_cxxopt="-U__LONG_DOUBLE_IEEE128__"
        '' + lib.optionalString cudaSupport ''
          build --config=cuda
          build --config=cuda_clang
          build --repo_env LOCAL_CUDA_PATH="${cuda_build_deps_joined}"
          build --repo_env LOCAL_CUDNN_PATH="${cudnnMerged}"
          build --repo_env LOCAL_NCCL_PATH="${lib.getDev nccl}"
          build --repo_env LOCAL_CUDA_COMPUTE_CAPABILITIES="${builtins.concatStringsSep "," cudaFlags.realArches}"
        '' +
        # Note that upstream conditions this on `wheel_cpu == "x86_64"`. We just
        # rely on `effectiveStdenv.hostPlatform.avxSupport` instead. So far so
        # good. See https://github.com/google/jax/blob/b9824d7de3cb30f1df738cc42e486db3e9d915ff/build/build.py#L322
        # for upstream's version.
        lib.optionalString (effectiveStdenv.hostPlatform.avxSupport && effectiveStdenv.hostPlatform.isUnix)
          ''
            build --config=avx_posix
          '' + lib.optionalString mklSupport ''
        build --config=mkl_open_source_only
      ''
      + ''
        CFG
      '';
    installPhase = ''
      cp -r . $out
    '';

    dontBuild = true;
  };

  test-build = (import ./bazel-test.nix { pkgs = nixpkgs; });
  bazel-build = (buildBazelPackage rec {
    name = "bazel-build-${pname}-${version}";

    # See https://github.com/google/jax/blob/main/.bazelversion for the latest.
    bazel = bazel_6;

    src = jaxlib-sources;
    nativeBuildInputs = [
      python
      cython
      nixpkgs.flatbuffers
      git
      setuptools
      wheel
      build
      which
      backend_cc_joined
    ];

    buildInputs =
      [
        curl
        double-conversion
        giflib
        jsoncpp
        libjpeg_turbo
        numpy
        openssl
        nixpkgs.flatbuffers
        nixpkgs.protobuf
        pybind11
        scipy
        snappy
        zlib
      ]
      ++ lib.optionals effectiveStdenv.isDarwin [ IOKit ]
      ++ lib.optionals (!effectiveStdenv.isDarwin) [ nsync ];

    bazelRunTarget = "//jaxlib/tools:build_wheel";
    runTargetFlags = [
      "--output_path=$out"
      "--cpu=${arch}"
      # This has no impact whatsoever...
      "--jaxlib_git_hash='12345678'"
    ];
    bazelRunFlags = [ "--verbose_failures" "--toolchain_resolution_debug=\"@com_google_absl//absl/time/internal/cctz:civil_time\""];

    removeRulesCC = false;
    hardeningDisable = ["all"];

    # The version is automatically set to ".dev" if this variable is not set.
    # https://github.com/google/jax/commit/e01f2617b85c5bdffc5ffb60b3d8d8ca9519a1f3
    env = {
      JAXLIB_RELEASE = "1";
      HERMETIC_PYTHON_VERSION = python-version;
      # GCC_HOST_COMPILER_PREFIX = lib.optionalString cudaSupport "${backend_cc_joined}/bin";
      # GCC_HOST_COMPILER_PATH = lib.optionalString cudaSupport "${backend_cc_joined}/bin/cc";
    };

    # We intentionally overfetch so we can share the fetch derivation across all the different configurations
    fetchAttrs = {
      TF_SYSTEM_LIBS = lib.concatStringsSep "," tf_system_libs;
      # we have to force @mkl_dnn_v1 since it's not needed on darwin
      bazelTargets = [
        bazelRunTarget
        "@mkl_dnn_v1//:mkl_dnn"
      ];
      bazelFlags = [
        "--config=avx_posix"
        "--config=mkl_open_source_only"
      ];
      sha256 = (if cudaSupport then { 
            x86_64-linux = "sha256-BcyGPJJRjr5rsdxKypmknmphaduQt8tiqg1QsbKHjlo=";
            powerpc64le-linux = "";
	    } else {
              x86_64-linux = "";
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

      TF_SYSTEM_LIBS = lib.concatStringsSep "," (
        tf_system_libs
        ++ lib.optionals (!effectiveStdenv.isDarwin) [
          "nsync" # fails to build on darwin
        ]
      );

      preBuild = ''
        export NIX_CFLAGS_COMPILE=
        export NIX_CFLAGS_LINK=
      '';
    };

    inherit meta;
  });

  platformTag =
    if effectiveStdenv.hostPlatform.isLinux then
      "manylinux2014_${arch}"
    else if effectiveStdenv.system == "x86_64-darwin" then
      "macosx_10_9_${arch}"
    else if effectiveStdenv.system == "aarch64-darwin" then
      "macosx_11_0_${arch}"
    else
      throw "Unsupported target platform: ${effectiveStdenv.hostPlatform}";
in
buildPythonPackage {
  inherit pname version;
  format = "wheel";

  src =
    let
      cp = "cp${builtins.replaceStrings [ "." ] [ "" ] python.pythonVersion}";
    in
    "${bazel-build}/jaxlib-${version}-${cp}-${cp}-${platformTag}.whl";

  # Note that jaxlib looks for "ptxas" in $PATH. See https://github.com/NixOS/nixpkgs/pull/164176#discussion_r828801621
  # for more info.
  postInstall = lib.optionalString cudaSupport ''
    mkdir -p $out/bin
    ln -s ${lib.getExe' cudaPackages.cuda_nvcc "ptxas"} $out/bin/ptxas

    find $out -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
      patchelf --add-rpath "${
        lib.makeLibraryPath [
          cuda_libs_joined
          (lib.getLib cudaPackages.cudnn)
          nccl
        ]
      }" "$lib"
    done
  '';

  nativeBuildInputs = lib.optionals cudaSupport [ autoAddDriverRunpath ];

  dependencies = [
    # absl-py
    curl
    double-conversion
    flatbuffers
    giflib
    jsoncpp
    libjpeg_turbo
    ml-dtypes
    numpy
    scipy
    # six
    snappy
  ];

  pythonImportsCheck = [
    "jaxlib"
    # `import jaxlib` loads surprisingly little. These imports are actually bugs that appeared in the 0.4.11 upgrade.
    "jaxlib.cpu_feature_guard"
    "jaxlib.xla_client"
  ];

  # Without it there are complaints about libcudart.so.11.0 not being found
  # because RPATH path entries added above are stripped.
  dontPatchELF = cudaSupport;

  passthru = {
    # Note "bazel.*.tar.gz" can be accessed as `jaxlib.bazel-build.deps`
   inherit bazel-build;
   inherit test-build;
  };

  inherit meta;
}