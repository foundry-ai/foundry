{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    cudaPackages = (import ./cuda.nix {nixpkgs=nixpkgs;});
    binutils = nixpkgs.binutils;

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

    cudaVersion = cudaPackages.cudaVersion;
    cudaFlags = cudaPackages.cudaFlags;
    cudatoolkit = cudaPackages.cudatoolkit;
    cudnn = cudaPackages.cudnn;
    nccl = cudaPackages.nccl;
    

    fetchFromGitHub = nixpkgs.fetchFromGitHub;
    buildBazelPackage = nixpkgs.buildBazelPackage;

    cudaSupport = stdenv.isLinux;
in
let
  pname = "jaxlib";
  version = "0.4.28";
  # It's necessary to consistently use backendStdenv when building with CUDA
  # support, otherwise we get libstdc++ errors downstream
  stdenv = throw "Use effectiveStdenv instead";
  # effectiveStdenv = if cudaSupport then cudaPackages.backendStdenv else nixpkgs.stdenv;
  effectiveStdenv = if cudaSupport 
  then cudaPackages.cudaStdenv 
  else nixpkgs.stdenv;

  buildBazelPackage = nixpkgs.buildBazelPackage.override {
    stdenv = effectiveStdenv;
  };

  meta = with lib; {
    description = "Source-built JAX backend. JAX is Autograd and XLA, brought together for high-performance machine learning research";
    homepage = "https://github.com/google/jax";
    license = licenses.asl20;
    maintainers = with maintainers; [ ndl ];
    platforms = platforms.linux;
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

  backend_cc_joined = nixpkgs.symlinkJoin {
    name = "jaxlib-cc-joined";
    paths = [
      effectiveStdenv.cc
      binutils.bintools # for ar, dwp, nm, objcopy, objdump, strip
    ];
  };

  xla = effectiveStdenv.mkDerivation {
    pname = "xla-src";
    version = "unstable";

    src = fetchFromGitHub {
      owner = "openxla";
      repo = "xla";
      # Update this according to https://github.com/google/jax/blob/jaxlib-v${version}/third_party/xla/workspace.bzl.
      rev = "e8247c3ea1d4d7f31cf27def4c7ac6f2ce64ecd4";
      hash = "sha256-ZhgMIVs3Z4dTrkRWDqaPC/i7yJz2dsYXrZbjzqvPX3E=";
    };

    dontBuild = true;

    # This is necessary for patchShebangs to know the right path to use.
    nativeBuildInputs = [ python ];

    # Main culprits we're targeting are third_party/tsl/third_party/gpus/crosstool/clang/bin/*.tpl
    postPatch = ''
      patchShebangs .
    '';

    installPhase = ''
      cp -r . $out
    '';
  };

  bazel-build = buildBazelPackage rec {
    name = "bazel-build-${pname}-${version}";

    # See https://github.com/google/jax/blob/main/.bazelversion for the latest.
    bazel = nixpkgs.bazel_6;

    src = fetchFromGitHub {
      owner = "google";
      repo = "jax";
      # google/jax contains tags for jax and jaxlib. Only use jaxlib tags!
      rev = "refs/tags/${pname}-v${version}";
      hash = "sha256-qSHPwi3is6Ts7pz5s4KzQHBMbcjGp+vAOsejW3o36Ek=";
    };
    patches = [
      ./patches/0002-bazelrc-edit.diff
      ./patches/0003-Patched-openxla-for-tf_runtime.patch
      ./patches/0005-Use-ppc64le-compatible-boringssl.patch
    ];
    nativeBuildInputs = [
      cython
      nixpkgs.flatbuffers
      nixpkgs.git
      setuptools
      wheel
      build
      nixpkgs.which
    ] ++ lib.optionals effectiveStdenv.isDarwin [ nixpkgs.cctools ]
      ++ lib.optionals cudaSupport [ cudatoolkit cudnn ];

    buildInputs =
      [
        backend_cc_joined
        nixpkgs.curl
        nixpkgs.double-conversion
        nixpkgs.giflib
        nixpkgs.jsoncpp
        nixpkgs.libjpeg_turbo
        numpy
        nixpkgs.openssl
        nixpkgs.flatbuffers
        nixpkgs.protobuf
        pybind11
        scipy
        python.pkgs.six
        nixpkgs.snappy
        nixpkgs.zlib
      ]
      ++ lib.optionals effectiveStdenv.isDarwin [ IOKit ]
      ++ lib.optionals (!effectiveStdenv.isDarwin) [ nixpkgs.nsync ]
      ++ lib.optionals cudaSupport [ cudatoolkit cudnn ];

    # We don't want to be quite so picky regarding bazel version
    postPatch = ''
      rm -f .bazelversion
    '';

    bazelRunTarget = "//jaxlib/tools:build_wheel";
    runTargetFlags = [
      "--output_path=$out"
      "--cpu=${arch}"
      # This has no impact whatsoever...
      "--jaxlib_git_hash='12345678'"
    ];


    GCC_HOST_COMPILER_PREFIX = lib.optionalString cudaSupport "${backend_cc_joined}/bin";
    GCC_HOST_COMPILER_PATH = lib.optionalString cudaSupport "${backend_cc_joined}/bin/gcc";

    removeRulesCC = false;

    # The version is automatically set to ".dev" if this variable is not set.
    # https://github.com/google/jax/commit/e01f2617b85c5bdffc5ffb60b3d8d8ca9519a1f3
    JAXLIB_RELEASE = "1";

    preConfigure =
      # Dummy ldconfig to work around "Can't open cache file /nix/store/<hash>-glibc-2.38-44/etc/ld.so.cache" error
      ''
        mkdir dummy-ldconfig
        echo "#!${effectiveStdenv.shell}" > dummy-ldconfig/ldconfig
        chmod +x dummy-ldconfig/ldconfig
        export PATH="$PWD/dummy-ldconfig:${python}/bin:$PATH"
        export PYTHON_BIN_PATH="${python}/bin/python"
      ''
      +

        # Construct .jax_configure.bazelrc. See https://github.com/google/jax/blob/b9824d7de3cb30f1df738cc42e486db3e9d915ff/build/build.py#L259-L345
        # for more info. We assume
        # * `cpu = None`
        # * `enable_nccl = True`
        # * `target_cpu_features = "release"`
        # * `rocm_amdgpu_targets = None`
        # * `enable_rocm = False`
        # * `build_gpu_plugin = False`
        # * `use_clang = False` (Should we use `effectiveStdenv.cc.isClang` instead?)
        #
        # Note: We should try just running https://github.com/google/jax/blob/ceb198582b62b9e6f6bdf20ab74839b0cf1db16e/build/build.py#L259-L266
        # instead of duplicating the logic here. Perhaps we can leverage the
        # `--configure_only` flag (https://github.com/google/jax/blob/ceb198582b62b9e6f6bdf20ab74839b0cf1db16e/build/build.py#L544-L548)?
        ''
          cat <<CFG > ./.jax_configure.bazelrc
          build --strategy=Genrule=standalone
          build --repo_env PYTHON_BIN_PATH="${python}/bin/python"
          build --action_env=PYENV_ROOT
          build --python_path="${python}/bin/python"
          build --distinct_host_configuration=false
          build --define PROTOBUF_INCLUDE_PATH="${nixpkgs.protobuf}/include"
        ''
      + lib.optionalString effectiveStdenv.hostPlatform.isPower64 ''
        build --cxxopt="-U__LONG_DOUBLE_IEEE128__"
        build --host_cxxopt="-U__LONG_DOUBLE_IEEE128__"
      ''
      + lib.optionalString cudaSupport ''
        build --config=cuda
        build --action_env CUDA_TOOLKIT_PATH="${cudatoolkit}"
        build --action_env CUDNN_INSTALL_PATH="${cudnn}"
        build --action_env TF_CUDA_PATHS="${cudatoolkit},${cudnn},${lib.getDev nccl}"
        build --action_env TF_CUDA_VERSION="${lib.versions.majorMinor cudaVersion}"
        build --action_env TF_CUDNN_VERSION="${lib.versions.major cudnn.version}"
        build:cuda --action_env TF_CUDA_COMPUTE_CAPABILITIES="${builtins.concatStringsSep "," cudaFlags.realArches}"
      ''
      # build --action_env CUDNN_INSTALL_PATH="${cudnnMerged}"
      # build --action_env TF_CUDNN_VERSION=""
      +
        # Note that upstream conditions this on `wheel_cpu == "x86_64"`. We just
        # rely on `effectiveStdenv.hostPlatform.avxSupport` instead. So far so
        # good. See https://github.com/google/jax/blob/b9824d7de3cb30f1df738cc42e486db3e9d915ff/build/build.py#L322
        # for upstream's version.
        lib.optionalString (effectiveStdenv.hostPlatform.avxSupport && effectiveStdenv.hostPlatform.isUnix)
          ''
            build --config=avx_posix
          ''
      + ''
        CFG
      '';

    # Make sure Bazel knows about our configuration flags during fetching so that the
    # relevant dependencies can be downloaded.
    bazelFlags =
      [
        "-c opt"
        # See https://bazel.build/external/advanced#overriding-repositories for
        # information on --override_repository flag.
        "--override_repository=xla=${xla}"
        "--copt=-Wno-dangling-pointer"
        "--host_copt=-Wno-dangling-pointer"
      ]
      ++ lib.optionals effectiveStdenv.cc.isClang [
        # bazel depends on the compiler frontend automatically selecting these flags based on file
        # extension but our clang doesn't.
        # https://github.com/NixOS/nixpkgs/issues/150655
        "--cxxopt=-x"
        "--cxxopt=c++"
        "--host_cxxopt=-x"
        "--host_cxxopt=c++"
      ];

    # We intentionally overfetch so we can share the fetch derivation across all the different configurations
    fetchAttrs = {
      TF_SYSTEM_LIBS = lib.concatStringsSep "," tf_system_libs;
      # we have to force @mkl_dnn_v1 since it's not needed on darwin
      bazelTargets = [
        bazelRunTarget
        "@mkl_dnn_v1//:mkl_dnn"
      ];

      bazelFlags =
        bazelFlags
        ++ [
          "--config=avx_posix"
          "--config=mkl_open_source_only"
        ]
        ++ lib.optionals cudaSupport [
          "--config=cuda"
        ];

      sha256 =
        (
          if cudaSupport then
            { 
              x86_64-linux = "sha256-rY0jDdP/uTM7vI7zxc9okk/Qy2/8LnZ24NrXtLld85A=";
              powerpc64le-linux = "sha256-bMoXnb3ZZYiXOdm7Q3aTAyF3jUyikY7ABKs+Ha/YZIk=";
            }
          else
            {
              x86_64-linux = lib.fakeSha256;
              powerpc64le-linux = "sha256-fD628yZ08zwFauMRyhGe+mf+vLzFjm/wpHuJizMZPD4="; # lib.fakeSha256; #"sha256-fD628yZ08zwFauMRyhGe+mf+vLzFjm/wpHuJizMZPD4=";
              aarch64-linux = lib.fakeSha256; 
            }
        ).${effectiveStdenv.system} or (throw "jaxlib: unsupported system: ${effectiveStdenv.system}");

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

      # Note: we cannot do most of this patching at `patch` phase as the deps
      # are not available yet. Framework search paths aren't added by bintools
      # hook. See https://github.com/NixOS/nixpkgs/pull/41914.
      preBuild = lib.optionalString effectiveStdenv.isDarwin ''
        export NIX_LDFLAGS+=" -F${IOKit}/Library/Frameworks"
        substituteInPlace ../output/external/rules_cc/cc/private/toolchain/osx_cc_wrapper.sh.tpl \
          --replace "/usr/bin/install_name_tool" "${cctools}/bin/install_name_tool"
        substituteInPlace ../output/external/rules_cc/cc/private/toolchain/unix_cc_configure.bzl \
          --replace "/usr/bin/libtool" "${cctools}/bin/libtool"
      '';
    };

    inherit meta;
  };
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
    ln -s ${cudatoolkit}/bin/ptxas $out/bin/ptxas

    find $out -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
      patchelf --add-rpath "${
        lib.makeLibraryPath [
          "${cudatoolkit}/lib64"
          "${lib.getLib effectiveStdenv.cc.cc}"
        ]
      }"  "$lib"
    done
  '';

  dependencies = [
    nixpkgs.curl
    nixpkgs.double-conversion
    nixpkgs.flatbuffers
    nixpkgs.giflib
    nixpkgs.jsoncpp
    nixpkgs.libjpeg_turbo
    nixpkgs.snappy
    ml-dtypes
    numpy
    scipy
  ] ++ lib.optionals cudaSupport [cudatoolkit];

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
    cudaPackages = cudaPackages;
  };

  inherit meta;
}
