{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    cc = stdenv.cc;

    setuptools = build-system.setuptools;
    wheel = build-system.wheel;
    setuptools-scm = build-system.setuptools-scm;
    numpy = build-system.numpy;

    bazel = nixpkgs.bazel;
    cctools = nixpkgs.darwin.cctools;

    numpy_dep = dependencies.numpy;
    ml-dtypes = dependencies.ml-dtypes;

    fetchPypi = python.pkgs.fetchPypi;
    pytestCheckHook = python.pkgs.pytestCheckHook;
    pythonOlder = python.pkgs.pythonOlder;

    # CommonCrypto = nixpkgs.darwin.CommonCrypto;
    CoreFoundation = nixpkgs.darwin.apple_sdk.frameworks.CoreFoundation;
    curl = nixpkgs.curl;
    CoreServices = nixpkgs.darwin.apple_sdk.frameworks.CoreServices;
    SystemConfiguration = nixpkgs.darwin.apple_sdk.frameworks.SystemConfiguration;
    Security = nixpkgs.darwin.apple_sdk.frameworks.Security;
    Foundation = nixpkgs.darwin.apple_sdk.frameworks.Foundation;


    bazelPath = "${bazel}/bin/bazel";
    grpcPatch = ./grpc.patch;
    bazelExtraFlags = ["-Wno-unused-command-line-argument"] ++
        (lib.optionals stdenv.isDarwin [
            # "-isystem ${libDER}/include"
            "-F${Security}/Library/Frameworks"
            "-F${CoreFoundation}/Library/Frameworks"
            "-F${CoreServices}/Library/Frameworks"
            "-F${Foundation}/Library/Frameworks"
            "-F${SystemConfiguration}/Library/Frameworks"
            "-L${nixpkgs.llvmPackages.libcxx}/lib"
            "-L${nixpkgs.libiconv}/lib"
            "-L${nixpkgs.darwin.libobjc}/lib"
            "-resource-dir=${nixpkgs.stdenv.cc}/resource-root"
            "-Wno-elaborated-enum-base"
    ]);
    bazelLinkerFlags = lib.optionals stdenv.isDarwin [
        "-F${Security}/Library/Frameworks"
        "-F${CoreFoundation}/Library/Frameworks"
        "-F${CoreServices}/Library/Frameworks"
        "-F${Foundation}/Library/Frameworks"
        "-F${SystemConfiguration}/Library/Frameworks"
        "-L${nixpkgs.llvmPackages.libcxx}/lib"
        "-L${nixpkgs.libiconv}/lib"
        "-L${nixpkgs.darwin.libobjc}/lib"
    ];
    rcFlags = (lib.strings.concatMapStrings 
        (x: ''build --cxxopt="${x}" --host_cxxopt="${x}"
        build --copt="${x}" --host_copt="${x}"
        '') bazelExtraFlags) + (lib.strings.concatMapStrings
        (x: ''build --linkopt="${x}" --host_linkopt="${x}"
        '') bazelLinkerFlags) + (lib.optionalString stdenv.isDarwin
            ''build --cxxopt="-xc++" --host_cxxopt="-xc++"
            build --sandbox_debug
        '');

    curlBuildExternal = nixpkgs.writeText "curl.BUILD" 
''cc_library(
    name = "curl",
    defines = ["TENSORSTORE_SYSTEM_CURL"],
    linkopts = ["-lcurl -L${curl.out}/lib"],
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)'';
    curlWorkspaceExternal = nixpkgs.writeText "workspace.bzl" 
''load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
def repo():
    maybe(
        native.new_local_repository,
        name = "se_curl",
        path = "${curl.dev}",
        build_file = "third_party/se_curl/system.BUILD.bazel"
    )'';

in
buildPythonPackage rec {
  pname = "tensorstore";
  version = "0.1.64";
  format = "setuptools";

  disabled = pythonOlder "3.7";

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-f6iekIdvtTd+/FTz83MmpvuD7J4TJlZYGadaToCUmIY=";
  };

  build-system = [setuptools wheel setuptools-scm numpy];
  dependencies = [numpy_dep ml-dtypes];

  buildInputs = [nixpkgs.libcxx];
  nativeBuildInputs = [bazel] ++ lib.optionals stdenv.isDarwin [
    cctools CoreFoundation CoreServices 
    Security Foundation SystemConfiguration
    curl.dev
  ];

  # use the darwin cctools libtool
  env.LIBTOOL = lib.optionalString stdenv.isDarwin "${cctools}/bin/libtool";
  # For darwin only:
  postPatch = ''
    substituteInPlace bazelisk.py \
      --replace-fail "def get_bazel_path():" \
       "def get_bazel_path():
    return '${bazelPath}'
"
    substituteInPlace .bazelversion \
      --replace-fail "6.4.0" "6.5.0"
    RCFLAGS='${rcFlags}'
    echo -e "$RCFLAGS\n$(cat .bazelrc)" > .bazelrc
    cp ${curlBuildExternal} third_party/se_curl/system.BUILD.bazel
    cp ${curlWorkspaceExternal} third_party/se_curl/workspace.bzl
    cp ${grpcPatch} third_party/com_github_grpc_grpc/patches/absl-fix.patch
    substituteInPlace third_party/com_github_grpc_grpc/workspace.bzl \
      --replace-fail 'patches = [' 'patches = [
        Label("//third_party:com_github_grpc_grpc/patches/absl-fix.patch"),'
  '';
  preBuild = ''
    # bazel clean --expunge
  '';
}