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
    libDER = nixpkgs.darwin.apple_sdk.libs.libDER;
    CoreServices = nixpkgs.darwin.apple_sdk.frameworks.CoreServices;
    Security = nixpkgs.darwin.apple_sdk.frameworks.Security;
    Foundation = nixpkgs.darwin.apple_sdk.frameworks.Foundation;
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
  nativeBuildInputs = [bazel] ++ lib.optional stdenv.isDarwin [cctools libDER]; # ++ lib.optional stdenv.isDarwin [CoreFoundation CoreServices Security Foundation cctools];
  bazelPath = "${bazel}/bin/bazel";

  # use the darwin cctools libtool
  env.LIBTOOL = lib.optionalString stdenv.isDarwin "${cctools}/bin/libtool";
  bazelExtraFlags = ["-Wno-unused-command-line-argument"] ++
    (lib.optionals stdenv.isDarwin [
        # "-I${libDER}/include"
        "-F${Security}/Library/Frameworks"
        "-F${CoreFoundation}/Library/Frameworks"
        "-F${CoreServices}/Library/Frameworks"
        "-F${Foundation}/Library/Frameworks"
        "-L${nixpkgs.llvmPackages.libcxx}/lib"
        "-L${nixpkgs.libiconv}/lib"
        "-L${nixpkgs.darwin.libobjc}/lib"
        "-resource-dir=${nixpkgs.stdenv.cc}/resource-root"
        "-Wno-elaborated-enum-base"
  ]);
  rcFlags = lib.strings.concatMapStrings 
    (x: ''build --cxxopt="${x}" --host_cxxopt="${x}"
    build --copt="${x}" --host_copt="${x}"
    '') bazelExtraFlags;
  # For darwin only:
  postPatch = ''
    substituteInPlace bazelisk.py \
      --replace-fail "def get_bazel_path():" \
       "def get_bazel_path():
    return '${bazelPath}'
"
    substituteInPlace .bazelversion \
      --replace-fail "6.4.0" "6.5.0"
    substituteInPlace .bazelrc \
      --replace-fail "# Configure C++17 mode" '
      build --cxxopt="-xc++" --host_cxxopt="-xc++"
      ${rcFlags}
    '
    '';
  preBuild = ''
    BAZELRC=$(cat .bazelrc)
    echo BAZELRC=$BAZELRC
    echo $CC
    echo $CXX
  '';
}