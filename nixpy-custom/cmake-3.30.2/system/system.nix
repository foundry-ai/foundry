{ nixpkgs }:
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    fetchurl = nixpkgs.fetchurl;
    darwin = nixpkgs.darwin;

    pkg-config = nixpkgs.pkg-config;
    ps = nixpkgs.ps;
    bzip2 = nixpkgs.bzip2;
    zlib = nixpkgs.zlib;
    libuv = nixpkgs.libuv;
    rhash = nixpkgs.rhash;
    openssl = nixpkgs.openssl;
    buildPackages = nixpkgs.buildPackages;
    curlMinimal = nixpkgs.curlMinimal;
    expat = nixpkgs.expat;
    libarchive = nixpkgs.libarchive;
    xz = nixpkgs.xz;

    isMinimalBuild = false;
    useSharedLibraries = false;
    useOpenSSL = true;
    inherit (darwin.apple_sdk.frameworks) CoreServices SystemConfiguration;
in
stdenv.mkDerivation (finalAttrs: {
  pname = "cmake";
  version = "3.30.2";

  src = fetchurl {
    url = "https://cmake.org/files/v${lib.versions.majorMinor finalAttrs.version}/cmake-${finalAttrs.version}.tar.gz";
    hash = "sha256-RgdMeB7M68Qz6Y8Lv6Jlyj/UOB8kXKOxQOdxFTHWDbI=";
  };

  patches = [
    # Don't search in non-Nix locations such as /usr, but do search in our libc.
    ./001-search-path.diff
    # Don't depend on frameworks.
    ./002-application-services.diff
    # Derived from https://github.com/libuv/libuv/commit/1a5d4f08238dd532c3718e210078de1186a5920d
    ./003-libuv-application-services.diff
  ]
  ++ lib.optional stdenv.isCygwin ./004-cygwin.diff
  # Derived from https://github.com/curl/curl/commit/31f631a142d855f069242f3e0c643beec25d1b51
  ++ lib.optional (stdenv.isDarwin && isMinimalBuild) ./005-remove-systemconfiguration-dep.diff
  # On Darwin, always set CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG.
  ++ lib.optional stdenv.isDarwin ./006-darwin-always-set-runtime-c-flag.diff;

  outputs = [ "out" ];
  separateDebugInfo = true;
  setOutputFlags = false;

  setupHooks = [
    ./setup-hook.sh
    ./check-pc-files-hook.sh
  ];

  depsBuildBuild = [ buildPackages.stdenv.cc ];

  nativeBuildInputs = finalAttrs.setupHooks ++ [
    pkg-config
  ];

  buildInputs = lib.optionals useSharedLibraries [
    bzip2
    curlMinimal
    expat
    libarchive
    xz
    zlib
    libuv
    rhash
  ]
  ++ lib.optional useOpenSSL openssl
  ++ lib.optional stdenv.isDarwin CoreServices
  ++ lib.optional (stdenv.isDarwin && !isMinimalBuild) SystemConfiguration;

  propagatedBuildInputs = lib.optional stdenv.isDarwin ps;

  preConfigure = ''
    fixCmakeFiles .
    substituteInPlace Modules/Platform/UnixPaths.cmake \
      --subst-var-by libc_bin ${lib.getBin stdenv.cc.libc} \
      --subst-var-by libc_dev ${lib.getDev stdenv.cc.libc} \
      --subst-var-by libc_lib ${lib.getLib stdenv.cc.libc}
    # CC_FOR_BUILD and CXX_FOR_BUILD are used to bootstrap cmake
    configureFlags="--parallel=''${NIX_BUILD_CORES:-1} CC=$CC_FOR_BUILD CXX=$CXX_FOR_BUILD $configureFlags"
  '';

  # The configuration script is not autoconf-based, although being similar;
  # triples and other interesting info are passed via CMAKE_* environment
  # variables and commandline switches
  configurePlatforms = [ ];

  configureFlags = [
    "CXXFLAGS=-Wno-elaborated-enum-base"
    "--docdir=share/doc/${finalAttrs.pname}-${finalAttrs.version}"
  ] ++ (if useSharedLibraries
        then [
          "--no-system-cppdap"
          "--no-system-jsoncpp"
          "--system-libs"
        ]
        else [
          "--no-system-libs"
        ]) # FIXME: cleanup
  # Workaround https://gitlab.kitware.com/cmake/cmake/-/issues/20568
  ++ lib.optionals stdenv.hostPlatform.is32bit [
    "CFLAGS=-D_FILE_OFFSET_BITS=64"
    "CXXFLAGS=-D_FILE_OFFSET_BITS=64"
  ]
  ++ [
    "--"
    # We should set the proper `CMAKE_SYSTEM_NAME`.
    # http://www.cmake.org/Wiki/CMake_Cross_Compiling
    #
    # Unfortunately cmake seems to expect absolute paths for ar, ranlib, and
    # strip. Otherwise they are taken to be relative to the source root of the
    # package being built.
    (lib.cmakeFeature "CMAKE_CXX_COMPILER" "${stdenv.cc.targetPrefix}c++")
    (lib.cmakeFeature "CMAKE_C_COMPILER" "${stdenv.cc.targetPrefix}cc")
    (lib.cmakeFeature "CMAKE_AR"
      "${lib.getBin stdenv.cc.bintools.bintools}/bin/${stdenv.cc.targetPrefix}ar")
    (lib.cmakeFeature "CMAKE_RANLIB"
      "${lib.getBin stdenv.cc.bintools.bintools}/bin/${stdenv.cc.targetPrefix}ranlib")
    (lib.cmakeFeature "CMAKE_STRIP"
      "${lib.getBin stdenv.cc.bintools.bintools}/bin/${stdenv.cc.targetPrefix}strip")

    (lib.cmakeBool "CMAKE_USE_OPENSSL" useOpenSSL)
    (lib.cmakeBool "BUILD_CursesDialog" false)
  ];
  dontUseCmakeConfigure = true;
  enableParallelBuilding = true;

  # `pkgsCross.musl64.cmake.override { stdenv = pkgsCross.musl64.llvmPackages_16.libcxxStdenv; }`
  # fails with `The C++ compiler does not support C++11 (e.g.  std::unique_ptr).`
  # The cause is a compiler warning `warning: argument unused during compilation: '-pie' [-Wunused-command-line-argument]`
  # interfering with the feature check.
  env.NIX_CFLAGS_COMPILE = "-Wno-unused-command-line-argument";
  # make install attempts to use the just-built cmake
  preInstall = lib.optionalString (stdenv.hostPlatform != stdenv.buildPlatform) ''
    sed -i 's|bin/cmake|${buildPackages.cmakeMinimal}/bin/cmake|g' Makefile
  '';
  doCheck = false; # fails
})
