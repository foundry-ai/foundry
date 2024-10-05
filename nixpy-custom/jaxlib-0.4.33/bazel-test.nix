{ pkgs }:
let 
  effectiveStdenv = pkgs.overrideCC 
    pkgs.stdenv pkgs.llvmPackages_18.clang;
  buildBazelPackage = pkgs.buildBazelPackage.override { 
    stdenv = effectiveStdenv; 
  };
in
  (buildBazelPackage {
    name = "bazel-test-build";
    version = "";

    bazel = pkgs.bazel_6;
    src = ./test-package;
    postPatch = ''
      cp -r ${./toolchain} toolchain
    '';
    bazelTargets = [
      "//:hello"
      "//:foo"
    ];
    bazelFlags = [ 
      "--subcommands"
      "--extra_toolchains=//toolchain:cc_nix_toolchain"
      "--verbose_failures"
      "--toolchain_resolution_debug=\"//:hello\""
      "--cxxopt=-x"
      "--cxxopt=c++"
      "--host_cxxopt=-x"
      "--host_cxxopt=c++"
    ];
    fetchAttrs = {
      sha256 = "sha256-TbWcWYidyXuAMgBnO2/k0NKCzc4wThf2uUeC3QxdBJY=";
    };
    buildAttrs = {
    };
    preInstall = ''
      exit 1
    '';
  }).overrideAttrs {
    nativeBuildInputs = [pkgs.bazel_6];
  }