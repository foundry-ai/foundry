package(default_visibility = ["//visibility:public"])

load(":nix_toolchain_config.bzl", "cc_nix_toolchain_config")
cc_nix_toolchain_config(name = "nix_toolchain_config")

cc_toolchain(
    name = "nix_toolchain",
    toolchain_identifier = "nix-toolchain",
    toolchain_config = ":nix_toolchain_config",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
)

toolchain(
    name = "cc_nix_toolchain",
    toolchain = ":nix_toolchain",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)