load("@rules_python//python:repositories.bzl", "python_register_toolchains")

def python_init_toolchains():
    python_register_toolchains(
        name = "python",
        # By default assume the interpreter is on the local file system, replace
        # with proper URL if it is not the case.
        base_url = "file://",
        ignore_root_user_error = True,
        python_version = "@PYTHON_VERSION@",
        tool_versions = {
            "@PYTHON_VERSION@": {
                # Path to .tar.gz with Python binary.
                "url": "@PYTHON_TAR_PATH@",
                "sha256": {
                    # By default we assume Linux x86_64 architecture, eplace with
                    # proper architecture if you were building on a different platform.
                    "x86_64-unknown-linux-gnu": "@PYTHON_SHA256@"
                },
            },
        },
    )