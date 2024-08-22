# Produces a minimal cudatoolkit
{ nixpkgs }:
let stdenv = nixpkgs.stdenv;
    lib = nixpkgs.lib;
    autoPatchelfHook = nixpkgs.autoPatchelfHook;

    cuda-version = "12.2.2";
    cudnn-version = "9.3.0";
    cuda-capabilities = ["5.0" "6.0" "7.0" "8.0" "8.6"];

    toolkit-releases = {
        "12.2.2" = {
            x86_64-linux = {
                url = "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run";
                hash = "sha256-Kzmq4+dhjZ9Zo8j6HxvGHynAsODfdfsFB2uts1KVLvI=";
            };
            powerpc64le-linux = {
                url = "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux_ppc64le.run";
                hash = "sha256-GISCeOfyvUtEgfVmVjPX49RumlYtF11f8nghgYiwE0I=";
            };
        };
    };
    cudnn-releases = {
        "9.0.0" = {
            powerpc64le-linux = {
                url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-ppc64le/cudnn-linux-ppc64le-9.0.0.312_cuda12-archive.tar.xz";
                hash = "sha256-uO9vJJEo4ZhYk6h4eiHeNcuD7EfG3G/RgJBh3Zo/+yA=";
            };
            x86_64-linux = {
                url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz";
                hash = "sha256-04kOYJ1lMO5biP+VtgyOaxwex/qWbsUzkl8g+Jb8xjA=";
            };
        };
        "9.3.0" = {
            x86_64-linux = {
                url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz";
                hash = "sha256-PW7xCqBtyTOaR34rBX4IX/hQC73ueeQsfhNlXJ7/LCY=";
            };

        };

    };
    toolkit-release = toolkit-releases."${cuda-version}"."${nixpkgs.system}";
    cudnn-release = cudnn-releases."${cudnn-version}"."${nixpkgs.system}";

    arch = {
        "powerpc64le-linux" = "ppc64le";
        "x86_64-linux" = "x86_64";
    }."${nixpkgs.system}";

    dropDot = ver: builtins.replaceStrings [ "." ] [ "" ] ver;
    archMapper = feat: lib.lists.map (computeCapability: "${feat}_${dropDot computeCapability}");
    gencodeMapper =
        feat: lib.lists.map (
            computeCapability:
            "-gencode=arch=compute_${dropDot computeCapability},code=${feat}_${dropDot computeCapability}"
        );
    flags = rec {
        # realArches :: List String
        # The real architectures are physical architectures supported by the CUDA version.
        # E.g. [ "sm_75" "sm_86" ]
        realArches = archMapper "sm" cuda-capabilities;

        # virtualArches :: List String
        # The virtual architectures are typically used for forward compatibility, when trying to support
        # an architecture newer than the CUDA version allows.
        # E.g. [ "compute_75" "compute_86" ]
        virtualArches = archMapper "compute" cuda-capabilities;

        # arches :: List String
        # By default, build for all supported architectures and forward compatibility via a virtual
        # architecture for the newest supported architecture.
        # E.g. [ "sm_75" "sm_86" "compute_86" ]
        arches = realArches ++ [lib.lists.last virtualArches];

        # gencode :: List String
        # A list of CUDA gencode arguments to pass to NVCC.
        # E.g. [ "-gencode=arch=compute_75,code=sm_75" ... "-gencode=arch=compute_86,code=compute_86" ]
        gencode = let
                base = gencodeMapper "sm" cuda-capabilities;
                forward = gencodeMapper "compute" [ (lib.lists.last cuda-capabilities) ];
            in base ++ forward;
        gencodeString = lib.strings.concatStringsSep " " gencode;
    };
in rec {
    cudaVersion = cuda-version;
    cudnnVersion = cudnn-version;
    cudaCapabilities = cuda-capabilities;
    cudaFlags = flags;
    cudaStdenv = nixpkgs.stdenvAdapters.useLibsFrom nixpkgs.stdenv nixpkgs."gcc12Stdenv";
    cudatoolkit=stdenv.mkDerivation {
        dontPatchELF = true;
        dontStrip = true;
        name = "cudatoolkit";
        version = cuda-version;

        src = nixpkgs.fetchurl {
            url = toolkit-release.url;
            hash = toolkit-release.hash;
        };
        nativeBuildInputs = [
            autoPatchelfHook
        ];
        buildInputs = lib.optionals stdenv.isx86_64 [
            nixpkgs.rdma-core
        ];
        preFixup = ''
            addAutoPatchelfSearchPath ${placeholder "out"}
            addAutoPatchelfSearchPath ${placeholder "out"}/nvvm
            addAutoPatchelfSearchPath ${placeholder "out"}/lib64
            addAutoPatchelfSearchPath ${placeholder "out"}/nvvm/lib64
            addAutoPatchelfSearchPath ${lib.getLib stdenv.cc.cc}
        '';
        autoPatchelfIgnoreMissingDeps = [
            "libcuda.so.1"
            "libcom_err.so.2"
        ];
        outputs = [
            "out"
            "doc"
        ];
        unpackPhase = ''
            sh $src --keep --noexec
        '';
        installPhase = ''
            runHook preInstall

            target=$out/targets/${arch}-linux/
            lib64=$target/lib64
            include=$target/include

            mkdir -p $out $out/bin $lib64 $include $target $doc
            ln -s $lib64 $out/lib64
            ln -s $include $out/include

            pushd .
            for dir in pkg/builds/* pkg/builds/cuda_nvcc/nvvm pkg/builds/cuda_cupti/extras/CUPTI; do
                if [ -d $dir/bin ]; then
                    mv $dir/bin/* $out/bin
                fi
                if [ -d $dir/doc ]; then
                    (cd $dir/doc && find . -type d -exec mkdir -p $doc/\{} \;)
                    (cd $dir/doc && find . \( -type f -o -type l \) -exec mv \{} $doc/\{} \;)
                fi
                if [ -L $dir/include ] || [ -d $dir/include ]; then
                    (cd $dir/include && find . -type d -exec mkdir -p $include/\{} \;)
                    (cd $dir/include && find . \( -type f -o -type l \) -exec mv \{} $include/\{} \;)
                fi
                if [ -L $dir/lib64 ] || [ -d $dir/lib64 ]; then
                    (cd $dir/lib64 && find . -type d -exec mkdir -p $lib64/\{} \;)
                    (cd $dir/lib64 && find . \( -type f -o -type l \) -exec mv \{} $lib64/\{} \;)
                fi
            done
            popd
            mv pkg/builds/cuda_nvcc/nvvm/ $out/nvvm
            mv pkg/builds/cuda_sanitizer_api $out/cuda_sanitizer_api
            ln -s $out/cuda_sanitizer_api/compute-sanitizer/compute-sanitizer $out/bin/compute-sanitizer

            # Change the #error on GCC > 4.9 to a #warning.
            sed -i $out/include/host_config.h -e 's/#error\(.*unsupported GNU version\)/#warning\1/'
            sed -i $out/include/crt/host_config.h -e 's/#error\(.*unsupported GNU version\)/#warning\1/'

            # Fix builds with newer glibc version
            sed -i "1 i#define _BITS_FLOATN_H" "$out/include/host_defines.h"

            # remove for now (we can't patchelf this...)
            rm $out/bin/cuda-gdb

            # sometimes this gets created?
            rm -f $out/include/include

            rm $out/bin/cuda-uninstaller
            runHook postInstall
        '';
        doInstallCheck = false;
        postInstallCheck = ''
            # Smoke test binaries
            pushd $out/bin
            for f in *; do
            case $f in
                crt)                           continue;;
                nvcc.profile)                  continue;;
                nsight_ee_plugins_manage.sh)   continue;;
                uninstall_cuda_toolkit_6.5.pl) continue;;
                computeprof|nvvp|nsight)       continue;; # GUIs don't feature "--version"
                *)                             echo "Executing '$f --version':"; ./$f --version;;
            esac
            done
            popd
        '';
    };
    cudnn = stdenv.mkDerivation {
        name = "cudnn";
        version = cudnn-version;

        src = nixpkgs.fetchurl {
            url = cudnn-release.url;
            hash = cudnn-release.hash;
        };
        doInstallCheck = false;

        nativeBuildInputs = [
            autoPatchelfHook
        ];
        buildInputs = [
            nixpkgs.libz
        ];
        preFixup = ''
            addAutoPatchelfSearchPath ${placeholder "out"}
            addAutoPatchelfSearchPath ${placeholder "out"}/lib
            addAutoPatchelfSearchPath ${lib.getLib stdenv.cc.cc}
        '';
        autoPatchelfIgnoreMissingDeps = [
            "libcuda.so.1"
            "libcom_err.so.2"
        ];

        installPhase = ''
            echo "Moving files to output..."
            mkdir $out
            mv * $out
        '';
    };
    nccl = cudaStdenv.mkDerivation (finalAttrs: {
        pname = "nccl";
        version = "2.21.5-1";
        src = nixpkgs.fetchFromGitHub {
            owner = "NVIDIA";
            repo = "nccl";
            rev = "v${finalAttrs.version}";
            hash = "sha256-IF2tILwW8XnzSmfn7N1CO7jXL95gUp02guIW5n1eaig=";
        };
        __structuredAttrs = true;
        strictDeps = true;

        outputs = [
            "out"
            "dev"
        ];

        nativeBuildInputs = [
            nixpkgs.which
            nixpkgs.python3
            cudatoolkit
        ];

        buildInputs = [ cudatoolkit ];

        env.NIX_CFLAGS_COMPILE = toString (
            [ "-Wno-unused-function" "-L${cudatoolkit}/lib64"  ] ++
            lib.optionals nixpkgs.stdenv.hostPlatform.isPower64
            [ "-U__LONG_DOUBLE_IEEE128__"]
        );

        postPatch = ''
            patchShebangs ./src/device/generate.py
        '';

        makeFlagsArray = [
            "PREFIX=$(out)"
            "NVCC_GENCODE=${cudaFlags.gencodeString}"
            "CUDA_HOME=${cudatoolkit}"
            "CUDA_LIB=${cudatoolkit}/lib"
            "CUDA_INC=${cudatoolkit}/include"
        ];

        enableParallelBuilding = true;

        postFixup = ''
            moveToOutput lib/libnccl_static.a $dev
        '';
    });
}