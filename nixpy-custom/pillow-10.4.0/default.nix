{buildPythonPackage, build-system, dependencies, nixpkgs, python, fetchurl} : 
    let 
        lib = nixpkgs.lib;
        libxcb = nixpkgs.xorg.libxcb;
        setuptools = build-system.setuptools;
    in
    buildPythonPackage rec {
        pname = "pillow";
        version = "10.4.0";
        format="pyproject";
        src = fetchurl {
            url="https://files.pythonhosted.org/packages/cd/74/ad3d526f3bf7b6d3f408b73fde271ec69dfac8b81341a318ce825f2b3812/pillow-10.4.0.tar.gz";
            hash="sha256-Fmwc1NJDCbMNYfefSpEUt7IxPXRQkSJ3hV/139fNSgY=";
        };
        build-system = [setuptools];
        buildInputs =
            with nixpkgs; [
                freetype
                libjpeg
                openjpeg
                libimagequant
                zlib
                libtiff
                libwebp
                libxcrypt
                tcl
                lcms2
            ] ++ lib.optionals (lib.versionAtLeast version "7.1.0") [ libxcb ];
        preConfigure =
            let
                libinclude' = pkg: ''"${pkg.out}/lib", "${pkg.out}/include"'';
                libinclude = pkg: ''"${pkg.out}/lib", "${pkg.dev}/include"'';
            in with nixpkgs;
            ''
            sed -i "setup.py" \
                -e 's|^FREETYPE_ROOT =.*$|FREETYPE_ROOT = ${libinclude freetype}|g ;
                    s|^JPEG_ROOT =.*$|JPEG_ROOT = ${libinclude libjpeg}|g ;
                    s|^JPEG2K_ROOT =.*$|JPEG2K_ROOT = ${libinclude openjpeg}|g ;
                    s|^IMAGEQUANT_ROOT =.*$|IMAGEQUANT_ROOT = ${libinclude' libimagequant}|g ;
                    s|^ZLIB_ROOT =.*$|ZLIB_ROOT = ${libinclude zlib}|g ;
                    s|^LCMS_ROOT =.*$|LCMS_ROOT = ${libinclude lcms2}|g ;
                    s|^TIFF_ROOT =.*$|TIFF_ROOT = ${libinclude libtiff}|g ;
                    s|^TCL_ROOT=.*$|TCL_ROOT = ${libinclude' tcl}|g ;
                    s|self\.disable_platform_guessing = None|self.disable_platform_guessing = True|g ;'
            export LDFLAGS="$LDFLAGS -L${libwebp}/lib"
            export CFLAGS="$CFLAGS -I${libwebp}/include"
            ''
            + lib.optionalString (lib.versionAtLeast version "7.1.0") ''
            export LDFLAGS="$LDFLAGS -L${libxcb}/lib"
            export CFLAGS="$CFLAGS -I${libxcb.dev}/include"
            '';
    }