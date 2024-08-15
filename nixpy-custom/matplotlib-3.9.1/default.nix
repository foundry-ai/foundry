{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    Cocoa = nixpkgs.darwin.apple_sdk.frameworks.Cocoa;

    meson-python = python.pkgs.meson-python;
    pybind11 = build-system.pybind11;
    setuptools-scm = build-system.setuptools-scm;
    numpy = build-system.numpy;
    fetchPypi = python.pkgs.fetchPypi;

    # numpy dependency is potentially different from the numpy build-system!
    numpy_dep = dependencies.numpy;
    contourpy = dependencies.contourpy;
    cycler = dependencies.cycler;
    fonttools = dependencies.fonttools;
    kiwisolver = dependencies.kiwisolver;
    packaging = dependencies.packaging;
    pillow = dependencies.pillow;
    pyparsing = dependencies.pyparsing;
    python-dateutil = dependencies.python-dateutil;

    gtk3 = nixpkgs.gtk3;
    cairo = nixpkgs.cairo;
    libX11 = nixpkgs.libX11;
    ffmpeg-headless = nixpkgs.ffmpeg-headless;
    freetype = nixpkgs.freetype;
    # freetype_old = freetype.overrideAttrs (_: {
    #     src = fetchurl {
    #     url = "https://download.savannah.gnu.org/releases/freetype/freetype-old/freetype-2.6.1.tar.gz";
    #     sha256 = "sha256-Cjx9+9ptoej84pIy6OltmHq6u79x68jHVlnkEyw2cBQ=";
    #     };
    #     patches = [ ];
    # });
    qhull = nixpkgs.qhull;
    gobject-introspection = nixpkgs.gobject-introspection;
    pkg-config = nixpkgs.pkg-config;
    ghostscript = nixpkgs.ghostscript;

    sage = python.pkgs.sage;

    enableTk = false;
    enableGtk3 = false;
    enableGhostscript = false;
in
buildPythonPackage rec {
    pname = "matplotlib";
    version = "3.9.1";
    format="pyproject";
    src = fetchPypi {
        inherit pname version;
        hash = "sha256-3gaxm425XdM9DcF8kmx8nr7Z9XIHS2+sT2UGimgU0BA=";
    };
    build-system = [setuptools-scm pybind11 meson-python numpy];
    dependencies = [contourpy cycler fonttools kiwisolver numpy_dep packaging pillow pyparsing python-dateutil];

    nativeBuildInputs = [ pkg-config ] ++ lib.optionals enableGtk3 [ gobject-introspection ];

    buildInputs = [
            ffmpeg-headless
            freetype
            qhull
    ] ++ lib.optionals enableGhostscript [ ghostscript ]
      ++ lib.optionals enableGtk3 [
        cairo
        gtk3
    ] ++ lib.optionals enableTk [
        # libX11
        # tcl
        # tk
    ] ++ lib.optionals stdenv.isDarwin [ Cocoa ];

    mesonFlags = lib.mapAttrsToList lib.mesonBool {
        system-freetype = true;
        system-qhull = true;
        # Otherwise GNU's `ar` binary fails to put symbols from libagg into the
        # matplotlib shared objects. See:
        # -https://github.com/matplotlib/matplotlib/issues/28260#issuecomment-2146243663
        # -https://github.com/matplotlib/matplotlib/issues/28357#issuecomment-2155350739
        b_lto = false;
    };

    # clang-11: error: argument unused during compilation: '-fno-strict-overflow' [-Werror,-Wunused-command-line-argument]
    hardeningDisable = lib.optionals stdenv.isDarwin [ "strictoverflow" ];
    doCheck = false;
}