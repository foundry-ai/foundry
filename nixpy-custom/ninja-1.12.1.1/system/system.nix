{ nixpkgs }:
let
    lib = nixpkgs.lib;
    stdenv = nixpkgs.stdenv;
    fetchFromGitHub = nixpkgs.fetchFromGitHub;
    fetchpatch = nixpkgs.fetchpatch;
    buildPackages = nixpkgs.buildPackages;
    re2c = nixpkgs.re2c;
    installShellFiles = nixpkgs.installShellFiles;
    buildDocs = false;
in
stdenv.mkDerivation rec {
  pname = "ninja";
  version = "1.12.1";

  src = fetchFromGitHub {
    owner = "ninja-build";
    repo = "ninja";
    rev = "v${version}";
    hash = "sha256-RT5u+TDvWxG5EVQEYj931EZyrHUSAqK73OKDAascAwA=";
  };

  depsBuildBuild = [ buildPackages.stdenv.cc ];

  nativeBuildInputs = [
    nixpkgs.python3
    re2c
    installShellFiles
  ];

  patches = lib.optionals stdenv.is32bit [
    # Otherwise ninja may fail on some files in a larger FS.
    (fetchpatch {
      name = "stat64.patch";
      url = "https://github.com/ninja-build/ninja/commit/7bba11ae704efc84cac5fde5e9be53f653f237d1.diff";
      hash = "sha256-tINS57xLh1lwnYFWCQs5OudfgtIShaOh5zbmv7w5BnQ=";
    })
  ];

  postPatch = ''
    # write rebuild args to file after bootstrap
    substituteInPlace configure.py --replace "subprocess.check_call(rebuild_args)" "open('rebuild_args','w').write(rebuild_args[0])"
  '';

  buildPhase = ''
    runHook preBuild

    # for list of env vars
    # see https://github.com/ninja-build/ninja/blob/v1.11.1/configure.py#L264
    CXX="$CXX_FOR_BUILD" \
    AR="$AR_FOR_BUILD" \
    CFLAGS="$CFLAGS_FOR_BUILD" \
    CXXFLAGS="$CXXFLAGS_FOR_BUILD" \
    LDFLAGS="$LDFLAGS_FOR_BUILD" \
    python configure.py --bootstrap
    python configure.py

    source rebuild_args
  '' + lib.optionalString buildDocs ''
    # "./ninja -vn manual" output copied here to support cross compilation.
    asciidoc -b docbook -d book -o build/manual.xml doc/manual.asciidoc
    xsltproc --nonet doc/docbook.xsl build/manual.xml > doc/manual.html
  '' + ''

    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall

    install -Dm555 -t $out/bin ninja
    installShellCompletion --name ninja \
      --bash misc/bash-completion \
      --zsh misc/zsh-completion
  '' + lib.optionalString buildDocs ''
    install -Dm444 -t $out/share/doc/ninja doc/manual.asciidoc doc/manual.html
  '' + ''
    runHook postInstall
  '';

# No setuphook for this packaged build
#   setupHook = ./setup-hook.sh;

  meta = with lib; {
    description = "Small build system with a focus on speed";
    mainProgram = "ninja";
    homepage = "https://ninja-build.org/";
    license = licenses.asl20;
    platforms = platforms.unix;
    maintainers = with maintainers; [ thoughtpolice bjornfor orivej ];
  };
}