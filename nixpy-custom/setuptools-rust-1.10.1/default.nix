{buildPythonPackage, build-system, 
dependencies, nixpkgs, python, fetchurl} : 
let setuptools = build-system.setuptools;
    setuptools-scm = build-system.setuptools-scm;

    semantic-version = dependencies.semantic-version;
    setuptools_dep = dependencies.setuptools;
    fetchPypi = python.pkgs.fetchPypi;
        in
    buildPythonPackage rec {
        pname = "setuptools-rust";
        version = "1.10.1";
        format="pyproject";
        src = fetchurl {
            url = "https://files.pythonhosted.org/packages/b8/86/4f34594f21f529623b8650fe729548e3a2ad6c9ad81583391f03f74dd11a/setuptools_rust-1.10.1.tar.gz";
            hash = "sha256-15A1/FTN+TQunt9LAJSR7KsGw6ZSs3w8E3x7qFVH0+Y=";
        };
        preCheck = ''
        cd $out
        '';
        build-system = [setuptools setuptools-scm];
        dependencies = [semantic-version setuptools_dep];
    }