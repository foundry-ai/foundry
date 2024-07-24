{buildPythonPackage, env, nixpkgs, python, fetchurl} : buildPythonPackage {
    pname = "pillow";
    version = "10.4.0";
    format="pyproject";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/cd/74/ad3d526f3bf7b6d3f408b73fde271ec69dfac8b81341a318ce825f2b3812/pillow-10.4.0.tar.gz";
        hash="sha256-Fmwc1NJDCbMNYfefSpEUt7IxPXRQkSJ3hV/139fNSgY=";
    };
    build-system = [];
}