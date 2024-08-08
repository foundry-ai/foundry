{buildPythonPackage, build-system, dependencies, 
        nixpkgs, python, fetchurl} : buildPythonPackage {
    pname = "meson-python";
    version = "0.13.1";
    format="wheel";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/9f/af/5f941f57dc516e72b018183a38fbcfb018a7e83afd3c756ecfba82f21c65/meson_python-0.13.1-py3-none-any.whl";
        hash="sha256-4z6j77rezBV2jCBdA7kFx7O/cq+uHh69hLQ4xKPtM5M=";
    };
    dependencies = [
        build-system.pyproject-metadata 
        build-system.meson 
        build-system.ninja] ++ (
        if builtins.hasAttr "tomli" build-system then [build-system.tomli]
        else []
    );
    doCheck = false;
}