{buildPythonPackage, build-system, dependencies, 
        nixpkgs, python, fetchurl} : buildPythonPackage {
    pname = "meson-python";
    version = "0.15.0";
    format="wheel";
    src = fetchurl {
        url="https://files.pythonhosted.org/packages/1f/60/b10b11ab470a690d5777310d6cfd1c9bdbbb0a1313a78c34a1e82e0b9d27/meson_python-0.15.0-py3-none-any.whl";
        hash="sha256-OuOCU/8CsulHoF42Ki6vWpoJ0TPFZmtBIzme5fvy5ZE=";
    };
    dependencies = [
        build-system.pyproject-metadata 
        build-system.meson 
        build-system.ninja] ++ (
        if builtins.hasAttr "tomli" build-system then [build-system.tomli]
        else []
    );
    setupHooks = [ ./add-build-flags.sh ];
    doCheck = false;
}