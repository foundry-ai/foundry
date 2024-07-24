{buildPythonPackage, env, nixpkgs, python, fetchurl} : buildPythonPackage {
    name = "ml_dtypes";
    version = "0.4.0";
    src = fetchurl {
        url="https://github.com/jax-ml/ml_dtypes/archive/b157c19cc98da40a754109993e02d7eab3d75358.zip";
        hash="sha256-7pfKaYnNLWw8ZHaXWX6EprdQru4/OdApfuepJiNvoEg=";
    };
    doCheck = false;
}
