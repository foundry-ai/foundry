{buildPythonPackage, fetchurl, nixpkgs, python}: rec {
  packages = rec {
    regex = buildPythonPackage {
      pname = "regex";
      version = "2024.7.24";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3f/51/64256d0dc72816a4fe3779449627c69ec8fee5a5625fd60ba048f53b3478/regex-2024.7.24.tar.gz";
        hash="sha256-nP0Anu0aRrJ8FAOa1bvF5xtjZ8Wy5tX12g6pFgCBdQY=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    sentry-sdk = buildPythonPackage {
      pname = "sentry-sdk";
      version = "2.11.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/bc/3c/a8ab3309d22c1d7142f811882e7d45449696f87c6e4e723b1433b6069b84/sentry_sdk-2.11.0-py2.py3-none-any.whl";
        hash="sha256-2WRxDi2+AV2dxP8K0WIl1ow7NpNrdCpv4FBFZbdgo7c=";
      };
      dependencies = with packages;
      with buildPackages;
      [certifi urllib3];
      doCheck = false;
    };
    fonttools = buildPythonPackage {
      pname = "fonttools";
      version = "4.53.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e4/b9/0394d67056d4ad36a3807b439571934b318f1df925593a95e9ec0516b1a7/fonttools-4.53.1-py3-none-any.whl";
        hash="sha256-8fh1iirREL1kMiA6NEJp9EWikH3CTva8z9CsThTg1x0=";
      };
      doCheck = false;
    };
    urllib3 = buildPythonPackage {
      pname = "urllib3";
      version = "2.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ca/1c/89ffc63a9605b583d5df2be791a27bc1a42b7c32bab68d3c8f2f73a98cd4/urllib3-2.2.2-py3-none-any.whl";
        hash="sha256-pEiy9k1oYVVGgDfhrOny0hmXduF/CkZhBIDTEfc+NHI=";
      };
      doCheck = false;
    };
    certifi = buildPythonPackage {
      pname = "certifi";
      version = "2024.7.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1c/d5/c84e1a17bf61d4df64ca866a1c9a913874b4e9bdc131ec689a0ad013fb36/certifi-2024.7.4-py3-none-any.whl";
        hash="sha256-wZjiGxKJwquF7k5nu0tO8+rQiSBZkBqNW2IvJKEQHpA=";
      };
      doCheck = false;
    };
    charset-normalizer = buildPythonPackage {
      pname = "charset-normalizer";
      version = "3.3.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/28/76/e6222113b83e3622caa4bb41032d0b1bf785250607392e1b778aca0b8a7d/charset_normalizer-3.3.2-py3-none-any.whl";
        hash="sha256-Pk0fZYcyLSeIg2qZxpBi+7CRMx7JQOAtEtF5wdU+Jfw=";
      };
      doCheck = false;
    };
    idna = buildPythonPackage {
      pname = "idna";
      version = "3.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e5/3e/741d8c82801c347547f8a2a06aa57dbb1992be9e948df2ea0eda2c8b79e8/idna-3.7-py3-none-any.whl";
        hash="sha256-gv7h/Hit1DSS06GJi/ptipBMyX2EJ/aD7Y55jQd2GqA=";
      };
      doCheck = false;
    };
    requests = buildPythonPackage {
      pname = "requests";
      version = "2.32.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f9/9b/335f9764261e915ed497fcdeb11df5dfd6f7bf257d4a6a2a686d80da4d54/requests-2.32.3-py3-none-any.whl";
        hash="sha256-cHYc/gPHc86yKqL2cbR1eXYUUXXN/KA4wCZU0GHW3MY=";
      };
      dependencies = with packages;
      with buildPackages;
      [certifi charset-normalizer idna urllib3];
      doCheck = false;
    };
    smmap = buildPythonPackage {
      pname = "smmap";
      version = "5.0.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a7/a5/10f97f73544edcdef54409f1d839f6049a0d79df68adbc1ceb24d1aaca42/smmap-5.0.1-py3-none-any.whl";
        hash="sha256-5thmj6X5PnBpNKYte02xnI2euM8q27de8bZ1qjMrado=";
      };
      doCheck = false;
    };
    gitdb = buildPythonPackage {
      pname = "gitdb";
      version = "4.0.11";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fd/5b/8f0c4a5bb9fd491c277c21eff7ccae71b47d43c4446c9d0c6cff2fe8c2c4/gitdb-4.0.11-py3-none-any.whl";
        hash="sha256-gaNAfd0u6N9ETLrOoA4tA45AFQrPowAWlv4Nzx0636Q=";
      };
      dependencies = with packages;
      with buildPackages;
      [smmap];
      doCheck = false;
    };
    gitpython = buildPythonPackage {
      pname = "gitpython";
      version = "3.1.43";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e9/bd/cc3a402a6439c15c3d4294333e13042b915bbeab54edc457c723931fed3f/GitPython-3.1.43-py3-none-any.whl";
        hash="sha256-7sfsVrkqrXUfmRKnNAS8ArohKiOtsscJjuZoQXBRof8=";
      };
      dependencies = with packages;
      with buildPackages;
      [gitdb];
      doCheck = false;
    };
    rpds-py = buildPythonPackage {
      pname = "rpds-py";
      version = "0.19.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/2f/fe/5217efe981c2ae8647b503ba3b8f55efc837df62f63667572b4bb75b30bc/rpds_py-0.19.1.tar.gz";
        hash="sha256-Md1XlIN/ALRvQJaqjMqlly9zqTiYLjLtgXu1IMRl5SA=";
      };
      build-system = with packages;
      with buildPackages;
      [maturin-1_7_0-824706860354622439a169eb472a146420fe82faf0e53565cd55e420a643ff24];
      doCheck = false;
    };
    referencing = buildPythonPackage {
      pname = "referencing";
      version = "0.35.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/59/2056f61236782a2c86b33906c025d4f4a0b17be0161b63b70fd9e8775d36/referencing-0.35.1-py3-none-any.whl";
        hash="sha256-7abTI01igU0cZOMFwTMcmjphMtpHWrY4LqqZeyHudd4=";
      };
      dependencies = with packages;
      with buildPackages;
      [attrs rpds-py];
      doCheck = false;
    };
    jsonschema-specifications = buildPythonPackage {
      pname = "jsonschema-specifications";
      version = "2023.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ee/07/44bd408781594c4d0a027666ef27fab1e441b109dc3b76b4f836f8fd04fe/jsonschema_specifications-2023.12.1-py3-none-any.whl";
        hash="sha256-h+T986lIWLiiuid42bpX2KnK/KfHSJxGug0wqLxqnDw=";
      };
      dependencies = with packages;
      with buildPackages;
      [referencing];
      doCheck = false;
    };
    attrs = buildPythonPackage {
      pname = "attrs";
      version = "23.2.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e0/44/827b2a91a5816512fcaf3cc4ebc465ccd5d598c45cefa6703fcf4a79018f/attrs-23.2.0-py3-none-any.whl";
        hash="sha256-mbh6SFpYILI7h58EwjBbRLlRtQL9ZL6RWHnXen6PxvE=";
      };
      doCheck = false;
    };
    jsonschema = buildPythonPackage {
      pname = "jsonschema";
      version = "4.23.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/69/4a/4f9dbeb84e8850557c02365a0eee0649abe5eb1d84af92a25731c6c0f922/jsonschema-4.23.0-py3-none-any.whl";
        hash="sha256-+622+LFEqPjPnwuJupRQHRQ+UEEaEnhjP1anrPf9VWY=";
      };
      dependencies = with packages;
      with buildPackages;
      [attrs jsonschema-specifications referencing rpds-py];
      doCheck = false;
    };
    tqdm = buildPythonPackage {
      pname = "tqdm";
      version = "4.66.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/18/eb/fdb7eb9e48b7b02554e1664afd3bd3f117f6b6d6c5881438a0b055554f9b/tqdm-4.66.4-py3-none-any.whl";
        hash="sha256-t1yla0E7AwvD8Ar1H9LBoaXqxqDBzKg8uzelxSq85kQ=";
      };
      doCheck = false;
    };
    protobuf = buildPythonPackage {
      pname = "protobuf";
      version = "5.27.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3a/fa/4c3ac5527ed2e5f3577167ecd5f8180ffcdc8bdd59c9f143409c19706456/protobuf-5.27.2-py3-none-any.whl";
        hash="sha256-VDMPB+SUnQlhRwfEiwbRoi+P+1djwVnv1cCSgyapFHA=";
      };
      doCheck = false;
    };
    psutil = buildPythonPackage {
      pname = "psutil";
      version = "6.0.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/18/c7/8c6872f7372eb6a6b2e4708b88419fb46b857f7a2e1892966b851cc79fc9/psutil-6.0.0.tar.gz";
        hash="sha256-j6rk8xC22Wn6JsoFRTOLIfc8axXbfEqNk0pUgvqoGPI=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      doCheck = false;
    };
    soupsieve = buildPythonPackage {
      pname = "soupsieve";
      version = "2.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/4c/f3/038b302fdfbe3be7da016777069f26ceefe11a681055ea1f7817546508e3/soupsieve-2.5-py3-none-any.whl";
        hash="sha256-6qM3/1WhV5tlSdxnlWXqwePQAFY7yxyKsND++8DCzcc=";
      };
      doCheck = false;
    };
    pyparsing = buildPythonPackage {
      pname = "pyparsing";
      version = "3.1.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9d/ea/6d76df31432a0e6fdf81681a895f009a4bb47b3c39036db3e1b528191d52/pyparsing-3.1.2-py3-none-any.whl";
        hash="sha256-+dt1kRgB7XeP5hu2Qwef+GYBrKmfyuY0WqZykgOPt0I=";
      };
      doCheck = false;
    };
    traitlets = buildPythonPackage {
      pname = "traitlets";
      version = "5.14.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/00/c0/8f5d070730d7836adc9c9b6408dec68c6ced86b304a9b26a14df072a6e8c/traitlets-5.14.3-py3-none-any.whl";
        hash="sha256-t06J45ex7SjMgx23rqdZumZAyz3hMJDKFFQmaI/xrE8=";
      };
      doCheck = false;
    };
    jupyter-core = buildPythonPackage {
      pname = "jupyter-core";
      version = "5.7.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c9/fb/108ecd1fe961941959ad0ee4e12ee7b8b1477247f30b1fdfd83ceaf017f0/jupyter_core-5.7.2-py3-none-any.whl";
        hash="sha256-T3MV0va0vPLj58tuRncuunYK5FnNH1nSnrV7CgG9dAk=";
      };
      dependencies = with packages;
      with buildPackages;
      [platformdirs traitlets];
      doCheck = false;
    };
    pytz = buildPythonPackage {
      pname = "pytz";
      version = "2024.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9c/3d/a121f284241f08268b21359bd425f7d4825cffc5ac5cd0e1b3d82ffd2b10/pytz-2024.1-py2.py3-none-any.whl";
        hash="sha256-MoFx9ONiMTnaSYNFGVCyjpWscG4T8/JjCoeXSeeosxk=";
      };
      doCheck = false;
    };
    pyobjc-framework-coretext = buildPythonPackage {
      pname = "pyobjc-framework-coretext";
      version = "10.3.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9e/9f/d363cb1548808f538d7ae267a9fcb999dfb5693056fdaa5bc93de089cfef/pyobjc_framework_coretext-10.3.1.tar.gz";
        hash="sha256-uPotUHjtd0QxrmS6iGFW4xmuwLjGzCPav9hneCZbQW8=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [pyobjc-core pyobjc-framework-cocoa pyobjc-framework-quartz];
      doCheck = false;
    };
    pyobjc-framework-cocoa = buildPythonPackage {
      pname = "pyobjc-framework-cocoa";
      version = "10.3.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a7/6c/b62e31e6e00f24e70b62f680e35a0d663ba14ff7601ae591b5d20e251161/pyobjc_framework_cocoa-10.3.1.tar.gz";
        hash="sha256-HPIHFNqqmGtIj7YtaXEwSfY1ydQaYMjal9g1cQRFKBo=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [pyobjc-core];
      doCheck = false;
    };
    pyobjc-core = buildPythonPackage {
      pname = "pyobjc-core";
      version = "10.3.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/40/a38d78627bd882d86c447db5a195ff307001ae02c1892962c656f2fd6b83/pyobjc_core-10.3.1.tar.gz";
        hash="sha256-sgSoDMwHD5qz+K9COjolpv14fiKFCNAMTDD4rFOLpyA=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      doCheck = false;
    };
    pyobjc-framework-quartz = buildPythonPackage {
      pname = "pyobjc-framework-quartz";
      version = "10.3.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f7/a2/f488d801197b9b4d28d0b8d85947f9e2c8a6e89c5e6d4a828fc7cccfb57a/pyobjc_framework_quartz-10.3.1.tar.gz";
        hash="sha256-ttfjRtc1yafxR8145tp57q5Bagt9OHRkTIOiN4bG+IY=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [pyobjc-core pyobjc-framework-cocoa];
      doCheck = false;
    };
    pyobjc-framework-applicationservices = buildPythonPackage {
      pname = "pyobjc-framework-applicationservices";
      version = "10.3.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/66/a6/3704b63c6e844739e3b7e324d1268fb6f7cb485550267719660779266c60/pyobjc_framework_applicationservices-10.3.1.tar.gz";
        hash="sha256-8ny2SqTRKc5nH9QmOMmF6ypW1UQhSpX+MhSgB+rMR5A=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [pyobjc-core pyobjc-framework-cocoa pyobjc-framework-coretext pyobjc-framework-quartz];
      doCheck = false;
    };
    typing-extensions = buildPythonPackage {
      pname = "typing-extensions";
      version = "4.12.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/26/9f/ad63fc0248c5379346306f8668cda6e2e2e9c95e01216d2b8ffd9ff037d0/typing_extensions-4.12.2-py3-none-any.whl";
        hash="sha256-BOXKA1Hg8/hcaFOVQHLfZZ0NE/rDJNAHIxa2fXeUcA0=";
      };
      doCheck = false;
    };
    tenacity = buildPythonPackage {
      pname = "tenacity";
      version = "8.5.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d2/3f/8ba87d9e287b9d385a02a7114ddcef61b26f86411e121c9003eb509a1773/tenacity-8.5.0-py3-none-any.whl";
        hash="sha256-tZTCpZRYMMJnzmt5oWYigyPtUnGPMDAsE1mDYRI0Zoc=";
      };
      doCheck = false;
    };
    fastjsonschema = buildPythonPackage {
      pname = "fastjsonschema";
      version = "2.20.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6d/ca/086311cdfc017ec964b2436fe0c98c1f4efcb7e4c328956a22456e497655/fastjsonschema-2.20.0-py3-none-any.whl";
        hash="sha256-WHXwsPp6AEOpHpOpuPeTvLu6lpHn/YPcqVwouibSHwo=";
      };
      doCheck = false;
    };
    click = buildPythonPackage {
      pname = "click";
      version = "8.1.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/00/2e/d53fa4befbf2cfa713304affc7ca780ce4fc1fd8710527771b58311a3229/click-8.1.7-py3-none-any.whl";
        hash="sha256-rnT7lsIKAneh1hXx5Nc8hBT1qY24t5mnkx0VgvM5DCg=";
      };
      doCheck = false;
    };
    absl-py = buildPythonPackage {
      pname = "absl-py";
      version = "2.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a2/ad/e0d3c824784ff121c03cc031f944bc7e139a8f1870ffd2845cc2dd76f6c4/absl_py-2.1.0-py3-none-any.whl";
        hash="sha256-UmoE6tq4tO5xnOaPIEFy6tECdUkIlwLZm5BZ8Sn/Ewg=";
      };
      doCheck = false;
    };
    llvmlite = let env = with packages;
    with buildPackages;
    {
      setuptools = setuptools;
    };
    in (import ./nix-custom/llvmlite-0.43.0) {
      buildPythonPackage=buildPythonPackage;
      env=env;
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    numba = buildPythonPackage {
      pname = "numba";
      version = "0.60.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3c/93/2849300a9184775ba274aba6f82f303343669b0592b7bb0849ea713dabb0/numba-0.60.0.tar.gz";
        hash="sha256-XfYVjlWE7s5fyDKUuUn9MLnxEl33cIhiIFIX4GiqvxY=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      dependencies = with packages;
      with buildPackages;
      [llvmlite numpy];
      doCheck = false;
    };
    contourpy = buildPythonPackage {
      pname = "contourpy";
      version = "1.2.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
        hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
      };
      build-system = with packages;
      with buildPackages;
      [pybind11-2_13_1-9263e47d155470de3088643965a87b03f19d995ec7b0c3185075cd59fe53d3b2 meson-python-0_16_0-44c3cdc264c259b3cda4852922014a7735d870b9f9331342e2889168bf5e0347 meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe];
      dependencies = with packages;
      with buildPackages;
      [numpy];
      doCheck = false;
    };
    pygments = buildPythonPackage {
      pname = "pygments";
      version = "2.18.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f7/3f/01c8b82017c199075f8f788d0d906b9ffbbc5a47dc9918a945e13d5a2bda/pygments-2.18.0-py3-none-any.whl";
        hash="sha256-uOasoFI/Ordv7lF5nEiOOHgqwG6vz5XnuoMphcjnsTo=";
      };
      doCheck = false;
    };
    scipy = buildPythonPackage {
      pname = "scipy";
      version = "1.14.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/4e/e5/0230da034a2e1b1feb32621d7cd57c59484091d6dccc9e6b855b0d309fc9/scipy-1.14.0.tar.gz";
        hash="sha256-tZI/SMuEA4D5hUM5F27yF2MRinMAqIIDzNC90m5YUns=";
      };
      build-system = with packages;
      with buildPackages;
      [numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a pybind11-2_12_0-10d5d8ed8114384ccd7aa22d16089f812fd323285b17146b3d78f10d8bee9700 meson-python-0_16_0-44c3cdc264c259b3cda4852922014a7735d870b9f9331342e2889168bf5e0347 cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340 pythran-0_16_1-a6251c3be4b4b3db5101960ad7bd1ba47182ea3588d71b7f0546c628ed6f2af8];
      dependencies = with packages;
      with buildPackages;
      [numpy];
      doCheck = false;
    };
    kiwisolver = buildPythonPackage {
      pname = "kiwisolver";
      version = "1.4.5";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b9/2d/226779e405724344fc678fcc025b812587617ea1a48b9442628b688e85ea/kiwisolver-1.4.5.tar.gz";
        hash="sha256-5X5WOlf7IqFC2jTziswvwaXIZLwpyhUXqIq8lj5g1uw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools cppy-1_2_1-3830e296859438472c12241d2dda4894cbc93a523576760582e534682058e1af wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158 setuptools-scm-8_1_0-2d127c77fef402dac40e1ffe1178b28d44a0b4b4c559dba25247a0a18025387c];
      doCheck = false;
    };
    huggingface-hub = buildPythonPackage {
      pname = "huggingface-hub";
      version = "0.24.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/93/14/6a82b1c2eab5a828f7d3d675811660eb68424e8b039191f418a94e8d9726/huggingface_hub-0.24.2-py3-none-any.whl";
        hash="sha256-q98yRNOidMSx+8XEoe9wADKz9gupPMY+TwNv0IKqKAU=";
      };
      dependencies = with packages;
      with buildPackages;
      [filelock fsspec packaging pyyaml requests tqdm typing-extensions];
      doCheck = false;
    };
    numcodecs = buildPythonPackage {
      pname = "numcodecs";
      version = "0.13.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f8/22/e5cba9013403186906390c0efb0ab0db60d4e580a8966650b2372ab967e1/numcodecs-0.13.0.tar.gz";
        hash="sha256-uk+scDbqWgeMev4dTf/rloUIDULxnJwWsS2thmcDqi4=";
      };
      build-system = with packages;
      with buildPackages;
      [py-cpuinfo-9_0_0-e1845c42d339b04877edca808af7a25bf1d94f98a662f25c5acd6a5c34078fbd setuptools numpy cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340 setuptools-scm-8_1_0-2d127c77fef402dac40e1ffe1178b28d44a0b4b4c559dba25247a0a18025387c];
      dependencies = with packages;
      with buildPackages;
      [numpy];
      doCheck = false;
    };
    toolz = buildPythonPackage {
      pname = "toolz";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/8a/d82202c9f89eab30f9fc05380daae87d617e2ad11571ab23d7c13a29bb54/toolz-0.12.1-py3-none-any.whl";
        hash="sha256-0icxNkwH1y7qCgrUW6+ywpN6tv04o1B79V6uh0SqfYU=";
      };
      doCheck = false;
    };
    tzdata = buildPythonPackage {
      pname = "tzdata";
      version = "2024.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/65/58/f9c9e6be752e9fcb8b6a0ee9fb87e6e7a1f6bcab2cdc73f02bb7ba91ada0/tzdata-2024.1-py2.py3-none-any.whl";
        hash="sha256-kGi8GWE2Rj9SReUe/ag4r6FarsqZA/SQUN+iZ5200lI=";
      };
      doCheck = false;
    };
    nbformat = buildPythonPackage {
      pname = "nbformat";
      version = "5.10.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a9/82/0340caa499416c78e5d8f5f05947ae4bc3cba53c9f038ab6e9ed964e22f1/nbformat-5.10.4-py3-none-any.whl";
        hash="sha256-O0jWyPvKSymb85gup9sa8hWA5P7Caa0Ie56BWIiRIAs=";
      };
      dependencies = with packages;
      with buildPackages;
      [fastjsonschema jsonschema jupyter-core traitlets];
      doCheck = false;
    };
    plotly = buildPythonPackage {
      pname = "plotly";
      version = "5.23.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b8/f0/bcf716a8e070370d6598c92fcd328bd9ef8a9bda2c5562da5a835c66700b/plotly-5.23.0-py3-none-any.whl";
        hash="sha256-dsvnj3Xt3BDFb1pO4+fMqt58CldGVUbwIJjAyu1sLRo=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging tenacity];
      doCheck = false;
    };
    safetensors = buildPythonPackage {
      pname = "safetensors";
      version = "0.4.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9c/21/acd1b6dd9dad9708fd388fdbe6618e461108cdbd56ff4eab6094c6e61035/safetensors-0.4.3.tar.gz";
        hash="sha256-L4X8UMTgeiHpXCTgdGD+b34oWdDOiAkoODUreYznEcI=";
      };
      build-system = with packages;
      with buildPackages;
      [maturin-1_7_0-824706860354622439a169eb472a146420fe82faf0e53565cd55e420a643ff24];
      doCheck = false;
    };
    tokenizers = buildPythonPackage {
      pname = "tokenizers";
      version = "0.19.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/48/04/2071c150f374aab6d5e92aaec38d0f3c368d227dd9e0469a1f0966ac68d1/tokenizers-0.19.1.tar.gz";
        hash="sha256-7lnmaA7Q/b5rckzzi9cEAKDB3WI7B6xykIcnDK6siOM=";
      };
      build-system = with packages;
      with buildPackages;
      [maturin-1_7_0-824706860354622439a169eb472a146420fe82faf0e53565cd55e420a643ff24];
      dependencies = with packages;
      with buildPackages;
      [huggingface-hub];
      doCheck = false;
    };
    filelock = buildPythonPackage {
      pname = "filelock";
      version = "3.15.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ae/f0/48285f0262fe47103a4a45972ed2f9b93e4c80b8fd609fa98da78b2a5706/filelock-3.15.4-py3-none-any.whl";
        hash="sha256-bKH/+uliJdq0xurxxPTyjNJWjT7CpE4VoIUgUE3kaOc=";
      };
      doCheck = false;
    };
    transformers = buildPythonPackage {
      pname = "transformers";
      version = "4.43.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/13/63/cccd0297770d7096c19c99d4c542f3068a30e73cdfd971a920bfa686cb3a/transformers-4.43.2-py3-none-any.whl";
        hash="sha256-KDyLR884ZAxcDK6mC+DfqUhmn6SOlzmwNxfL9eiyDxE=";
      };
      dependencies = with packages;
      with buildPackages;
      [filelock huggingface-hub numpy packaging pyyaml regex requests safetensors tokenizers tqdm];
      doCheck = false;
    };
    cycler = buildPythonPackage {
      pname = "cycler";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e7/05/c19819d5e3d95294a6f5947fb9b9629efb316b96de511b418c53d245aae6/cycler-0.12.1-py3-none-any.whl";
        hash="sha256-hc73z/Ii2GRBYVKYCEZZcuUTQFmUWbisPMusWoVODTA=";
      };
      doCheck = false;
    };
    matplotlib = buildPythonPackage {
      pname = "matplotlib";
      version = "3.9.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/22/06/9e8ba6ec8b716a215404a5d1938b61f5a28001be493cf35344dda9a4072a/matplotlib-3.9.1.tar.gz";
        hash="sha256-3gaxm425XdM9DcF8kmx8nr7Z9XIHS2+sT2UGimgU0BA=";
      };
      build-system = with packages;
      with buildPackages;
      [numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a pybind11-2_13_1-9263e47d155470de3088643965a87b03f19d995ec7b0c3185075cd59fe53d3b2 meson-python-0_16_0-44c3cdc264c259b3cda4852922014a7735d870b9f9331342e2889168bf5e0347 setuptools-scm-8_1_0-0d702ec1b648b58b185f9cbd34c50e0a44ebfbea10cfb89eb5322a980c098fab];
      dependencies = with packages;
      with buildPackages;
      [contourpy cycler fonttools kiwisolver numpy packaging pillow pyparsing python-dateutil];
      doCheck = false;
    };
    optax = buildPythonPackage {
      pname = "optax";
      version = "0.2.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
        hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
      };
      dependencies = with packages;
      with buildPackages;
      [absl-py chex etils jax jaxlib numpy];
      doCheck = false;
    };
    nest-asyncio = buildPythonPackage {
      pname = "nest-asyncio";
      version = "1.6.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/c4/c2971a3ba4c6103a3d10c4b0f24f461ddc027f0f09763220cf35ca1401b3/nest_asyncio-1.6.0-py3-none-any.whl";
        hash="sha256-h69u/WteiXyBBQR372XGLisvNdUXA8rgGv8pBbGFLhw=";
      };
      doCheck = false;
    };
    tensorstore = buildPythonPackage {
      pname = "tensorstore";
      version = "0.1.63";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/47/8b/e38852acdc76853f35ca455b41eb059ee4a417120bd9dc7785e160296b13/tensorstore-0.1.63.tar.gz";
        hash="sha256-ar3ghNaTK05zPfEJwegZqff17Y5oNyp4ghwPPnaiBGk=";
      };
      build-system = with packages;
      with buildPackages;
      [numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a setuptools-scm-8_1_0-0d702ec1b648b58b185f9cbd34c50e0a44ebfbea10cfb89eb5322a980c098fab setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [ml-dtypes numpy];
      doCheck = false;
    };
    orbax-checkpoint = buildPythonPackage {
      pname = "orbax-checkpoint";
      version = "0.5.22";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/02/b6/d87435560140a7bd6bb51f7c63558375abccf6cadb119630236a01552154/orbax_checkpoint-0.5.22-py3-none-any.whl";
        hash="sha256-0TNOI+tS/jJPFEMawq5OW00b02s4vlj0OlSbraZjRnY=";
      };
      dependencies = with packages;
      with buildPackages;
      [absl-py etils jax jaxlib msgpack nest-asyncio numpy protobuf pyyaml tensorstore typing-extensions];
      doCheck = false;
    };
    msgpack = buildPythonPackage {
      pname = "msgpack";
      version = "1.0.8";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/08/4c/17adf86a8fbb02c144c7569dc4919483c01a2ac270307e2d59e1ce394087/msgpack-1.0.8.tar.gz";
        hash="sha256-lcArDifnBuSNDlQm0XEMp44PBijW6J1bWluRpfEidPM=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340];
      doCheck = false;
    };
    flax = buildPythonPackage {
      pname = "flax";
      version = "0.8.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1c/a9/6978d2547b1d8ca0ce75b534c0ba5c60e8e7b918c5c1800225aa0169cb7f/flax-0.8.5-py3-none-any.whl";
        hash="sha256-yW5G0cSKMA0BDr9cSEbxY73XrMbv/1/yv7HLWwiqZdg=";
      };
      dependencies = with packages;
      with buildPackages;
      [jax msgpack numpy optax orbax-checkpoint pyyaml rich tensorstore typing-extensions];
      doCheck = false;
    };
    pluggy = buildPythonPackage {
      pname = "pluggy";
      version = "1.5.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/88/5f/e351af9a41f866ac3f1fac4ca0613908d9a41741cfcf2228f4ad853b697d/pluggy-1.5.0-py3-none-any.whl";
        hash="sha256-ROGtksjKAC3mN34WXz4PG+YyZqtNVUdAUyM1uddepmk=";
      };
      doCheck = false;
    };
    packaging = buildPythonPackage {
      pname = "packaging";
      version = "24.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/08/aa/cc0199a5f0ad350994d660967a8efb233fe0416e4639146c089643407ce6/packaging-24.1-py3-none-any.whl";
        hash="sha256-W48iF9vb0vfzhMQcYoVE5tUvLQ9TxtDD6mGqXR1/8SQ=";
      };
      doCheck = false;
    };
    ffmpegio-core = buildPythonPackage {
      pname = "ffmpegio-core";
      version = "0.10.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/69/44/d62a059a2c93161d022ea4045960cd7d374a48d12f9f6ac35c396cab45f2/ffmpegio_core-0.10.0-py3-none-any.whl";
        hash="sha256-HKtA7dl3sSBWlceriebFyX8z3XOBd9mYaibCbcpYFFs=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging pluggy];
      doCheck = false;
    };
    ffmpegio = buildPythonPackage {
      pname = "ffmpegio";
      version = "0.10.0.post0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/52/fd/b85fec7fc96cb6e0aa8b307e1ec0de9986826b3574d67b60762868e7feea/ffmpegio-0.10.0.post0-py3-none-any.whl";
        hash="sha256-br1OgDQ+cnqjGQQRHrXFc7QXWwQ57kEsZSs5LVnM+gU=";
      };
      dependencies = with packages;
      with buildPackages;
      [ffmpegio-core numpy];
      doCheck = false;
    };
    python-dateutil = buildPythonPackage {
      pname = "python-dateutil";
      version = "2.9.0.post0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl";
        hash="sha256-qLK8e/+uKCKByBQKl9OqnBTaCxNt/oP4UO6ppfdHBCc=";
      };
      dependencies = with packages;
      with buildPackages;
      [six];
      doCheck = false;
    };
    mdurl = buildPythonPackage {
      pname = "mdurl";
      version = "0.1.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl";
        hash="sha256-hACKQeUWFaSfyZZhkf+RUJ48QLk5F25kP9UKXCGWuPg=";
      };
      doCheck = false;
    };
    pandas = buildPythonPackage {
      pname = "pandas";
      version = "2.2.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/88/d9/ecf715f34c73ccb1d8ceb82fc01cd1028a65a5f6dbc57bfa6ea155119058/pandas-2.2.2.tar.gz";
        hash="sha256-nnkBmrpDy0/ank2YP46IygNzrbtpeunGxDCTIY3ii1Q=";
      };
      build-system = with packages;
      with buildPackages;
      [cython-3_0_5-eecf9b399ab12ed1f0673b9f8f31d7c4a637972cd6c1a680d83bd9169e8118dd meson-python-0_13_1-80c1db885775e48a127ecfcbead2035ed6a1831e80e81ae959ec48d2afd835a5 meson-1_2_1-6ea0d3abd41fff844f366136a50b8f68f75aaf36b46b38d13f0a9dc5f25e44c6 numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a versioneer-0_29-4a2512b3cdcd7699f877273c71b8d21a04a497b3b5d9ff259b82f25cf1990daf wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [numpy python-dateutil pytz tzdata];
      doCheck = false;
    };
    markdown-it-py = buildPythonPackage {
      pname = "markdown-it-py";
      version = "3.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl";
        hash="sha256-NVIWhFxgvZYjLNjYxA6Pl2XMhvRogOQ6j9ItwaGoyrE=";
      };
      dependencies = with packages;
      with buildPackages;
      [mdurl];
      doCheck = false;
    };
    termcolor = buildPythonPackage {
      pname = "termcolor";
      version = "2.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d9/5f/8c716e47b3a50cbd7c146f45881e11d9414def768b7cd9c5e6650ec2a80a/termcolor-2.4.0-py3-none-any.whl";
        hash="sha256-kpfA35yZRFwkEugy6IKniEA4olYXxgzqKtaUiNQEDWM=";
      };
      doCheck = false;
    };
    pynput = buildPythonPackage {
      pname = "pynput";
      version = "1.7.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ef/1d/fdef3fdc9dc8dedc65898c8ad0e8922a914bb89c5308887e45f9aafaec36/pynput-1.7.7-py2.py3-none-any.whl";
        hash="sha256-r8Q/ZRaEyYgY3gSKvHat+fLT15cIPLB8H4K+dkotRMs=";
      };
      dependencies = with packages;
      with buildPackages;
      [pyobjc-framework-applicationservices pyobjc-framework-quartz six];
      doCheck = false;
    };
    pillow = buildPythonPackage {
      pname = "pillow";
      version = "10.4.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/cd/74/ad3d526f3bf7b6d3f408b73fde271ec69dfac8b81341a318ce825f2b3812/pillow-10.4.0.tar.gz";
        hash="sha256-Fmwc1NJDCbMNYfefSpEUt7IxPXRQkSJ3hV/139fNSgY=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    opencv-python = buildPythonPackage {
      pname = "opencv-python";
      version = "4.10.0.84";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/4a/e7/b70a2d9ab205110d715906fc8ec83fbb00404aeb3a37a0654fdb68eb0c8c/opencv-python-4.10.0.84.tar.gz";
        hash="sha256-ctI05Fgullj/6o6crltj1IitBplO8S2B3DA7F0cvNSY=";
      };
      build-system = with packages;
      with buildPackages;
      [numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a pip-24_1_2-c506f4afc1c13f35b33177ef3b20d80b704ad9bf8866df2a2ef4fc7f25686747 setuptools-59_2_0-6d505cd5fc2a97a44b4703fdc7fcb90b83cbb1f9bf979ca01a6ebad8d95064c0 scikit-build-0_18_0-bbae32db54641d39652ea79c4cd525d1e885508fa018473f97b9a0720b1a6530 cmake-3_30_1-b57abfab4d4a448ae6ef989de008c7423c88791f6b8101f1927fc6819585a4fc];
      dependencies = with packages;
      with buildPackages;
      [numpy numpy numpy numpy numpy];
      doCheck = false;
    };
    robosuite = buildPythonPackage {
      pname = "robosuite";
      version = "1.4.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/60/2f/bbcdf3130fc2c53c098d8699048724ac14292795dfb671168d1d5013fa03/robosuite-1.4.1-py3-none-any.whl";
        hash="sha256-Ylh81YiNnT2BPz/8xwF/b3fvphiVJSegubsTn/0KN0E=";
      };
      dependencies = with packages;
      with buildPackages;
      [mujoco numba numpy opencv-python pillow pynput scipy termcolor];
      doCheck = false;
    };
    shapely = buildPythonPackage {
      pname = "shapely";
      version = "2.0.5";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ad/99/c47247f4d688bbb5346df5ff1de5d9792b6d95cbbb2fd7b71f45901c1878/shapely-2.0.5.tar.gz";
        hash="sha256-v/I2a8eGv6bLNT1rR9BEPFcMMndmEuUn7ke232P8/jI=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools numpy cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340];
      dependencies = with packages;
      with buildPackages;
      [numpy];
      doCheck = false;
    };
    pyopengl = buildPythonPackage {
      pname = "pyopengl";
      version = "3.1.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/99/48/00e31747821d3fc56faddd00a4725454d1e694a8b67d715cf20f531506a5/PyOpenGL-3.1.7-py3-none-any.whl";
        hash="sha256-pqsZzykN9hAar3RwhDqcRiB3iYVXRjmdCvklIaCpK3o=";
      };
      doCheck = false;
    };
    glfw = buildPythonPackage {
      pname = "glfw";
      version = "2.7.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1f/fe/fd827e0e0babed43c08949644d1c2cafe5bc0f0ddcd369248eb27841c81c/glfw-2.7.0.tar.gz";
        hash="sha256-DiCa04+oxb5nylkNexdTPZWtHrV9Cj8HuYEx22m3kAA=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    zipp = buildPythonPackage {
      pname = "zipp";
      version = "3.19.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/20/38/f5c473fe9b90c8debdd29ea68d5add0289f1936d6f923b6b9cc0b931194c/zipp-3.19.2-py3-none-any.whl";
        hash="sha256-8JF1X2ZwVfLQKzLFN3Gnpsi0fh/bxLcqi5Bys+74AVw=";
      };
      doCheck = false;
    };
    importlib-resources = buildPythonPackage {
      pname = "importlib-resources";
      version = "6.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/75/06/4df55e1b7b112d183f65db9503bff189e97179b256e1ea450a3c365241e0/importlib_resources-6.4.0-py3-none-any.whl";
        hash="sha256-UNEPBD35MZAtQZTqB+xXlg9mqARJ/4Z7/ngrTEhrp4w=";
      };
      doCheck = false;
    };
    fsspec = buildPythonPackage {
      pname = "fsspec";
      version = "2024.6.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5e/44/73bea497ac69bafde2ee4269292fa3b41f1198f4bb7bbaaabde30ad29d4a/fsspec-2024.6.1-py3-none-any.whl";
        hash="sha256-PLRD+LzS77MSlaW5/bAq7oHYRSyA0o+XptCVnmzuEB4=";
      };
      doCheck = false;
    };
    trimesh = buildPythonPackage {
      pname = "trimesh";
      version = "4.4.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/71/63/bbe0395bb7dd414188fb0df0e97132098e541a6ad7b64c7faf3b1886d222/trimesh-4.4.3-py3-none-any.whl";
        hash="sha256-d9w5Y+w+T1YLN6ZvjdkTLu6fMz0Eg30rBojqEvBcCZo=";
      };
      dependencies = with packages;
      with buildPackages;
      [numpy];
      doCheck = false;
    };
    mujoco = buildPythonPackage {
      pname = "mujoco";
      version = "3.2.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fc/70/e94bb93c0dad847b7fe13c11ff510253583c610dbc04ec4bf191267292dc/mujoco-3.2.0.tar.gz";
        hash="sha256-R388jEIbzd60en1SRC8mKSSlvdW/xWl1xDnUET7QvKc=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      dependencies = with packages;
      with buildPackages;
      [absl-py etils glfw numpy pyopengl];
      doCheck = false;
    };
    etils = buildPythonPackage {
      pname = "etils";
      version = "1.9.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/f4/305f3ea85aecd23422c606c179fb6d00bd7d255b10d55b4c797a3a680144/etils-1.9.2-py3-none-any.whl";
        hash="sha256-7Ned4fv+qbDWkkdWz6kisF7TNgxFzyFwdn2kvuAAHSA=";
      };
      dependencies = with packages;
      with buildPackages;
      [fsspec importlib-resources typing-extensions zipp];
      doCheck = false;
    };
    mujoco-mjx = buildPythonPackage {
      pname = "mujoco-mjx";
      version = "3.2.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d9/44/e0c44caacf42abccec5aadfef04990e50c2d33947a1626f5047ace32dab7/mujoco_mjx-3.2.0-py3-none-any.whl";
        hash="sha256-rGDGtMrMqjI2XknZkQ10LBKBtYLw6pLA4AwHz6nXNYo=";
      };
      dependencies = with packages;
      with buildPackages;
      [absl-py etils jax jaxlib mujoco scipy trimesh];
      doCheck = false;
    };
    fasteners = buildPythonPackage {
      pname = "fasteners";
      version = "0.19";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/61/bf/fd60001b3abc5222d8eaa4a204cd8c0ae78e75adc688f33ce4bf25b7fafa/fasteners-0.19-py3-none-any.whl";
        hash="sha256-dYgZy12Uze306DaYi3TeOWzqy44nlNIfgtEx/Z7ncjc=";
      };
      doCheck = false;
    };
    asciitree = buildPythonPackage {
      pname = "asciitree";
      version = "0.3.3";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/2d/6a/885bc91484e1aa8f618f6f0228d76d0e67000b0fdd6090673b777e311913/asciitree-0.3.3.tar.gz";
        hash="sha256-SqS5tkn4Xj/LNDNj2XVkqh+2LiSWd/LhipZ2UUXMD24=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    zarr = buildPythonPackage {
      pname = "zarr";
      version = "2.18.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5d/bd/8d881d8ca6d80fcb8da2b2f94f8855384daf649499ddfba78ffd1ee2caa3/zarr-2.18.2-py3-none-any.whl";
        hash="sha256-pjh1SQL5fvqZtAYIP9yAeg4szxKpSRFzidKkupsF3zg=";
      };
      dependencies = with packages;
      with buildPackages;
      [asciitree fasteners numcodecs numpy];
      doCheck = false;
    };
    einops = buildPythonPackage {
      pname = "einops";
      version = "0.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/44/5a/f0b9ad6c0a9017e62d4735daaeb11ba3b6c009d69a26141b258cd37b5588/einops-0.8.0-py3-none-any.whl";
        hash="sha256-lXL7YwRiZKhiaTsKhwiK873IwGj94D3mNFPLveJFRl8=";
      };
      doCheck = false;
    };
    chex = buildPythonPackage {
      pname = "chex";
      version = "0.1.86";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e6/ed/625d545d08c6e258d2d63a93a0bf8ed8a296c09d254208e73f9d4fb0b746/chex-0.1.86-py3-none-any.whl";
        hash="sha256-JRwgghCSMjo9nCjhz4DkpYGAl4vsNo9TGUm9mEfu5Wg=";
      };
      dependencies = with packages;
      with buildPackages;
      [absl-py jax jaxlib numpy toolz typing-extensions];
      doCheck = false;
    };
    rich = buildPythonPackage {
      pname = "rich";
      version = "13.7.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/87/67/a37f6214d0e9fe57f6ae54b2956d550ca8365857f42a1ce0392bb21d9410/rich-13.7.1-py3-none-any.whl";
        hash="sha256-TtuuMU9Z60gvVOnjC/ANMzUKqpT0v81OnjEQ5k0NciI=";
      };
      dependencies = with packages;
      with buildPackages;
      [markdown-it-py pygments];
      doCheck = false;
    };
    setuptools = buildPythonPackage {
      pname = "setuptools";
      version = "71.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/51/a0/ee460cc54e68afcf33190d198299c9579a5eafeadef0016ae8563237ccb6/setuptools-71.1.0-py3-none-any.whl";
        hash="sha256-M4dP3FmzGIMEsufIDZApCX6jFicYCJb7VJxXjOuKCFU=";
      };
      doCheck = false;
    };
    setproctitle = buildPythonPackage {
      pname = "setproctitle";
      version = "1.3.3";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ff/e1/b16b16a1aa12174349d15b73fd4b87e641a8ae3fb1163e80938dbbf6ae98/setproctitle-1.3.3.tar.gz";
        hash="sha256-yRPhUefqAVZ4N/8DeiPKh0AZKIAZi3+7kLFtGBYHyq4=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    platformdirs = buildPythonPackage {
      pname = "platformdirs";
      version = "4.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/68/13/2aa1f0e1364feb2c9ef45302f387ac0bd81484e9c9a4c5688a322fbdfd08/platformdirs-4.2.2-py3-none-any.whl";
        hash="sha256-LXoWV+NqgOqRHbgyqKbs5e5T2N4h7dXMWHmvZTCxv+4=";
      };
      doCheck = false;
    };
    docker-pycreds = buildPythonPackage {
      pname = "docker-pycreds";
      version = "0.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f5/e8/f6bd1eee09314e7e6dee49cbe2c5e22314ccdb38db16c9fc72d2fa80d054/docker_pycreds-0.4.0-py2.py3-none-any.whl";
        hash="sha256-cmYRJGhieGgAUQbsGc0NcicC0rfVkSoo4ZuCbD03r0k=";
      };
      dependencies = with packages;
      with buildPackages;
      [six];
      doCheck = false;
    };
    wandb = buildPythonPackage {
      pname = "wandb";
      version = "0.17.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9f/ae/3bb23208fa238944d8f1bb13bf03feaa3dd0af29316505d2a79d7323ed1c/wandb-0.17.5-py3-none-any.whl";
        hash="sha256-HA9gRGtRVhtnKAoGA4j/rSpgePz99QJLmZglLSN7Rjk=";
      };
      dependencies = with packages;
      with buildPackages;
      [click docker-pycreds gitpython platformdirs protobuf psutil pyyaml requests sentry-sdk setproctitle setuptools];
      doCheck = false;
    };
    six = buildPythonPackage {
      pname = "six";
      version = "1.16.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d9/5a/e7c31adbe875f2abbb91bd84cf2dc52d792b5a01506781dbcf25c91daf11/six-1.16.0-py2.py3-none-any.whl";
        hash="sha256-irsvHYaJCi37mJ+ad8/P0+R8KjVLAREXcTJviqJuAlQ=";
      };
      doCheck = false;
    };
    pyyaml = buildPythonPackage {
      pname = "pyyaml";
      version = "6.0.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/cd/e5/af35f7ea75cf72f2cd079c95ee16797de7cd71f29ea7c68ae5ce7be1eda0/PyYAML-6.0.1.tar.gz";
        hash="sha256-v99GCxc2x3Xyup9qkryjC8IJUGe4qdd4dtH61sw7SkM=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158 cython-0_29_37-70ce88e0e16e90456eca2cdd68685260e331aa8da341ec4c95df613a40c8d4a7];
      doCheck = false;
    };
    contextlib2 = buildPythonPackage {
      pname = "contextlib2";
      version = "21.6.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/76/56/6d6872f79d14c0cb02f1646cbb4592eef935857c0951a105874b7b62a0c3/contextlib2-21.6.0-py2.py3-none-any.whl";
        hash="sha256-P722RGav0jq69sl3Ynt1thOaWj6M44QFxbQTrtegRx8=";
      };
      doCheck = false;
    };
    ml-collections = buildPythonPackage {
      pname = "ml-collections";
      version = "0.1.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/aa/ea/853aa32dfa1006d3eb43384712f35b8f2d6f0a757b8c779d40c29e3e8515/ml_collections-0.1.1.tar.gz";
        hash="sha256-P+/McuxDOqHl0yMHo+R0u7Z/QFvoFOpSohZr/J2+aMw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      dependencies = with packages;
      with buildPackages;
      [absl-py contextlib2 pyyaml six];
      doCheck = false;
    };
    opt-einsum = buildPythonPackage {
      pname = "opt-einsum";
      version = "3.3.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/bc/19/404708a7e54ad2798907210462fd950c3442ea51acc8790f3da48d2bee8b/opt_einsum-3.3.0-py3-none-any.whl";
        hash="sha256-JFXlnjlH08J1R3339SBbMGNeJm/m3DAOPZ+WRr/OoUc=";
      };
      dependencies = with packages;
      with buildPackages;
      [numpy];
      doCheck = false;
    };
    ml-dtypes = let env = with packages;
    with buildPackages;
    {
      numpy = numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a;
      setuptools = setuptools-70_1_1-6ef60e6ca78ad9a7bf8979a2fef9ee5079bc76156c8bd044bed129d5743a312a;
    };
    in (import ./nix-custom/ml_dtypes-0.4.0) {
      buildPythonPackage=buildPythonPackage;
      env=env;
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    jaxlib = let env = with packages;
    with buildPackages;
    {
      ml-dtypes = ml-dtypes;
      numpy = numpy;
      setuptools = setuptools;
      scipy = scipy;
      wheel = wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158;
    };
    in (import ./nix-custom/jaxlib-0.4.30) {
      buildPythonPackage=buildPythonPackage;
      env=env;
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    h5py = buildPythonPackage {
      pname = "h5py";
      version = "3.11.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/52/8f/e557819155a282da36fb21f8de4730cfd10a964b52b3ae8d20157ac1c668/h5py-3.11.0.tar.gz";
        hash="sha256-e36PeAcqLt7IfJg28l80ID/UkqRHVwmhi0F6M8+yH6k=";
      };
      build-system = with packages;
      with buildPackages;
      [numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a pkgconfig-1_5_5-0d2df3706e3013a39c01acb40571b9367fe03a4915085e8374ce14040b323035 setuptools cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340];
      doCheck = false;
    };
    sentencepiece = buildPythonPackage {
      pname = "sentencepiece";
      version = "0.2.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c9/d2/b9c7ca067c26d8ff085d252c89b5f69609ca93fb85a00ede95f4857865d4/sentencepiece-0.2.0.tar.gz";
        hash="sha256-pSwZFx2q8uaX3Gy+Z2hOD6NBsSSJZvauu1Qd5lTRWEM=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    trajax = buildPythonPackage {
      pname = "trajax";
      version = "0.0.1";
      src = fetchurl {
        url="https://github.com/google/trajax/archive/c94a637c5a397b3d4100153f25b4b165507b5b20.tar.gz";
        hash="sha256-xN/LSQI/zvf367Ba9MFRIzpP/AmFbAOT1M1ShuW75pI=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      dependencies = with packages;
      with buildPackages;
      [absl-py jax jaxlib ml-collections scipy];
      doCheck = false;
    };
    beautifulsoup4 = buildPythonPackage {
      pname = "beautifulsoup4";
      version = "4.12.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b1/fe/e8c672695b37eecc5cbf43e1d0638d88d66ba3a44c4d321c796f4e59167f/beautifulsoup4-4.12.3-py3-none-any.whl";
        hash="sha256-uAh4yfQBETE+VdqLogvboG2Po5afxoMEFndBu/nggu0=";
      };
      dependencies = with packages;
      with buildPackages;
      [soupsieve];
      doCheck = false;
    };
    numpy = let env = with packages;
    with buildPackages;
    {
      meson-python = meson-python-0_15_0-29ccb277ce25aa5d3be8b684d44e64e4596529292435ff9d4f2df1081c724147;
      ninja = ninja-1_11_1_1-0c6eb68995cfb827551216b6414b072b470e77b21cf76a3af3f0dd49428b3304;
      pyproject-metadata = pyproject-metadata-0_8_0-1ffbfb773b6e13d61ffaba777a56c22b450209f77f53cd2889e74a6b519702d0;
      packaging = packaging;
      tomli = tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734;
      cython = cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340;
      meson = meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe;
    };
    in (import ./nix-custom/numpy-1.26.4) {
      buildPythonPackage=buildPythonPackage;
      env=env;
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    jax = buildPythonPackage {
      pname = "jax";
      version = "0.4.30";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fd/f2/9dbb75de3058acfd1600cf0839bcce7ea391148c9d2b4fa5f5666e66f09e/jax-0.4.30-py3-none-any.whl";
        hash="sha256-KJswrgO1L39Lr27wgqn04+KcEIDiLRNRLF7PAtXxpVs=";
      };
      dependencies = with packages;
      with buildPackages;
      [jaxlib ml-dtypes numpy opt-einsum scipy];
      doCheck = false;
    };
    language-model = buildPythonPackage {
      pname = "language-model";
      version = "0.1.0";
      format="pyproject";
      src = ./projects/language-model;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [stanza stanza-models];
      doCheck = false;
    };
    image-classifier = buildPythonPackage {
      pname = "image-classifier";
      version = "0.1.0";
      format="pyproject";
      src = ./projects/image-classifier;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [stanza stanza-models];
      doCheck = false;
    };
    cond-diffusion = buildPythonPackage {
      pname = "cond-diffusion";
      version = "0.1.0";
      format="pyproject";
      src = ./projects/cond-diffusion;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [stanza];
      doCheck = false;
    };
    stanza-models = buildPythonPackage {
      pname = "stanza-models";
      version = "0.1.0";
      format="pyproject";
      src = ./projects/models;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [stanza transformers];
      doCheck = false;
    };
    stanza = buildPythonPackage {
      pname = "stanza";
      version = "0.1.0";
      format="pyproject";
      src = ./packages/stanza;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [jax rich flax optax pandas chex numpy ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
      doCheck = false;
    };
    stanza-meta = buildPythonPackage {
      pname = "stanza-meta";
      version = "0.1.0";
      format="pyproject";
      src = ./.;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [stanza stanza-models cond-diffusion image-classifier language-model wandb];
      doCheck = false;
    };
  };
  buildPackages = rec {
    tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 = buildPythonPackage {
      pname = "tomli";
      version = "2.0.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/97/75/10a9ebee3fd790d20926a90a2547f0bf78f371b2f13aa822c759680ca7b9/tomli-2.0.1-py3-none-any.whl";
        hash="sha256-k53j56YWGvDIh++Rt9QaU+fFocqXYyX0KctG6pvDDsw=";
      };
      doCheck = false;
    };
    maturin-1_7_0-824706860354622439a169eb472a146420fe82faf0e53565cd55e420a643ff24 = buildPythonPackage {
      pname = "maturin";
      version = "1.7.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/80/da/a4bbd6e97f3645f4ebd725321aa235e22e31037dfd92caf4539f721c0a5a/maturin-1.7.0.tar.gz";
        hash="sha256-G6UnfdeDLcYYHWmgBRgrl7NSCUWCUFhIT/2SlvLvtZw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 setuptools-rust-1_9_0-9e7c0be00f52872ac7efbcc9b84f8e5e595bb6d0a11b448c10bbd76537de22ef wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      dependencies = with packages;
      with buildPackages;
      [tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734];
      doCheck = false;
    };
    wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158 = buildPythonPackage {
      pname = "wheel";
      version = "0.43.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7d/cd/d7460c9a869b16c3dd4e1e403cce337df165368c71d6af229a74699622ce/wheel-0.43.0-py3-none-any.whl";
        hash="sha256-VcVwQF8UJjDGufcv4J2bZ88Ud/z1Q65bjcsfW3N32oE=";
      };
      doCheck = false;
    };
    semantic-version-2_10_0-3cdc10bed1cae71103698fb7ffe5f5094350e5b6554d57c190fdc16f361a0a1a = buildPythonPackage {
      pname = "semantic-version";
      version = "2.10.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6a/23/8146aad7d88f4fcb3a6218f41a60f6c2d4e3a72de72da1825dc7c8f7877c/semantic_version-2.10.0-py2.py3-none-any.whl";
        hash="sha256-3nijuOD+2nTKvFSqstpwIRPjOsnZ650jibzx9Yt9kXc=";
      };
      doCheck = false;
    };
    setuptools-rust-1_9_0-9e7c0be00f52872ac7efbcc9b84f8e5e595bb6d0a11b448c10bbd76537de22ef = buildPythonPackage {
      pname = "setuptools-rust";
      version = "1.9.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f7/7f/8b1c33598b03ad612b8ced223f9ca54076b789fabf5a66ce37cc096d9cf7/setuptools_rust-1.9.0-py3-none-any.whl";
        hash="sha256-QJyvSdz3rZvVELS/QBH7rVBOdF+umPV/4cBvOpdxljg=";
      };
      dependencies = with packages;
      with buildPackages;
      [setuptools tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 semantic-version-2_10_0-3cdc10bed1cae71103698fb7ffe5f5094350e5b6554d57c190fdc16f361a0a1a];
      doCheck = false;
    };
    meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe = buildPythonPackage {
      pname = "meson";
      version = "1.5.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c4/cd/47e45d3abada2e1edb9e2ca9334be186d2e7f97a01b09b5b82799c4d7bd3/meson-1.5.0-py3-none-any.whl";
        hash="sha256-UrNPSQO4gt9SrQ1TMUbUuZLAGOp3OZ+CVXlzdnKueyA=";
      };
      doCheck = false;
    };
    pyproject-metadata-0_8_0-1ffbfb773b6e13d61ffaba777a56c22b450209f77f53cd2889e74a6b519702d0 = buildPythonPackage {
      pname = "pyproject-metadata";
      version = "0.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/aa/5f/bb5970d3d04173b46c9037109f7f05fc8904ff5be073ee49bb6ff00301bc/pyproject_metadata-0.8.0-py3-none-any.whl";
        hash="sha256-rYWNRI4dOh+0CKxbrJ6ndD56i7tHLyaTqqM00ttC9SY=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging];
      doCheck = false;
    };
    meson-python-0_16_0-44c3cdc264c259b3cda4852922014a7735d870b9f9331342e2889168bf5e0347 = buildPythonPackage {
      pname = "meson-python";
      version = "0.16.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/91/c0/104cb6244c83fe6bc3886f144cc433db0c0c78efac5dc00e409a5a08c87d/meson_python-0.16.0-py3-none-any.whl";
        hash="sha256-hC3J9dwp5V/Haf8bb+MoQS/myHAiD8MhBgodLTleaeg=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging pyproject-metadata-0_8_0-1ffbfb773b6e13d61ffaba777a56c22b450209f77f53cd2889e74a6b519702d0 tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe];
      doCheck = false;
    };
    pybind11-2_13_1-9263e47d155470de3088643965a87b03f19d995ec7b0c3185075cd59fe53d3b2 = buildPythonPackage {
      pname = "pybind11";
      version = "2.13.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/84/fb/1a249de406daf2b4ebd2d714b739e8519034617daec085e3833c1a3ed57c/pybind11-2.13.1-py3-none-any.whl";
        hash="sha256-l4gVNqvgzUJgqczFv20c8xEzGPCK8f64LUuflek/CqQ=";
      };
      doCheck = false;
    };
    pythran-0_16_1-a6251c3be4b4b3db5101960ad7bd1ba47182ea3588d71b7f0546c628ed6f2af8 = buildPythonPackage {
      pname = "pythran";
      version = "0.16.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/73/32/f892675c5009cd4c1895ded3d6153476bf00adb5ad1634d03635620881f5/pythran-0.16.1.tar.gz";
        hash="sha256-hhdIwPnH1CKzJySxFLOBfYGO1Oq4bAl4GqCj986rt/k=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340 = buildPythonPackage {
      pname = "cython";
      version = "3.0.10";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b6/83/b0a63fc7b315edd46821a1a381d18765c1353d201246da44558175cddd56/Cython-3.0.10-py2.py3-none-any.whl";
        hash="sha256-/LtnnAtDUU1ZFXf9DSACHFXCQMqcyvvbgtP7leXt/uI=";
      };
      doCheck = false;
    };
    pybind11-2_12_0-10d5d8ed8114384ccd7aa22d16089f812fd323285b17146b3d78f10d8bee9700 = buildPythonPackage {
      pname = "pybind11";
      version = "2.12.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/26/55/e776489172f576b782e616f58273e1f3de56a91004b0d20504169dd345af/pybind11-2.12.0-py3-none-any.whl";
        hash="sha256-341guU+ecU2BAT2yMzk9Qw6/nzVRZCuCKRzxsU0a/b0=";
      };
      doCheck = false;
    };
    numpy-2_0_0-d074598ef26d77d12a624d89bf99e6e29040189a02a62f359c06324ed9f8911a = let env = with packages;
    with buildPackages;
    {
      meson-python = meson-python-0_15_0-29ccb277ce25aa5d3be8b684d44e64e4596529292435ff9d4f2df1081c724147;
      pyproject-metadata = pyproject-metadata-0_8_0-1ffbfb773b6e13d61ffaba777a56c22b450209f77f53cd2889e74a6b519702d0;
      packaging = packaging;
      cython = cython-3_0_10-491d7b7995daceea9c248a53c500454f45b4c43da7ec8b48fb8d08dfcc88b340;
      tomli = tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734;
      meson = meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe;
    };
    in (import ./nix-custom/numpy-2.0.1) {
      buildPythonPackage=buildPythonPackage;
      env=env;
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    meson-python-0_15_0-29ccb277ce25aa5d3be8b684d44e64e4596529292435ff9d4f2df1081c724147 = buildPythonPackage {
      pname = "meson-python";
      version = "0.15.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1f/60/b10b11ab470a690d5777310d6cfd1c9bdbbb0a1313a78c34a1e82e0b9d27/meson_python-0.15.0-py3-none-any.whl";
        hash="sha256-OuOCU/8CsulHoF42Ki6vWpoJ0TPFZmtBIzme5fvy5ZE=";
      };
      dependencies = with packages;
      with buildPackages;
      [pyproject-metadata-0_8_0-1ffbfb773b6e13d61ffaba777a56c22b450209f77f53cd2889e74a6b519702d0 tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe];
      doCheck = false;
    };
    setuptools-scm-8_1_0-2d127c77fef402dac40e1ffe1178b28d44a0b4b4c559dba25247a0a18025387c = buildPythonPackage {
      pname = "setuptools-scm";
      version = "8.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/b9/1906bfeb30f2fc13bb39bf7ddb8749784c05faadbd18a21cf141ba37bff2/setuptools_scm-8.1.0-py3-none-any.whl";
        hash="sha256-iXoyJqb9Sm6y8Gh0XklzMmGiH3Cxuyj84DOf65eNmvM=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging setuptools tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734];
      doCheck = false;
    };
    cppy-1_2_1-3830e296859438472c12241d2dda4894cbc93a523576760582e534682058e1af = buildPythonPackage {
      pname = "cppy";
      version = "1.2.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/31/5e/b8faf2b2aeb679c0f4359fd1a4716fe90d65f72f72639413ffb95f3c3b46/cppy-1.2.1-py3-none-any.whl";
        hash="sha256-xbXqw9P0JZOgfTUnWwvCf0R7drmtjyfGLjz6KG3BmIo=";
      };
      doCheck = false;
    };
    py-cpuinfo-9_0_0-e1845c42d339b04877edca808af7a25bf1d94f98a662f25c5acd6a5c34078fbd = buildPythonPackage {
      pname = "py-cpuinfo";
      version = "9.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e0/a9/023730ba63db1e494a271cb018dcd361bd2c917ba7004c3e49d5daf795a2/py_cpuinfo-9.0.0-py3-none-any.whl";
        hash="sha256-hZYlvCUfZOIfB30JnUFiaJx2K11qTDyXVT1WJByWdNU=";
      };
      doCheck = false;
    };
    setuptools-scm-8_1_0-0d702ec1b648b58b185f9cbd34c50e0a44ebfbea10cfb89eb5322a980c098fab = buildPythonPackage {
      pname = "setuptools-scm";
      version = "8.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/b9/1906bfeb30f2fc13bb39bf7ddb8749784c05faadbd18a21cf141ba37bff2/setuptools_scm-8.1.0-py3-none-any.whl";
        hash="sha256-iXoyJqb9Sm6y8Gh0XklzMmGiH3Cxuyj84DOf65eNmvM=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging setuptools tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734];
      doCheck = false;
    };
    etils-1_9_2-b2a70415c9e50ce4f485bc33651027182c5352dc8ba8837bb7cf253a3d555ed2 = buildPythonPackage {
      pname = "etils";
      version = "1.9.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/f4/305f3ea85aecd23422c606c179fb6d00bd7d255b10d55b4c797a3a680144/etils-1.9.2-py3-none-any.whl";
        hash="sha256-7Ned4fv+qbDWkkdWz6kisF7TNgxFzyFwdn2kvuAAHSA=";
      };
      dependencies = with packages;
      with buildPackages;
      [typing-extensions];
      doCheck = false;
    };
    etils-1_9_2-e690b505d64269059a9daf26a1453a5e37a66878ac076ee68c07b35f25c1f91e = buildPythonPackage {
      pname = "etils";
      version = "1.9.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/f4/305f3ea85aecd23422c606c179fb6d00bd7d255b10d55b4c797a3a680144/etils-1.9.2-py3-none-any.whl";
        hash="sha256-7Ned4fv+qbDWkkdWz6kisF7TNgxFzyFwdn2kvuAAHSA=";
      };
      dependencies = with packages;
      with buildPackages;
      [fsspec importlib-resources zipp typing-extensions];
      doCheck = false;
    };
    versioneer-0_29-4a2512b3cdcd7699f877273c71b8d21a04a497b3b5d9ff259b82f25cf1990daf = buildPythonPackage {
      pname = "versioneer";
      version = "0.29";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b0/79/f0f1ca286b78f6f33c521a36b5cbd5bd697c0d66217d8856f443aeb9dd77/versioneer-0.29-py3-none-any.whl";
        hash="sha256-DxoTe7XWgR6Wp5uwSGeYrq6bnG78JLOJZZzrsO45bLk=";
      };
      dependencies = with packages;
      with buildPackages;
      [tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734];
      doCheck = false;
    };
    meson-1_2_1-6ea0d3abd41fff844f366136a50b8f68f75aaf36b46b38d13f0a9dc5f25e44c6 = buildPythonPackage {
      pname = "meson";
      version = "1.2.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e5/74/a1f1c6ba14e11e0fb050d2c61a78b6db108dd38383b6c0ab51c1becbbeff/meson-1.2.1-py3-none-any.whl";
        hash="sha256-CPg/wXUT6ZzW6Cx1VMH1ivcEJSEYh/j5xzY7KpAglGI=";
      };
      doCheck = false;
    };
    meson-python-0_13_1-80c1db885775e48a127ecfcbead2035ed6a1831e80e81ae959ec48d2afd835a5 = buildPythonPackage {
      pname = "meson-python";
      version = "0.13.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9f/af/5f941f57dc516e72b018183a38fbcfb018a7e83afd3c756ecfba82f21c65/meson_python-0.13.1-py3-none-any.whl";
        hash="sha256-4z6j77rezBV2jCBdA7kFx7O/cq+uHh69hLQ4xKPtM5M=";
      };
      dependencies = with packages;
      with buildPackages;
      [pyproject-metadata-0_8_0-1ffbfb773b6e13d61ffaba777a56c22b450209f77f53cd2889e74a6b519702d0 tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 meson-1_5_0-18dba963e11b1db517c2e82626c7fc094984cf6810fc4399da35b316570f8bbe];
      doCheck = false;
    };
    cython-3_0_5-eecf9b399ab12ed1f0673b9f8f31d7c4a637972cd6c1a680d83bd9169e8118dd = buildPythonPackage {
      pname = "cython";
      version = "3.0.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fb/fe/e213d8e9cb21775bb8f9c92ff97861504129e23e33d118be1a90ca26a13e/Cython-3.0.5-py2.py3-none-any.whl";
        hash="sha256-dSBjaVBPxELBCobs9XuRWS3KdE5Fkq8ipH6ad01T3RA=";
      };
      doCheck = false;
    };
    cmake-3_30_1-b57abfab4d4a448ae6ef989de008c7423c88791f6b8101f1927fc6819585a4fc = buildPythonPackage {
      pname = "cmake";
      version = "3.30.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/cd/00/94a5cba1229e1406c032ea6445f387c3cfb9aaa020b87d9160b75ac09ea2/cmake-3.30.1.tar.gz";
        hash="sha256-UfAc5CmlWsv+HxuvUH8P5pFiQ6nz5YaKko0HOuSxjvk=";
      };
      build-system = with packages;
      with buildPackages;
      [scikit-build-core-0_9_8-a954e4ebb2581ba729aed89c26876c0d7beba36b3331e0899e05d2715ba6956a];
      doCheck = false;
    };
    exceptiongroup-1_2_2-58fb8386e8e0e1a53e5968798568874eec31281b14f23555bbb9b2fa9a21f4af = buildPythonPackage {
      pname = "exceptiongroup";
      version = "1.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/02/cc/b7e31358aac6ed1ef2bb790a9746ac2c69bcb3c8588b41616914eb106eaf/exceptiongroup-1.2.2-py3-none-any.whl";
        hash="sha256-MRG50THCOL7C+PUW4SPhS6JDVj+xNdP+iFmQWFqneVs=";
      };
      doCheck = false;
    };
    pathspec-0_12_1-c817e70ea9d1ea0449a0f56bb07885995031db2a10fc8744145c9762b4531f72 = buildPythonPackage {
      pname = "pathspec";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/cc/20/ff623b09d963f88bfde16306a54e12ee5ea43e9b597108672ff3a408aad6/pathspec-0.12.1-py3-none-any.whl";
        hash="sha256-oNUD4TikwSOydJCk977aagHG8ojfDkqLecfrDce0zAg=";
      };
      doCheck = false;
    };
    scikit-build-core-0_9_8-a954e4ebb2581ba729aed89c26876c0d7beba36b3331e0899e05d2715ba6956a = buildPythonPackage {
      pname = "scikit-build-core";
      version = "0.9.8";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/0e/b7/62ede14d44d448bbb7424d5992e394d6980824312de05c9b4816a41602f0/scikit_build_core-0.9.8-py3-none-any.whl";
        hash="sha256-5uzF/Vi2qOv+oOns6qoqaAsVq/WS78k1ITrBXkOah8Y=";
      };
      dependencies = with packages;
      with buildPackages;
      [packaging tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 pathspec-0_12_1-c817e70ea9d1ea0449a0f56bb07885995031db2a10fc8744145c9762b4531f72 exceptiongroup-1_2_2-58fb8386e8e0e1a53e5968798568874eec31281b14f23555bbb9b2fa9a21f4af];
      doCheck = false;
    };
    distro-1_9_0-f026cb8b499a97121ebac8b9d718f8043f2d7e0a7cb8f51465817593a9dc942a = buildPythonPackage {
      pname = "distro";
      version = "1.9.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/12/b3/231ffd4ab1fc9d679809f356cebee130ac7daa00d6d6f3206dd4fd137e9e/distro-1.9.0-py3-none-any.whl";
        hash="sha256-e//ZJdZRaPhQJ9jamva92rZYE1uEBnCiI1ibwMjvArI=";
      };
      doCheck = false;
    };
    scikit-build-0_18_0-bbae32db54641d39652ea79c4cd525d1e885508fa018473f97b9a0720b1a6530 = buildPythonPackage {
      pname = "scikit-build";
      version = "0.18.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/24/43/a0b5837cf30db1561a04187edd262bdefaffcb61222cb441eadef35f9103/scikit_build-0.18.0-py3-none-any.whl";
        hash="sha256-6hcfVSnm4LW2Zhk0M4Ma9hoo1+35c7M4hOyMeCoV7jg=";
      };
      dependencies = with packages;
      with buildPackages;
      [distro-1_9_0-f026cb8b499a97121ebac8b9d718f8043f2d7e0a7cb8f51465817593a9dc942a packaging setuptools tomli-2_0_1-cace7ca51ca218837e00fa1bf284b047cb49d76d19d52a222ba762c7d0050734 wheel-0_43_0-67b69bf1c83179b425d8114821a3d09288d9e983a13eaa8bbb043fabee9c1158];
      doCheck = false;
    };
    setuptools-59_2_0-6d505cd5fc2a97a44b4703fdc7fcb90b83cbb1f9bf979ca01a6ebad8d95064c0 = buildPythonPackage {
      pname = "setuptools";
      version = "59.2.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/18/ad/ec41343a49a0371ea40daf37b1ba2c11333cdd121cb378161635d14b9750/setuptools-59.2.0-py3-none-any.whl";
        hash="sha256-St3j0eHIm94cZDxk2JzdlMv9jHUlLuRZ1FALzLnH0F0=";
      };
      doCheck = false;
    };
    pip-24_1_2-c506f4afc1c13f35b33177ef3b20d80b704ad9bf8866df2a2ef4fc7f25686747 = buildPythonPackage {
      pname = "pip";
      version = "24.1.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e7/54/0c1c068542cee73d8863336e974fc881e608d0170f3af15d0c0f28644531/pip-24.1.2-py3-none-any.whl";
        hash="sha256-fNIH7tTGCw9BG0RM0UZBmP4YZnHDI7bNbUM+2A/J0kc=";
      };
      doCheck = false;
    };
    cython-0_29_37-70ce88e0e16e90456eca2cdd68685260e331aa8da341ec4c95df613a40c8d4a7 = buildPythonPackage {
      pname = "cython";
      version = "0.29.37";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7e/26/9d8de10005fedb1eceabe713348d43bae1dbab1786042ca0751a2e2b0f8c/Cython-0.29.37-py2.py3-none-any.whl";
        hash="sha256-lfHWqD7ycp5ns/pzGMgpzlsHrGTAhM1q8RwijgNkZiw=";
      };
      doCheck = false;
    };
    setuptools-70_1_1-6ef60e6ca78ad9a7bf8979a2fef9ee5079bc76156c8bd044bed129d5743a312a = buildPythonPackage {
      pname = "setuptools";
      version = "70.1.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/7a/629889a5d76200287aa5483d753811bd247bbd1b03175186f759e0c7d3a7/setuptools-70.1.1-py3-none-any.whl";
        hash="sha256-pYqP3gVB2rBBl1C8xSH734WF9uXLQZCd86Ry73uBypU=";
      };
      doCheck = false;
    };
    pkgconfig-1_5_5-0d2df3706e3013a39c01acb40571b9367fe03a4915085e8374ce14040b323035 = buildPythonPackage {
      pname = "pkgconfig";
      version = "1.5.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/32/af/89487c7bbf433f4079044f3dc32f9a9f887597fe04614a37a292e373e16b/pkgconfig-1.5.5-py3-none-any.whl";
        hash="sha256-0gAju+tC7m1Cig+sbgkEYx9UWYWhDN1xogqli8R6Qgk=";
      };
      doCheck = false;
    };
    stanza-0_1_0-c1cde0e216b34de262edead9b59fb5ac0d7734e7e5792a89cd74cab070cfc17e = buildPythonPackage {
      pname = "stanza";
      version = "0.1.0";
      format="pyproject";
      src = ./packages/stanza;
      build-system = with packages;
      with buildPackages;
      [pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722];
      dependencies = with packages;
      with buildPackages;
      [jax numpy beautifulsoup4 trajax sentencepiece h5py rich chex einops zarr mujoco-mjx shapely robosuite pandas ffmpegio flax optax matplotlib plotly nbformat];
      doCheck = false;
    };
    pdm-backend-2_3_3-7cf7f54826f821fbd236f48cf41ebd94382356ec01b3171694c91e418724c722 = buildPythonPackage {
      pname = "pdm-backend";
      version = "2.3.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/eb/fe/483cf0918747a32800795f430319ec292f833eb871ba6da3ebed4553a575/pdm_backend-2.3.3-py3-none-any.whl";
        hash="sha256-226G3oyoTkJkw1piCHexSrqAkq16NN45VxVVMURmiCM=";
      };
      doCheck = false;
    };
  };
}