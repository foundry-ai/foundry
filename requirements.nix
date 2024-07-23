{buildPythonPackage, fetchurl}: rec {
  packages = rec {
    regex = buildPythonPackage {
      pname = "regex";
      version = "2024.5.15";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7a/db/5ddc89851e9cc003929c3b08b9b88b429459bf9acbf307b4556d51d9e49b/regex-2024.5.15.tar.gz";
        hash="sha256-0+4C2eX0gsyDCRNKke6qy90iYboRGw/vN0jutJE+aiw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    sentry-sdk = buildPythonPackage {
      pname = "sentry-sdk";
      version = "2.10.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/34/d8/ec3e43d4ce31e4f4cb6adb7210950362d71ce87a96c89934c4ac94f7110f/sentry_sdk-2.10.0.tar.gz";
        hash="sha256-VF/MbjbDNfqm1s2oRmm24XAl8x7787IhHsFO/gCLddE=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    fonttools = buildPythonPackage {
      pname = "fonttools";
      version = "4.53.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c6/cb/cd80a0da995adde8ade6044a8744aee0da5efea01301cadf770f7fbe7dcc/fonttools-4.53.1.tar.gz";
        hash="sha256-4Sh3io6bwRFZzlRH92dmzvvYdvRL15r/AwKHJU5HUsQ=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/63/70/2bf7780ad2d390a8d301ad0b550f1581eadbd9a20f896afe06353c2a2913/requests-2.32.3.tar.gz";
        hash="sha256-VTZUF3NOsYJVWQqf+euX6eHaho1MzWQCOZ6vaK8gp2A=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      doCheck = false;
    };
    gitpython = buildPythonPackage {
      pname = "gitpython";
      version = "3.1.43";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b6/a1/106fd9fa2dd989b6fb36e5893961f82992cf676381707253e0bf93eb1662/GitPython-3.1.43.tar.gz";
        hash="sha256-NfMUqfh4Rn9UU8wf7ilcPhjlLxuZ8Q9s9bFoLpaKnnw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    rpds-py = buildPythonPackage {
      pname = "rpds-py";
      version = "0.19.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/66/7c/207cecba303ceba1e8b435bd322a221898190b2abf9f6a09828dfb2e2e2c/rpds_py-0.19.0-cp312-cp312-manylinux_2_17_ppc64le.manylinux2014_ppc64le.whl";
        hash="sha256-aIqmuKpyTbFZZRR1H/t2d2bgLlxKh0hqs2uOHrwa7aw=";
      };
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
      doCheck = false;
    };
    tqdm = buildPythonPackage {
      pname = "tqdm";
      version = "4.66.4";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5a/c0/b7599d6e13fe0844b0cda01b9aaef9a0e87dbb10b06e4ee255d3fa1c79a2/tqdm-4.66.4.tar.gz";
        hash="sha256-5Nk2yd6HJ5KPO+YHlZDpfZq/6NOaWQvmeOtZGf/Bhrs=";
      };
      build-system = with packages;
      with buildPackages;
      [packaging setuptools setuptools-scm_8_1_0 wheel_0_43_0];
      doCheck = false;
    };
    protobuf = buildPythonPackage {
      pname = "protobuf";
      version = "5.27.2";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/71/a5/d61e4263e62e6db1990c120d682870e5c50a30fb6b26119a214c7a014847/protobuf-5.27.2.tar.gz";
        hash="sha256-8+ze8ia5r4VgdfKCJ/8skM46WU0JLDm+5VE1c/JeJxQ=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    soupsieve = buildPythonPackage {
      pname = "soupsieve";
      version = "2.5";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ce/21/952a240de1c196c7e3fbcd4e559681f0419b1280c617db21157a0390717b/soupsieve-2.5.tar.gz";
        hash="sha256-VmPVp7O/ru4LxDcuf8SPnP9JQLPuxUpkUcxSmfEJdpA=";
      };
      build-system = with packages;
      with buildPackages;
      [hatchling_1_25_0 packaging pathspec_0_12_1 pluggy trove-classifiers_2024_7_2];
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
      doCheck = false;
    };
    pytz = buildPythonPackage {
      pname = "pytz";
      version = "2024.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/90/26/9f1f00a5d021fff16dee3de13d43e5e978f3d58928e129c3a62cf7eb9738/pytz-2024.1.tar.gz";
        hash="sha256-KilzXqnBi68UtEiEa95aSAMO0mdXhHLYlVzQ50Q6mBI=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    python-xlib = buildPythonPackage {
      pname = "python-xlib";
      version = "0.33";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/86/f5/8c0653e5bb54e0cbdfe27bf32d41f27bc4e12faa8742778c17f2a71be2c0/python-xlib-0.33.tar.gz";
        hash="sha256-Va95BqLHXObLKApYR3YIBgJET3WBWnr/TSh7stcBizI=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    typing-extensions = buildPythonPackage {
      pname = "typing-extensions";
      version = "4.12.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/df/db/f35a00659bc03fec321ba8bce9420de607a1d37f8342eee1863174c69557/typing_extensions-4.12.2.tar.gz";
        hash="sha256-Gn6tVcflWd1N7ohW46iLQSJav+HOjfV7fBORX+Eh/7g=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
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
    click = buildPythonPackage {
      pname = "click";
      version = "8.1.7";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/96/d3/f04c7bfcf5c1862a2a5b845c6b2b360488cf47af55dfa79c98f6a6bf98b5/click-8.1.7.tar.gz";
        hash="sha256-yphTrUWeeH4hkiEVeMyQfnWU4pTHzMg0MQcitBucpt4=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    absl-py = buildPythonPackage {
      pname = "absl-py";
      version = "2.1.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7a/8f/fc001b92ecc467cc32ab38398bd0bfb45df46e7523bf33c2ad22a505f06e/absl-py-2.1.0.tar.gz";
        hash="sha256-eCB5DvuzFnOc3otOGTVyQ/w2CKFSAkKIUT3ZaNfZWf8=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    llvmlite = buildPythonPackage {
      pname = "llvmlite";
      version = "0.43.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9f/3d/f513755f285db51ab363a53e898b85562e950f79a2e6767a364530c2f645/llvmlite-0.43.0.tar.gz";
        hash="sha256-ritbXD72c1SCT7dVF8jbX76TvALNlnHzxiJxYmvAQdU=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
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
      [meson_1_5_0 meson-python_0_16_0 packaging pybind11_2_13_1 pyproject-metadata_0_8_0];
      doCheck = false;
    };
    packaging = buildPythonPackage {
      pname = "packaging";
      version = "24.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/51/65/50db4dda066951078f0a96cf12f4b9ada6e4b811516bf0262c0f4f7064d4/packaging-24.1.tar.gz";
        hash="sha256-Am7XLI7T/M5b+JUFciWGmJJ/0dvaEKXpgc3wrDf08AI=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
      doCheck = false;
    };
    kiwisolver = buildPythonPackage {
      pname = "kiwisolver";
      version = "1.4.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f3/70/26c99be8eb034cc8e3f62e0760af1fbdc97a842a7cbc252f7978507d41c2/kiwisolver-1.4.5-cp312-cp312-manylinux_2_17_ppc64le.manylinux2014_ppc64le.whl";
        hash="sha256-OrpzEa+C4zXdHjb//2iqymCcpikMLLbYIaOaoHXY4/8=";
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
      [cython_3_0_10 meson_1_5_0 meson-python_0_16_0 numpy_2_0_1 packaging pybind11_2_12_0 pyproject-metadata_0_8_0 pythran_0_16_1];
      doCheck = false;
    };
    evdev = buildPythonPackage {
      pname = "evdev";
      version = "1.7.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/12/bb/f622a8a5e64d46ca83020a761877c0ead19140903c9aaf1431f3c531fdf6/evdev-1.7.1.tar.gz";
        hash="sha256-DHLDcL2inYV+GI2TEBnDJlGpweqXfAjI2TmxztFjf94=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    numcodecs = buildPythonPackage {
      pname = "numcodecs";
      version = "0.13.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f8/22/e5cba9013403186906390c0efb0ab0db60d4e580a8966650b2372ab967e1/numcodecs-0.13.0.tar.gz";
        hash="sha256-uk+scDbqWgeMev4dTf/rloUIDULxnJwWsS2thmcDqi4=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 numpy_2_0_1 packaging py-cpuinfo_9_0_0 setuptools setuptools-scm_8_1_0];
      doCheck = false;
    };
    toolz = buildPythonPackage {
      pname = "toolz";
      version = "0.12.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3e/bf/5e12db234df984f6df3c7f12f1428aa680ba4e101f63f4b8b3f9e8d2e617/toolz-0.12.1.tar.gz";
        hash="sha256-7Mo0JmSJPxd6E9rA5rQcvYrCWjWOXyFTFtQ+IQAiT00=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    tzdata = buildPythonPackage {
      pname = "tzdata";
      version = "2024.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/74/5b/e025d02cb3b66b7b76093404392d4b44343c69101cc85f4d180dd5784717/tzdata-2024.1.tar.gz";
        hash="sha256-JnQSD42JGQl1HDirzf04asCloRJ5VPvDMq9rXOrgfv0=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    nbformat = buildPythonPackage {
      pname = "nbformat";
      version = "5.10.4";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6d/fd/91545e604bc3dad7dca9ed03284086039b294c6b3d75c0d2fa45f9e9caf3/nbformat-5.10.4.tar.gz";
        hash="sha256-MiFosU+Tel0RNimI7KwqSVLT2OOiy+sjGVhGMSJtWzo=";
      };
      build-system = with packages;
      with buildPackages;
      [hatch-nodejs-version_0_3_2 hatchling_1_25_0 packaging pathspec_0_12_1 pluggy trove-classifiers_2024_7_2];
      doCheck = false;
    };
    huggingface-hub = buildPythonPackage {
      pname = "huggingface-hub";
      version = "0.24.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/66/84/9240cb3fc56112c7093ef84ece44a555386263e7a19c81a4c847fd7e2bba/huggingface_hub-0.24.0.tar.gz";
        hash="sha256-bHCSc2tXfYnVezzf6gJvGw3CI0rng/oNWcrxv31S36c=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    plotly = buildPythonPackage {
      pname = "plotly";
      version = "5.22.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/4a/42/e16addffa3eee93bde84aceee20e3eaf579d1df554633c884d50b050b466/plotly-5.22.0.tar.gz";
        hash="sha256-hZ/a29hrV3CuJGblQrdhskfRxrSdrtdluVu4xwY+dGk=";
      };
      build-system = with packages;
      with buildPackages;
      [aiofiles_22_1_0 aiosqlite_0_20_0 anyio_4_4_0 argon2-cffi_23_1_0 argon2-cffi-bindings_21_2_0 asttokens_2_4_1 attrs babel_2_15_0 beautifulsoup4 bleach_6_1_0 certifi cffi_1_16_0 charset-normalizer comm_0_2_2 debugpy_1_8_2 decorator_5_1_1 defusedxml_0_7_1 entrypoints_0_4 executing_2_0_1 fastjsonschema idna ipykernel_6_29_5 ipython_8_26_0 ipython-genutils_0_2_0 jedi_0_19_1 jinja2_3_1_4 json5_0_9_25 jsonschema jsonschema-specifications jupyter-client_7_4_9 jupyter-core jupyter-events_0_10_0 jupyter-server_2_14_2 jupyter-server-fileid_0_9_2 jupyter-server-terminals_0_5_3 jupyter-server-ydoc_0_8_0 jupyter-ydoc_0_2_5 jupyterlab_3_6_7 jupyterlab-pygments_0_3_0 jupyterlab-server_2_27_3 markupsafe_2_1_5 matplotlib-inline_0_1_7 mistune_3_0_2 nbclassic_1_1_0 nbclient_0_10_0 nbconvert_7_16_4 nbformat nest-asyncio notebook_6_5_7 notebook-shim_0_2_4 overrides_7_7_0 packaging pandocfilters_1_5_1 parso_0_8_4 pexpect_4_9_0 platformdirs prometheus-client_0_20_0 prompt-toolkit_3_0_47 psutil ptyprocess_0_7_0 pure-eval_0_2_3 pycparser_2_22 pygments python-dateutil python-json-logger_2_0_7 pyyaml pyzmq_26_0_3 referencing requests rfc3339-validator_0_1_4 rfc3986-validator_0_1_1 rpds-py send2trash_1_8_3 setuptools six sniffio_1_3_1 soupsieve stack-data_0_6_3 terminado_0_18_1 tinycss2_1_3_0 tornado_6_4_1 traitlets typing-extensions urllib3 wcwidth_0_2_13 webencodings_0_5_1 websocket-client_1_8_0 wheel_0_43_0 y-py_0_6_2 ypy-websocket_0_8_4];
      doCheck = false;
    };
    cycler = buildPythonPackage {
      pname = "cycler";
      version = "0.12.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a9/95/a3dbbb5028f35eafb79008e7522a75244477d2838f38cbb722248dabc2a8/cycler-0.12.1.tar.gz";
        hash="sha256-iLsSjwK6NB2o70RyRanhOPrnd/aiOUPaRUAHfTYB6xw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      [meson_1_5_0 meson-python_0_16_0 numpy_2_0_1 packaging pybind11_2_13_1 pyproject-metadata_0_8_0 setuptools setuptools-scm_8_1_0];
      doCheck = false;
    };
    optax = buildPythonPackage {
      pname = "optax";
      version = "0.2.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d6/5f/e8b09028b37a8c1c159359e59469f3504b550910d472d8ee59543b1735d9/optax-0.2.3.tar.gz";
        hash="sha256-7Hq5JUQLDFpRLh8k+6D7Pn12Cn/V0kltemkenTfaAdk=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
      doCheck = false;
    };
    nest-asyncio = buildPythonPackage {
      pname = "nest-asyncio";
      version = "1.6.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/83/f8/51569ac65d696c8ecbee95938f89d4abf00f47d58d48f6fbabfe8f0baefe/nest_asyncio-1.6.0.tar.gz";
        hash="sha256-bxctVEmsoVr9bGRoUfTjHgLFmNVTpmfjjK+pl8/sVf4=";
      };
      build-system = with packages;
      with buildPackages;
      [packaging setuptools setuptools-scm_8_1_0 wheel_0_43_0];
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
      [numpy_2_0_1 packaging setuptools setuptools-scm_8_1_0 wheel_0_43_0];
      doCheck = false;
    };
    orbax-checkpoint = buildPythonPackage {
      pname = "orbax-checkpoint";
      version = "0.5.22";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f1/43/4b36ff621eef197083a3b700a60848b0c110cdfdea5f8e0eedff9b45c5a7/orbax_checkpoint-0.5.22.tar.gz";
        hash="sha256-VNU7yA64PkuqYpQzsG/moHNqG3aM3E792mNVhN4Q7SQ=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
      doCheck = false;
    };
    msgpack = buildPythonPackage {
      pname = "msgpack";
      version = "1.0.8";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/08/4c/17adf86a8fbb02c144c7569dc4919483c01a2ac270307e2d59e1ce394087/msgpack-1.0.8.tar.gz";
        hash="sha256-lcArDifnBuSNDlQm0XEMp44PBijW6J1bWluRpfEidPM=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 setuptools];
      doCheck = false;
    };
    flax = buildPythonPackage {
      pname = "flax";
      version = "0.8.5";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/29/fb/b259e2e33e5740c62fc05677c395e96809e83c0467677d3e714b125ec24c/flax-0.8.5.tar.gz";
        hash="sha256-Spy3lQ7OVLCt2qc9d+uiTkYTjb54PQGYe+edIMyysJs=";
      };
      build-system = with packages;
      with buildPackages;
      [packaging setuptools setuptools-scm_8_1_0];
      doCheck = false;
    };
    pluggy = buildPythonPackage {
      pname = "pluggy";
      version = "1.5.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/96/2d/02d4312c973c6050a18b314a5ad0b3210edb65a906f868e31c111dede4a6/pluggy-1.5.0.tar.gz";
        hash="sha256-LP+ojpT9yXjExXTxX55Zt/QgHUORlcNxXKniSG8dDPE=";
      };
      build-system = with packages;
      with buildPackages;
      [packaging setuptools setuptools-scm_8_1_0];
      doCheck = false;
    };
    ffmpegio-core = buildPythonPackage {
      pname = "ffmpegio-core";
      version = "0.10.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c7/f8/d0efc4022efdcc5fe0100010034bce2e4c3c39b1332733a56af25a9b539b/ffmpegio_core-0.10.0.tar.gz";
        hash="sha256-qla1BCc+4qZJYNqhWsB5YMbZ3MleAoIenCZKJqP19Xo=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    ffmpegio = buildPythonPackage {
      pname = "ffmpegio";
      version = "0.10.0.post0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/39/32/663b4b9bccd0b950bdc7a4e335d016920dc861ce7afdd92914e64ee7bacd/ffmpegio-0.10.0.post0.tar.gz";
        hash="sha256-1ocuTX4otvN7Uqc55lL5ZQtrm/WgjPmWE+WbFFLgDWU=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    python-dateutil = buildPythonPackage {
      pname = "python-dateutil";
      version = "2.9.0.post0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/66/c0/0c8b6ad9f17a802ee498c46e004a0eb49bc148f2fd230864601a86dcf6db/python-dateutil-2.9.0.post0.tar.gz";
        hash="sha256-N91UII2n4c2HU4ghfV4A69QXkkn5D7ckN+kaNUWaCtM=";
      };
      build-system = with packages;
      with buildPackages;
      [packaging setuptools setuptools-scm_7_1_0 typing-extensions wheel_0_43_0];
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
      [maturin_1_7_0];
      doCheck = false;
    };
    mdurl = buildPythonPackage {
      pname = "mdurl";
      version = "0.1.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d6/54/cfe61301667036ec958cb99bd3efefba235e65cdeb9c84d24a8293ba1d90/mdurl-0.1.2.tar.gz";
        hash="sha256-u0E9KfXuo48x3UdU3XN31EZRFvsgdYX5e/klWIaHwbo=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
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
      [cython_3_0_5 meson_1_2_1 meson-python_0_13_1 numpy_2_0_1 packaging pyproject-metadata_0_8_0 setuptools versioneer_0_29 wheel_0_43_0];
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
      [maturin_1_7_0];
      doCheck = false;
    };
    filelock = buildPythonPackage {
      pname = "filelock";
      version = "3.15.4";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/08/dd/49e06f09b6645156550fb9aee9cc1e59aba7efbc972d665a1bd6ae0435d4/filelock-3.15.4.tar.gz";
        hash="sha256-IgeTjLwYRDRcsBpalVJNrjDwzgieulsAN4KVoX4+kMs=";
      };
      build-system = with packages;
      with buildPackages;
      [hatch-vcs_0_4_0 hatchling_1_25_0 packaging pathspec_0_12_1 pluggy setuptools setuptools-scm_8_1_0 trove-classifiers_2024_7_2];
      doCheck = false;
    };
    transformers = buildPythonPackage {
      pname = "transformers";
      version = "4.42.4";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/84/eb/259afff0df9ece338dc224007bbe7dd6c9aae8e26957dc4033a3ec857588/transformers-4.42.4.tar.gz";
        hash="sha256-+VbiXiTfhR9lDLLBWLb0NS366dcC8EwRPtJPw2znri0=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    termcolor = buildPythonPackage {
      pname = "termcolor";
      version = "2.4.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/10/56/d7d66a84f96d804155f6ff2873d065368b25a07222a6fd51c4f24ef6d764/termcolor-2.4.0.tar.gz";
        hash="sha256-qrnlYEfIrEHteY+jbYkqN6yms+kVnz4MJLxkqbOse3o=";
      };
      build-system = with packages;
      with buildPackages;
      [hatch-vcs_0_4_0 hatchling_1_25_0 packaging pathspec_0_12_1 pluggy setuptools setuptools-scm_8_1_0 trove-classifiers_2024_7_2];
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
      doCheck = false;
    };
    pillow = buildPythonPackage {
      pname = "pillow";
      version = "10.4.0";
      format="pyproject";
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
      [cmake_3_30_1 distro_1_9_0 numpy_2_0_1 packaging pip_24_1_2 scikit-build_0_18_0 setuptools_59_2_0 wheel_0_43_0];
      doCheck = false;
    };
    robosuite = buildPythonPackage {
      pname = "robosuite";
      version = "1.4.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/be/82/50fd8fa786e1bbd27d7de7ae9b8c6ca8a36df791c351bee20ef4cbf27687/robosuite-1.4.1.tar.gz";
        hash="sha256-4gmw94IbuEsr83Y662T6lCrZ1YoVGe6/s5+7aCMeu0I=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    shapely = buildPythonPackage {
      pname = "shapely";
      version = "2.0.5";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ad/99/c47247f4d688bbb5346df5ff1de5d9792b6d95cbbb2fd7b71f45901c1878/shapely-2.0.5.tar.gz";
        hash="sha256-v/I2a8eGv6bLNT1rR9BEPFcMMndmEuUn7ke232P8/jI=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 numpy_2_0_1 setuptools];
      doCheck = false;
    };
    pyopengl = buildPythonPackage {
      pname = "pyopengl";
      version = "3.1.7";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/72/b6/970868d44b619292f1f54501923c69c9bd0ab1d2d44cf02590eac2706f4f/PyOpenGL-3.1.7.tar.gz";
        hash="sha256-7vMaOIjmmE/U2ObJlhsYTJgTyoJgTTf+PagOsACnbIY=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools_59_2_0];
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
    trimesh = buildPythonPackage {
      pname = "trimesh";
      version = "4.4.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/38/9d/7cd8987c7df13ed4104ef103b732abc8e3bee5b4a3077e890d16322e9fdf/trimesh-4.4.3.tar.gz";
        hash="sha256-pBEK1oMtI8z03zKHKjgE7uohZCE1KRPRUU7Z2tIAHV4=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    mujoco = buildPythonPackage {
      pname = "mujoco";
      version = "3.2.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fc/70/e94bb93c0dad847b7fe13c11ff510253583c610dbc04ec4bf191267292dc/mujoco-3.2.0.tar.gz";
        hash="sha256-R388jEIbzd60en1SRC8mKSSlvdW/xWl1xDnUET7QvKc=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools_59_2_0];
      doCheck = false;
    };
    etils = buildPythonPackage {
      pname = "etils";
      version = "1.9.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ba/49/d480aeb4fc441d933acce97261bea002234a45fb847599c9a93c31e51b2e/etils-1.9.2.tar.gz";
        hash="sha256-FdzTWsDAzCQEtGrAhGrzzE6Hb9PYDzb1eVHifoudY3k=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
      doCheck = false;
    };
    mujoco-mjx = buildPythonPackage {
      pname = "mujoco-mjx";
      version = "3.2.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/29/34/20fead94e402b786f8d0cb57df23530c3a9517b5893c8af0a47261d40b27/mujoco_mjx-3.2.0.tar.gz";
        hash="sha256-If1qveAjGvURvUEmAkHy8UnVTDmsSYNapvPSfff95P8=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools_59_2_0];
      doCheck = false;
    };
    fasteners = buildPythonPackage {
      pname = "fasteners";
      version = "0.19";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5f/d4/e834d929be54bfadb1f3e3b931c38e956aaa3b235a46a3c764c26c774902/fasteners-0.19.tar.gz";
        hash="sha256-tPN8OsUtikRa86ZrzlezO16QuXxpa3uYT1MM+PDe0Jw=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools_59_2_0 wheel_0_43_0];
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
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/00/ac5c518ff1c1b1cc87a62f86ad9d19c647c19d969a91faa40d3b6342ccaa/zarr-2.18.2.tar.gz";
        hash="sha256-m7OTuKCjj7Eh27kTsEfXXbKN6YkPbWRKIXpzz0rnT0c=";
      };
      build-system = with packages;
      with buildPackages;
      [packaging setuptools setuptools-scm_8_1_0];
      doCheck = false;
    };
    einops = buildPythonPackage {
      pname = "einops";
      version = "0.8.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/79/ca/9f5dcb8bead39959454c3912266bedc4c315839cee0e0ca9f4328f4588c1/einops-0.8.0.tar.gz";
        hash="sha256-Y0hlF/7TRXEqg4XBAMsnkQjZ1H5q5ZCZsHZX6YPeroU=";
      };
      build-system = with packages;
      with buildPackages;
      [hatchling_1_25_0 packaging pathspec_0_12_1 pluggy trove-classifiers_2024_7_2];
      doCheck = false;
    };
    chex = buildPythonPackage {
      pname = "chex";
      version = "0.1.86";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/26/a2/46649fb9f6a33cc7c2822161cc5481f0ffe5965fde1e6fc4c3003cd22323/chex-0.1.86.tar.gz";
        hash="sha256-6LD5YzDrpBRGWeFhfA96V7Fh6MuwIeVcbVBWxzeAkdE=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    pygments = buildPythonPackage {
      pname = "pygments";
      version = "2.18.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/8e/62/8336eff65bcbc8e4cb5d05b55faf041285951b6e80f33e2bff2024788f31/pygments-2.18.0.tar.gz";
        hash="sha256-eG/4AvMukTEb/ziJ9umoboFQX+mfJzW7bWCuDFAE8Zk=";
      };
      build-system = with packages;
      with buildPackages;
      [hatchling_1_25_0 packaging pathspec_0_12_1 pluggy trove-classifiers_2024_7_2];
      doCheck = false;
    };
    markdown-it-py = buildPythonPackage {
      pname = "markdown-it-py";
      version = "3.0.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/38/71/3b932df36c1a044d397a1f92d1cf91ee0a503d91e470cbd670aa66b07ed0/markdown-it-py-3.0.0.tar.gz";
        hash="sha256-4/YKlPoGbcUux2Zh43yFHLIy2S+YhrFctWCqraLfj+s=";
      };
      build-system = with packages;
      with buildPackages;
      [flit-core_3_9_0];
      doCheck = false;
    };
    rich = buildPythonPackage {
      pname = "rich";
      version = "13.7.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/01/c954e134dc440ab5f96952fe52b4fdc64225530320a910473c1fe270d9aa/rich-13.7.1.tar.gz";
        hash="sha256-m+MIyx/i8fV9Z86Z6Vrzih4rxxrZgTsOJHz3/7zDpDI=";
      };
      build-system = with packages;
      with buildPackages;
      [poetry-core_1_9_0];
      doCheck = false;
    };
    setuptools = buildPythonPackage {
      pname = "setuptools";
      version = "71.1.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/32/c0/5b8013b5a812701c72e3b1e2b378edaa6514d06bee6704a5ab0d7fa52931/setuptools-71.1.0.tar.gz";
        hash="sha256-Ay1C7p+1NuMwh/tmysX4QOuTke0FY3s/KnanyPtHeTY=";
      };
      build-system = with packages;
      with buildPackages;
      [];
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
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f5/52/0763d1d976d5c262df53ddda8d8d4719eedf9594d046f117c25a27261a19/platformdirs-4.2.2.tar.gz";
        hash="sha256-OLe1H1Eu7Z6EoieItLzh3hfArbE01r7LCYNuN9hlTNM=";
      };
      build-system = with packages;
      with buildPackages;
      [hatch-vcs_0_4_0 hatchling_1_25_0 packaging pathspec_0_12_1 pluggy setuptools setuptools-scm_8_1_0 trove-classifiers_2024_7_2];
      doCheck = false;
    };
    docker-pycreds = buildPythonPackage {
      pname = "docker-pycreds";
      version = "0.4.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c5/e6/d1f6c00b7221e2d7c4b470132c931325c8b22c51ca62417e300f5ce16009/docker-pycreds-0.4.0.tar.gz";
        hash="sha256-bOMnC8r0BMxMPifktscNNSHeroL7UIdnhw/b93LVhNQ=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      doCheck = false;
    };
    six = buildPythonPackage {
      pname = "six";
      version = "1.16.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/71/39/171f1c67cd00715f190ba0b100d606d440a28c93c7714febeca8b79af85e/six-1.16.0.tar.gz";
        hash="sha256-HmHDdHehYmRY4297HYKqXJsJT6SAKJIHLknenGDEySY=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      [cython_0_29_37 setuptools wheel_0_43_0];
      doCheck = false;
    };
    contextlib2 = buildPythonPackage {
      pname = "contextlib2";
      version = "21.6.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c7/13/37ea7805ae3057992e96ecb1cffa2fa35c2ef4498543b846f90dd2348d8f/contextlib2-21.6.0.tar.gz";
        hash="sha256-qx4r/h0B2Wjht+jZAjvFHvNQm7ohe7cwzuOCfh7oKGk=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
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
      doCheck = false;
    };
    opt-einsum = buildPythonPackage {
      pname = "opt-einsum";
      version = "3.3.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7d/bf/9257e53a0e7715bc1127e15063e831f076723c6cd60985333a1c18878fb8/opt_einsum-3.3.0.tar.gz";
        hash="sha256-WfZHX3e7w33PfNdIUZwOxgci6R5jyhFOaIIcDFSkZUk=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    ml-dtypes = buildPythonPackage {
      pname = "ml-dtypes";
      version = "0.4.0";
      format="pyproject";
      src = /home/dpfrom/satori/ml_dtypes-0.4.0;
      build-system = with packages;
      with buildPackages;
      [numpy_2_0_0 setuptools_68_1_2];
      doCheck = false;
    };
    jaxlib = buildPythonPackage {
      pname = "jaxlib";
      version = "0.4.30";
      src = /home/dpfrom/satori/jaxlib-0.4.30;
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    h5py = buildPythonPackage {
      pname = "h5py";
      version = "3.11.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/52/8f/e557819155a282da36fb21f8de4730cfd10a964b52b3ae8d20157ac1c668/h5py-3.11.0.tar.gz";
        hash="sha256-e36PeAcqLt7IfJg28l80ID/UkqRHVwmhi0F6M8+yH6k=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 numpy_2_0_1 pkgconfig_1_5_5 setuptools];
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
      doCheck = false;
    };
    beautifulsoup4 = buildPythonPackage {
      pname = "beautifulsoup4";
      version = "4.12.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/ca/824b1195773ce6166d388573fc106ce56d4a805bd7427b624e063596ec58/beautifulsoup4-4.12.3.tar.gz";
        hash="sha256-dOPRko7cBw0hdIGFxG4/szSQ8i9So63e6a7g9Pd4EFE=";
      };
      build-system = with packages;
      with buildPackages;
      [hatchling_1_25_0 packaging pathspec_0_12_1 pluggy trove-classifiers_2024_7_2];
      doCheck = false;
    };
    numpy = buildPythonPackage {
      pname = "numpy";
      version = "1.26.4";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/65/6e/09db70a523a96d25e115e71cc56a6f9031e7b8cd166c1ac8438307c14058/numpy-1.26.4.tar.gz";
        hash="sha256-KgKrqe0S5KxOs+qUIcQgMBoMZGDZgw10qd+H76SRIBA=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 meson_1_5_0 meson-python_0_15_0 packaging pyproject-metadata_0_8_0];
      doCheck = false;
    };
    jax = buildPythonPackage {
      pname = "jax";
      version = "0.4.30";
      src = /home/dpfrom/satori/jax-0.4.30;
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    language-model = buildPythonPackage {
      pname = "language-model";
      version = "0.1.0";
      format="pyproject";
      src = /home/dpfrom/stanza/projects/language-model;
      build-system = with packages;
      with buildPackages;
      [pdm-backend_2_3_3];
      doCheck = false;
    };
    image-classifier = buildPythonPackage {
      pname = "image-classifier";
      version = "0.1.0";
      format="pyproject";
      src = /home/dpfrom/stanza/projects/image-classifier;
      build-system = with packages;
      with buildPackages;
      [pdm-backend_2_3_3];
      doCheck = false;
    };
    cond-diffusion = buildPythonPackage {
      pname = "cond-diffusion";
      version = "0.1.0";
      format="pyproject";
      src = /home/dpfrom/stanza/projects/cond-diffusion;
      build-system = with packages;
      with buildPackages;
      [pdm-backend_2_3_3];
      doCheck = false;
    };
    stanza-models = buildPythonPackage {
      pname = "stanza-models";
      version = "0.1.0";
      format="pyproject";
      src = /home/dpfrom/stanza/projects/models;
      build-system = with packages;
      with buildPackages;
      [pdm-backend_2_3_3];
      doCheck = false;
    };
    stanza = buildPythonPackage {
      pname = "stanza";
      version = "0.1.0";
      format="pyproject";
      src = /home/dpfrom/stanza/packages/stanza;
      build-system = with packages;
      with buildPackages;
      [pdm-backend_2_3_3];
      doCheck = false;
    };
    stanza-meta = buildPythonPackage {
      pname = "stanza-meta";
      version = "0.1.0";
      format="pyproject";
      src = /home/dpfrom/stanza;
      build-system = with packages;
      with buildPackages;
      [pdm-backend_2_3_3];
      doCheck = false;
    };
  };
  buildPackages = rec {
    setuptools-scm_8_1_0 = buildPythonPackage {
      pname = "setuptools-scm";
      version = "8.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/b9/1906bfeb30f2fc13bb39bf7ddb8749784c05faadbd18a21cf141ba37bff2/setuptools_scm-8.1.0-py3-none-any.whl";
        hash="sha256-iXoyJqb9Sm6y8Gh0XklzMmGiH3Cxuyj84DOf65eNmvM=";
      };
      doCheck = false;
    };
    wheel_0_43_0 = buildPythonPackage {
      pname = "wheel";
      version = "0.43.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7d/cd/d7460c9a869b16c3dd4e1e403cce337df165368c71d6af229a74699622ce/wheel-0.43.0-py3-none-any.whl";
        hash="sha256-VcVwQF8UJjDGufcv4J2bZ88Ud/z1Q65bjcsfW3N32oE=";
      };
      doCheck = false;
    };
    pathspec_0_12_1 = buildPythonPackage {
      pname = "pathspec";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/cc/20/ff623b09d963f88bfde16306a54e12ee5ea43e9b597108672ff3a408aad6/pathspec-0.12.1-py3-none-any.whl";
        hash="sha256-oNUD4TikwSOydJCk977aagHG8ojfDkqLecfrDce0zAg=";
      };
      doCheck = false;
    };
    trove-classifiers_2024_7_2 = buildPythonPackage {
      pname = "trove-classifiers";
      version = "2024.7.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/0f/b0/09794439a62a7dc18bffdbf145aaf50297fd994890b11da27a13e376b947/trove_classifiers-2024.7.2-py3-none-any.whl";
        hash="sha256-zMV6M3F2RN9NrKAY5+w+9XqDXEjpah5x/AfrftrGevY=";
      };
      doCheck = false;
    };
    hatchling_1_25_0 = buildPythonPackage {
      pname = "hatchling";
      version = "1.25.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/0c/8b/90e80904fdc24ce33f6fc6f35ebd2232fe731a8528a22008458cf197bc4d/hatchling-1.25.0-py3-none-any.whl";
        hash="sha256-tHlI5F1NlzA0WE3UyznBS2pwInzyh6t+wK15g0CKiCw=";
      };
      doCheck = false;
    };
    flit-core_3_9_0 = buildPythonPackage {
      pname = "flit-core";
      version = "3.9.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/38/45/618e84e49a6c51e5dd15565ec2fcd82ab273434f236b8f108f065ded517a/flit_core-3.9.0-py3-none-any.whl";
        hash="sha256-eq2jUvsMf1U4xPr+3fMU06apLujisd5wSCMp5C3nAwE=";
      };
      doCheck = false;
    };
    meson_1_5_0 = buildPythonPackage {
      pname = "meson";
      version = "1.5.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c4/cd/47e45d3abada2e1edb9e2ca9334be186d2e7f97a01b09b5b82799c4d7bd3/meson-1.5.0-py3-none-any.whl";
        hash="sha256-UrNPSQO4gt9SrQ1TMUbUuZLAGOp3OZ+CVXlzdnKueyA=";
      };
      doCheck = false;
    };
    pyproject-metadata_0_8_0 = buildPythonPackage {
      pname = "pyproject-metadata";
      version = "0.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/aa/5f/bb5970d3d04173b46c9037109f7f05fc8904ff5be073ee49bb6ff00301bc/pyproject_metadata-0.8.0-py3-none-any.whl";
        hash="sha256-rYWNRI4dOh+0CKxbrJ6ndD56i7tHLyaTqqM00ttC9SY=";
      };
      doCheck = false;
    };
    meson-python_0_16_0 = buildPythonPackage {
      pname = "meson-python";
      version = "0.16.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/91/c0/104cb6244c83fe6bc3886f144cc433db0c0c78efac5dc00e409a5a08c87d/meson_python-0.16.0-py3-none-any.whl";
        hash="sha256-hC3J9dwp5V/Haf8bb+MoQS/myHAiD8MhBgodLTleaeg=";
      };
      doCheck = false;
    };
    pybind11_2_13_1 = buildPythonPackage {
      pname = "pybind11";
      version = "2.13.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/84/fb/1a249de406daf2b4ebd2d714b739e8519034617daec085e3833c1a3ed57c/pybind11-2.13.1-py3-none-any.whl";
        hash="sha256-l4gVNqvgzUJgqczFv20c8xEzGPCK8f64LUuflek/CqQ=";
      };
      doCheck = false;
    };
    pythran_0_16_1 = buildPythonPackage {
      pname = "pythran";
      version = "0.16.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/73/32/f892675c5009cd4c1895ded3d6153476bf00adb5ad1634d03635620881f5/pythran-0.16.1.tar.gz";
        hash="sha256-hhdIwPnH1CKzJySxFLOBfYGO1Oq4bAl4GqCj986rt/k=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    cython_3_0_10 = buildPythonPackage {
      pname = "cython";
      version = "3.0.10";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b6/83/b0a63fc7b315edd46821a1a381d18765c1353d201246da44558175cddd56/Cython-3.0.10-py2.py3-none-any.whl";
        hash="sha256-/LtnnAtDUU1ZFXf9DSACHFXCQMqcyvvbgtP7leXt/uI=";
      };
      doCheck = false;
    };
    numpy_2_0_1 = buildPythonPackage {
      pname = "numpy";
      version = "2.0.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1c/8a/0db635b225d2aa2984e405dc14bd2b0c324a0c312ea1bc9d283f2b83b038/numpy-2.0.1.tar.gz";
        hash="sha256-SFuHI1eWQQw1GaaZz+H6qwl+UJ6Q67BdzQmNsq6H57M=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 meson_1_5_0 meson-python_0_16_0 packaging pyproject-metadata_0_8_0];
      doCheck = false;
    };
    pybind11_2_12_0 = buildPythonPackage {
      pname = "pybind11";
      version = "2.12.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/26/55/e776489172f576b782e616f58273e1f3de56a91004b0d20504169dd345af/pybind11-2.12.0-py3-none-any.whl";
        hash="sha256-341guU+ecU2BAT2yMzk9Qw6/nzVRZCuCKRzxsU0a/b0=";
      };
      doCheck = false;
    };
    py-cpuinfo_9_0_0 = buildPythonPackage {
      pname = "py-cpuinfo";
      version = "9.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e0/a9/023730ba63db1e494a271cb018dcd361bd2c917ba7004c3e49d5daf795a2/py_cpuinfo-9.0.0-py3-none-any.whl";
        hash="sha256-hZYlvCUfZOIfB30JnUFiaJx2K11qTDyXVT1WJByWdNU=";
      };
      doCheck = false;
    };
    hatch-nodejs-version_0_3_2 = buildPythonPackage {
      pname = "hatch-nodejs-version";
      version = "0.3.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/fe/b23e9bca77cafecd1a10450066a1a4ca329149ad36aa86cdf8e67c2d2fa5/hatch_nodejs_version-0.3.2-py3-none-any.whl";
        hash="sha256-1z5yjxomLSFK/pwKQOFhAT7wt6bHj/hDKTiA9qRu3nk=";
      };
      doCheck = false;
    };
    entrypoints_0_4 = buildPythonPackage {
      pname = "entrypoints";
      version = "0.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/35/a8/365059bbcd4572cbc41de17fd5b682be5868b218c3c5479071865cab9078/entrypoints-0.4-py3-none-any.whl";
        hash="sha256-8XS1/4J1BP082XzD+GSfNpP1FTjH5L3z7wAshCnUL58=";
      };
      doCheck = false;
    };
    jupyter-client_7_4_9 = buildPythonPackage {
      pname = "jupyter-client";
      version = "7.4.9";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fd/a7/ef3b7c8b9d6730a21febdd0809084e4cea6d2a7e43892436adecdd0acbd4/jupyter_client-7.4.9-py3-none-any.whl";
        hash="sha256-IUZoquoggZX0wT0o6ycrp5+UX8DPPxHHCSwgssoZgOc=";
      };
      doCheck = false;
    };
    notebook_6_5_7 = buildPythonPackage {
      pname = "notebook";
      version = "6.5.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/af/9c/0620631da9d7013e95a8f985043cad229a0d8fb537a7e3f8ff8467565a8c/notebook-6.5.7-py3-none-any.whl";
        hash="sha256-pq+ppP9NFJoHcf+LjIgaenOzg1+a3QYGaW1unZisHNA=";
      };
      doCheck = false;
    };
    bleach_6_1_0 = buildPythonPackage {
      pname = "bleach";
      version = "6.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ea/63/da7237f805089ecc28a3f36bca6a21c31fcbc2eb380f3b8f1be3312abd14/bleach-6.1.0-py3-none-any.whl";
        hash="sha256-MiXzVM/ENrl4nGbE7gMBlL7gVo+/nL2tO8i1wmxfErY=";
      };
      doCheck = false;
    };
    nbclient_0_10_0 = buildPythonPackage {
      pname = "nbclient";
      version = "0.10.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/66/e8/00517a23d3eeaed0513e718fbc94aab26eaa1758f5690fc8578839791c79/nbclient-0.10.0-py3-none-any.whl";
        hash="sha256-8T41KTMqHx+B2CpTIQMiR2oWi7cJCgKJx5X+nMEcnT8=";
      };
      doCheck = false;
    };
    mistune_3_0_2 = buildPythonPackage {
      pname = "mistune";
      version = "3.0.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f0/74/c95adcdf032956d9ef6c89a9b8a5152bf73915f8c633f3e3d88d06bd699c/mistune-3.0.2-py3-none-any.whl";
        hash="sha256-cUgYVMMP28k4lj02BbclAfXBCpMg7NQSwSHBY6HH0gU=";
      };
      doCheck = false;
    };
    pandocfilters_1_5_1 = buildPythonPackage {
      pname = "pandocfilters";
      version = "1.5.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ef/af/4fbc8cab944db5d21b7e2a5b8e9211a03a79852b1157e2c102fcc61ac440/pandocfilters-1.5.1-py2.py3-none-any.whl";
        hash="sha256-k744KASpzbCnJnWF8Vfl0XMbvlVFqFsmjW9f5iMt4rw=";
      };
      doCheck = false;
    };
    webencodings_0_5_1 = buildPythonPackage {
      pname = "webencodings";
      version = "0.5.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f4/24/2a3e3df732393fed8b3ebf2ec078f05546de641fe1b667ee316ec1dcf3b7/webencodings-0.5.1-py2.py3-none-any.whl";
        hash="sha256-oK8SE/PCImSXqX4rOqAafkvuT0A/lb4W/JrNKUdRSng=";
      };
      doCheck = false;
    };
    tinycss2_1_3_0 = buildPythonPackage {
      pname = "tinycss2";
      version = "1.3.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/2c/4d/0db5b8a613d2a59bbc29bc5bb44a2f8070eb9ceab11c50d477502a8a0092/tinycss2-1.3.0-py3-none-any.whl";
        hash="sha256-VKjb3/szTVNoUb4CJgMOlQWWW7LzDyGkqCxV+yqA+uc=";
      };
      doCheck = false;
    };
    jupyterlab-pygments_0_3_0 = buildPythonPackage {
      pname = "jupyterlab-pygments";
      version = "0.3.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b1/dd/ead9d8ea85bf202d90cc513b533f9c363121c7792674f78e0d8a854b63b4/jupyterlab_pygments-0.3.0-py3-none-any.whl";
        hash="sha256-hBqJAglx2h2Gk/GpmZeu/F3EJLsbJR/WMiRiobiEJ4A=";
      };
      doCheck = false;
    };
    defusedxml_0_7_1 = buildPythonPackage {
      pname = "defusedxml";
      version = "0.7.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/07/6c/aa3f2f849e01cb6a001cd8554a88d4c77c5c1a31c95bdf1cf9301e6d9ef4/defusedxml-0.7.1-py2.py3-none-any.whl";
        hash="sha256-o1Ln5Ch3AobMiZ4lQrbNrtsrSVP/JpohAQPsWPYZimE=";
      };
      doCheck = false;
    };
    nbconvert_7_16_4 = buildPythonPackage {
      pname = "nbconvert";
      version = "7.16.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b8/bb/bb5b6a515d1584aa2fd89965b11db6632e4bdc69495a52374bcc36e56cfa/nbconvert-7.16.4-py3-none-any.whl";
        hash="sha256-BYc8Yg/lILYyK/ilrVYmkjQ/40UqvaV2XHo0t9GqPrM=";
      };
      doCheck = false;
    };
    pycparser_2_22 = buildPythonPackage {
      pname = "pycparser";
      version = "2.22";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/13/a3/a812df4e2dd5696d1f351d58b8fe16a405b234ad2886a0dab9183fb78109/pycparser-2.22-py3-none-any.whl";
        hash="sha256-w3ArbT3Yx6vBr6Vl1+Y9U6HQvYbNwk7ddUcPTeSZz8w=";
      };
      doCheck = false;
    };
    cffi_1_16_0 = buildPythonPackage {
      pname = "cffi";
      version = "1.16.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b4/5f/c6e7e8d80fbf727909e4b1b5b9352082fc1604a14991b1d536bfaee5a36c/cffi-1.16.0-cp312-cp312-manylinux_2_17_ppc64le.manylinux2014_ppc64le.whl";
        hash="sha256-/MjrbVkCuxz23E8YfuPqgKHroKiaukClyyClCH2WE1c=";
      };
      doCheck = false;
    };
    terminado_0_18_1 = buildPythonPackage {
      pname = "terminado";
      version = "0.18.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6a/9e/2064975477fdc887e47ad42157e214526dcad8f317a948dee17e1659a62f/terminado-0.18.1-py3-none-any.whl";
        hash="sha256-pEaOGze7MY+KhlFPZYFOGvyXfPKbOZKkUA2d0wXczrA=";
      };
      doCheck = false;
    };
    sniffio_1_3_1 = buildPythonPackage {
      pname = "sniffio";
      version = "1.3.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e9/44/75a9c9421471a6c4805dbf2356f7c181a29c1879239abab1ea2cc8f38b40/sniffio-1.3.1-py3-none-any.whl";
        hash="sha256-L22kGNHx4P3dhER49BaA55TmBRkVeRoDT/ZeXxAFJaI=";
      };
      doCheck = false;
    };
    anyio_4_4_0 = buildPythonPackage {
      pname = "anyio";
      version = "4.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7b/a2/10639a79341f6c019dedc95bd48a4928eed9f1d1197f4c04f546fc7ae0ff/anyio-4.4.0-py3-none-any.whl";
        hash="sha256-wbLY9GqKgSUTAS4RB8sOaMFxWaellCCABaV9x3bhvcc=";
      };
      doCheck = false;
    };
    overrides_7_7_0 = buildPythonPackage {
      pname = "overrides";
      version = "7.7.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/2c/ab/fc8290c6a4c722e5514d80f62b2dc4c4df1a68a41d1364e625c35990fcf3/overrides-7.7.0-py3-none-any.whl";
        hash="sha256-x+2dBi94uOTBp7cL2HlrNerU2fUQIn75xdx2JsYNfkk=";
      };
      doCheck = false;
    };
    prometheus-client_0_20_0 = buildPythonPackage {
      pname = "prometheus-client";
      version = "0.20.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c7/98/745b810d822103adca2df8decd4c0bbe839ba7ad3511af3f0d09692fc0f0/prometheus_client-0.20.0-py3-none-any.whl";
        hash="sha256-zeUkqFvOg8o1nMg38ouMDbXKx6plOliP1+hLoGHDKec=";
      };
      doCheck = false;
    };
    jupyter-server-terminals_0_5_3 = buildPythonPackage {
      pname = "jupyter-server-terminals";
      version = "0.5.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/07/2d/2b32cdbe8d2a602f697a649798554e4f072115438e92249624e532e8aca6/jupyter_server_terminals-0.5.3-py3-none-any.whl";
        hash="sha256-Qe4NfcDr8oCcZo4PxybfryWPzT52lWiZbKcxthlK6ao=";
      };
      doCheck = false;
    };
    argon2-cffi-bindings_21_2_0 = buildPythonPackage {
      pname = "argon2-cffi-bindings";
      version = "21.2.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b9/e9/184b8ccce6683b0aa2fbb7ba5683ea4b9c5763f1356347f1312c32e3c66e/argon2-cffi-bindings-21.2.0.tar.gz";
        hash="sha256-u4nO/6bHkYB9EwXOt32/rMWqSZiR0sVWYcZFllH8OeM=";
      };
      build-system = with packages;
      with buildPackages;
      [cffi_1_16_0 packaging pycparser_2_22 setuptools setuptools-scm_8_1_0 wheel_0_43_0];
      doCheck = false;
    };
    argon2-cffi_23_1_0 = buildPythonPackage {
      pname = "argon2-cffi";
      version = "23.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a4/6a/e8a041599e78b6b3752da48000b14c8d1e8a04ded09c88c714ba047f34f5/argon2_cffi-23.1.0-py3-none-any.whl";
        hash="sha256-xnBkK3i6KWQYGKsuaL1Oani6U7fv97TDgVrhar+Rx+o=";
      };
      doCheck = false;
    };
    websocket-client_1_8_0 = buildPythonPackage {
      pname = "websocket-client";
      version = "1.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5a/84/44687a29792a70e111c5c477230a72c4b957d88d16141199bf9acb7537a3/websocket_client-1.8.0-py3-none-any.whl";
        hash="sha256-F7RMyZf1xJjoCbIs3y2cep5xwCyMwrbFbnwtEjm/pSY=";
      };
      doCheck = false;
    };
    send2trash_1_8_3 = buildPythonPackage {
      pname = "send2trash";
      version = "1.8.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/40/b0/4562db6223154aa4e22f939003cb92514c79f3d4dccca3444253fd17f902/Send2Trash-1.8.3-py3-none-any.whl";
        hash="sha256-DDEifgvQiWHHZlR0o9HvcZOSn+3aQjOENom6oFa+Rsk=";
      };
      doCheck = false;
    };
    jupyter-server_2_14_2 = buildPythonPackage {
      pname = "jupyter-server";
      version = "2.14.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/57/e1/085edea6187a127ca8ea053eb01f4e1792d778b4d192c74d32eb6730fed6/jupyter_server-2.14.2-py3-none-any.whl";
        hash="sha256-R/9QYSfC94UaF79HE0NCCPxJCVXQ6GMulQFKmpr77v0=";
      };
      doCheck = false;
    };
    json5_0_9_25 = buildPythonPackage {
      pname = "json5";
      version = "0.9.25";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/8a/3c/4f8791ee53ab9eeb0b022205aa79387119a74cc9429582ce04098e6fc540/json5-0.9.25-py3-none-any.whl";
        hash="sha256-NO19g0sTQahph+1S8/ds2O4YQ5SQa24ioeDeuaspTo8=";
      };
      doCheck = false;
    };
    babel_2_15_0 = buildPythonPackage {
      pname = "babel";
      version = "2.15.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/27/45/377f7e32a5c93d94cd56542349b34efab5ca3f9e2fd5a68c5e93169aa32d/Babel-2.15.0-py3-none-any.whl";
        hash="sha256-CHBr2tjQo0EyZqthvWw00MKNbh57rfQKLOvmdkTi4fs=";
      };
      doCheck = false;
    };
    markupsafe_2_1_5 = buildPythonPackage {
      pname = "markupsafe";
      version = "2.1.5";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/87/5b/aae44c6655f3801e81aa3eef09dbbf012431987ba564d7231722f68df02d/MarkupSafe-2.1.5.tar.gz";
        hash="sha256-0oPTeokLpMGuc/+t+ARkNcdue8Ike7tjwAvRpwnGVEs=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools];
      doCheck = false;
    };
    jinja2_3_1_4 = buildPythonPackage {
      pname = "jinja2";
      version = "3.1.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/31/80/3a54838c3fb461f6fec263ebf3a3a41771bd05190238de3486aae8540c36/jinja2-3.1.4-py3-none-any.whl";
        hash="sha256-vF3Sq7cnpTGVZ7eoE+ai5zGMOfT0h8/myJxvnH0lGX0=";
      };
      doCheck = false;
    };
    jupyterlab-server_2_27_3 = buildPythonPackage {
      pname = "jupyterlab-server";
      version = "2.27.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/54/09/2032e7d15c544a0e3cd831c51d77a8ca57f7555b2e1b2922142eddb02a84/jupyterlab_server-2.27.3-py3-none-any.whl";
        hash="sha256-5pdIj2bD20nfZ1FYp3s7AXUg13LG4VSMfZvMXfeUTuQ=";
      };
      doCheck = false;
    };
    pyzmq_26_0_3 = buildPythonPackage {
      pname = "pyzmq";
      version = "26.0.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c4/9a/0e2ab500fd5a5a41e7d003e4a49faa7a0333db13e54498a3cf749b9eedd0/pyzmq-26.0.3.tar.gz";
        hash="sha256-26fZ8uBH36K8o7AfT4SqUkZyUgPWKE43kPLKFfumtAo=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 packaging pathspec_0_12_1 scikit-build-core_0_9_8];
      doCheck = false;
    };
    scikit-build-core_0_9_8 = buildPythonPackage {
      pname = "scikit-build-core";
      version = "0.9.8";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/0e/b7/62ede14d44d448bbb7424d5992e394d6980824312de05c9b4816a41602f0/scikit_build_core-0.9.8-py3-none-any.whl";
        hash="sha256-5uzF/Vi2qOv+oOns6qoqaAsVq/WS78k1ITrBXkOah8Y=";
      };
      doCheck = false;
    };
    python-json-logger_2_0_7 = buildPythonPackage {
      pname = "python-json-logger";
      version = "2.0.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/35/a6/145655273568ee78a581e734cf35beb9e33a370b29c5d3c8fee3744de29f/python_json_logger-2.0.7-py3-none-any.whl";
        hash="sha256-84C4JqmR67495NiXruxCdgA1rHYDReV7gSk43Is14r0=";
      };
      doCheck = false;
    };
    rfc3986-validator_0_1_1 = buildPythonPackage {
      pname = "rfc3986-validator";
      version = "0.1.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9e/51/17023c0f8f1869d8806b979a2bffa3f861f26a3f1a66b094288323fba52f/rfc3986_validator-0.1.1-py2.py3-none-any.whl";
        hash="sha256-LyNcQy70WZcLQwY2kza51dvdoxtRDKHjJ2NuAfUov6k=";
      };
      doCheck = false;
    };
    rfc3339-validator_0_1_4 = buildPythonPackage {
      pname = "rfc3339-validator";
      version = "0.1.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7b/44/4e421b96b67b2daff264473f7465db72fbdf36a07e05494f50300cc7b0c6/rfc3339_validator-0.1.4-py2.py3-none-any.whl";
        hash="sha256-JPbsHtoU74I9qeNuxxExJLOcBNUKTT06PChZV353kfo=";
      };
      doCheck = false;
    };
    jupyter-events_0_10_0 = buildPythonPackage {
      pname = "jupyter-events";
      version = "0.10.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a5/94/059180ea70a9a326e1815176b2370da56376da347a796f8c4f0b830208ef/jupyter_events-0.10.0-py3-none-any.whl";
        hash="sha256-S3ITCHXlnVdxbTJ+pw0+vDrxlE03F+WkmLigbGwVmWA=";
      };
      doCheck = false;
    };
    parso_0_8_4 = buildPythonPackage {
      pname = "parso";
      version = "0.8.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c6/ac/dac4a63f978e4dcb3c6d3a78c4d8e0192a113d288502a1216950c41b1027/parso-0.8.4-py2.py3-none-any.whl";
        hash="sha256-pBhnCiApHazS3dyAw3fFw3kTeO4ejRK//DVCBkPUPxg=";
      };
      doCheck = false;
    };
    jedi_0_19_1 = buildPythonPackage {
      pname = "jedi";
      version = "0.19.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/20/9f/bc63f0f0737ad7a60800bfd472a4836661adae21f9c2535f3957b1e54ceb/jedi-0.19.1-py2.py3-none-any.whl";
        hash="sha256-6YPGVP5cAoZ670zfzlovu0pQrcCvFF9wUEI48Y715+A=";
      };
      doCheck = false;
    };
    debugpy_1_8_2 = buildPythonPackage {
      pname = "debugpy";
      version = "1.8.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b4/32/dd0707c8557f99496811763c5333ea87bcec1eb233c1efa324c9a8082bff/debugpy-1.8.2-py2.py3-none-any.whl";
        hash="sha256-FuFt86mKNcY8OrHk0Zvky8f92pLZ3cBZKU8YkQko4Mo=";
      };
      doCheck = false;
    };
    tornado_6_4_1 = buildPythonPackage {
      pname = "tornado";
      version = "6.4.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ee/66/398ac7167f1c7835406888a386f6d0d26ee5dbf197d8a571300be57662d3/tornado-6.4.1.tar.gz";
        hash="sha256-ktOrUxg9jFD4IEpR5vkdGKFdXvJh6E1FKADU/2/FBOk=";
      };
      build-system = with packages;
      with buildPackages;
      [setuptools wheel_0_43_0];
      doCheck = false;
    };
    comm_0_2_2 = buildPythonPackage {
      pname = "comm";
      version = "0.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e6/75/49e5bfe642f71f272236b5b2d2691cf915a7283cc0ceda56357b61daa538/comm-0.2.2-py3-none-any.whl";
        hash="sha256-5vuGy3D/Zh7oycFOfTbW3jtAZvFEG+QGPfnFAJ8KZNM=";
      };
      doCheck = false;
    };
    wcwidth_0_2_13 = buildPythonPackage {
      pname = "wcwidth";
      version = "0.2.13";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fd/84/fd2ba7aafacbad3c4201d395674fc6348826569da3c0937e75505ead3528/wcwidth-0.2.13-py2.py3-none-any.whl";
        hash="sha256-PaaQSORUDYSvMhMYKf+Ujx4CLBxr241hAhF6rHhPaFk=";
      };
      doCheck = false;
    };
    prompt-toolkit_3_0_47 = buildPythonPackage {
      pname = "prompt-toolkit";
      version = "3.0.47";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e8/23/22750c4b768f09386d1c3cc4337953e8936f48a888fa6dddfb669b2c9088/prompt_toolkit-3.0.47-py3-none-any.whl";
        hash="sha256-DXv6ZwAdXjnQLCJLZjq8M2h0BQM6jEItDWdaWhM2HRA=";
      };
      doCheck = false;
    };
    ptyprocess_0_7_0 = buildPythonPackage {
      pname = "ptyprocess";
      version = "0.7.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/22/a6/858897256d0deac81a172289110f31629fc4cee19b6f01283303e18c8db3/ptyprocess-0.7.0-py2.py3-none-any.whl";
        hash="sha256-S0Hzln/OOvV8x+lLiIYmwYvzegg+NlHKj+62bUkv7zU=";
      };
      doCheck = false;
    };
    pexpect_4_9_0 = buildPythonPackage {
      pname = "pexpect";
      version = "4.9.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9e/c3/059298687310d527a58bb01f3b1965787ee3b40dce76752eda8b44e9a2c5/pexpect-4.9.0-py2.py3-none-any.whl";
        hash="sha256-cjbR4IDkk2vi3D4ybOwK9yrPkhKn4dBgIQ5wpH4lNSM=";
      };
      doCheck = false;
    };
    asttokens_2_4_1 = buildPythonPackage {
      pname = "asttokens";
      version = "2.4.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/45/86/4736ac618d82a20d87d2f92ae19441ebc7ac9e7a581d7e58bbe79233b24a/asttokens-2.4.1-py2.py3-none-any.whl";
        hash="sha256-BR7UnD3K6JE+p80I5Gpgbbowt5mTIJY2xIdbwdY3vCQ=";
      };
      doCheck = false;
    };
    jupyter-server-fileid_0_9_2 = buildPythonPackage {
      pname = "jupyter-server-fileid";
      version = "0.9.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3f/fd/bee81a24415f9b10d3a2408b7b64b895d2773b82cd82c8ea5a0f1cbd54e7/jupyter_server_fileid-0.9.2-py3-none-any.whl";
        hash="sha256-dqL7zqaVCWhIXc1QnC1qxBfKEeYasa1EekdfCHjKgI8=";
      };
      doCheck = false;
    };
    aiosqlite_0_20_0 = buildPythonPackage {
      pname = "aiosqlite";
      version = "0.20.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/00/c4/c93eb22025a2de6b83263dfe3d7df2e19138e345bca6f18dba7394120930/aiosqlite-0.20.0-py3-none-any.whl";
        hash="sha256-NqHerKDKxA6+MqrJl3puK7x/UYnyP0pU1ZCJhnKeW9Y=";
      };
      doCheck = false;
    };
    aiofiles_22_1_0 = buildPythonPackage {
      pname = "aiofiles";
      version = "22.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/48/d5d1ab7cfe46e573c3694fa1365442a7d7cadc3abb03d8507e58a3755bb2/aiofiles-22.1.0-py3-none-any.whl";
        hash="sha256-EUL6joDbrka7YzlXOtTIwIQTWPecbrUKST3OyhRiG60=";
      };
      doCheck = false;
    };
    ypy-websocket_0_8_4 = buildPythonPackage {
      pname = "ypy-websocket";
      version = "0.8.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/bd/f3/f6a8dfcae1716d260e6cf3ecea864a6abfddadd6a059bed80bd5618b67c1/ypy_websocket-0.8.4-py3-none-any.whl";
        hash="sha256-sboN/Ml2LwyhaNI3gGLTyhKZ05B2sPFF2WE1kSEEK+U=";
      };
      doCheck = false;
    };
    y-py_0_6_2 = buildPythonPackage {
      pname = "y-py";
      version = "0.6.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/4e/29/df7d9b506deff4158b80433c19294889951afe0cef911ab99dbbcf8704d5/y_py-0.6.2-cp312-cp312-manylinux_2_17_ppc64le.manylinux2014_ppc64le.whl";
        hash="sha256-Kkl+vmF77GpCD8RzeIVsquQKsGUudW8+1AxfH+KhIiA=";
      };
      doCheck = false;
    };
    executing_2_0_1 = buildPythonPackage {
      pname = "executing";
      version = "2.0.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/80/03/6ea8b1b2a5ab40a7a60dc464d3daa7aa546e0a74d74a9f8ff551ea7905db/executing-2.0.1-py2.py3-none-any.whl";
        hash="sha256-6sScqUUWzMdT+ftc6CYDFW5ZCydSWovDLM6K4wLrYbw=";
      };
      doCheck = false;
    };
    notebook-shim_0_2_4 = buildPythonPackage {
      pname = "notebook-shim";
      version = "0.2.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f9/33/bd5b9137445ea4b680023eb0469b2bb969d61303dedb2aac6560ff3d14a1/notebook_shim-0.2.4-py3-none-any.whl";
        hash="sha256-QRpb5OnciCoHTMvK5nHtpkzOsGh2fpo0GQlphlYOHO8=";
      };
      doCheck = false;
    };
    jupyter-ydoc_0_2_5 = buildPythonPackage {
      pname = "jupyter-ydoc";
      version = "0.2.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e1/36/66e6cb851a43c95f00f47b36d2cb3e17d37f862449dc8258b2c04f02544b/jupyter_ydoc-0.2.5-py3-none-any.whl";
        hash="sha256-V1kXDxEscDIKhCF92Y0odpkHauZaf4jUWNV5QKnyuII=";
      };
      doCheck = false;
    };
    pure-eval_0_2_3 = buildPythonPackage {
      pname = "pure-eval";
      version = "0.2.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/8e/37/efad0257dc6e593a18957422533ff0f87ede7c9c6ea010a2177d738fb82f/pure_eval-0.2.3-py3-none-any.whl";
        hash="sha256-HbjjW2ez0hjYGK5lPifwbDqkIJAfp7CBypjL7ch04NA=";
      };
      doCheck = false;
    };
    ipython-genutils_0_2_0 = buildPythonPackage {
      pname = "ipython-genutils";
      version = "0.2.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fa/bc/9bd3b5c2b4774d5f33b2d544f1460be9df7df2fe42f352135381c347c69a/ipython_genutils-0.2.0-py2.py3-none-any.whl";
        hash="sha256-ct03IzeZ5hlmbJ9jmp2oPDQBOnPou8eaemNI2Txh+rg=";
      };
      doCheck = false;
    };
    ipykernel_6_29_5 = buildPythonPackage {
      pname = "ipykernel";
      version = "6.29.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/94/5c/368ae6c01c7628438358e6d337c19b05425727fbb221d2a3c4303c372f42/ipykernel-6.29.5-py3-none-any.whl";
        hash="sha256-r9tmulqjVLCbkTebrCiuSv67sw6LOVEMlpCvt6EEIbU=";
      };
      doCheck = false;
    };
    stack-data_0_6_3 = buildPythonPackage {
      pname = "stack-data";
      version = "0.6.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f1/7b/ce1eafaf1a76852e2ec9b22edecf1daa58175c090266e9f6c64afcd81d91/stack_data-0.6.3-py3-none-any.whl";
        hash="sha256-1VWODCWkywhTzdrT132piRoIy4Xdn5+RufjNZuUR5pU=";
      };
      doCheck = false;
    };
    matplotlib-inline_0_1_7 = buildPythonPackage {
      pname = "matplotlib-inline";
      version = "0.1.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/8f/8e/9ad090d3553c280a8060fbf6e24dc1c0c29704ee7d1c372f0c174aa59285/matplotlib_inline-0.1.7-py3-none-any.whl";
        hash="sha256-3xktOaT/jyGxiV1y5qE/X8xQmfAPqEOE4OoowswGU8o=";
      };
      doCheck = false;
    };
    decorator_5_1_1 = buildPythonPackage {
      pname = "decorator";
      version = "5.1.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d5/50/83c593b07763e1161326b3b8c6686f0f4b0f24d5526546bee538c89837d6/decorator-5.1.1-py3-none-any.whl";
        hash="sha256-uMP4WQC53EIyJZE8WqzpRyn+H6l2OziTmpUibwLTcYY=";
      };
      doCheck = false;
    };
    nbclassic_1_1_0 = buildPythonPackage {
      pname = "nbclassic";
      version = "1.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/59/d1/86fcfe1d1e5b66ec61632b014612e2d074a9d2bdf0eed53b90c2536e8dd3/nbclassic-1.1.0-py3-none-any.whl";
        hash="sha256-jA/W424yChhlf/RO2Ww6QA8XqQOjdE/DIjA6UVd48ro=";
      };
      doCheck = false;
    };
    jupyter-server-ydoc_0_8_0 = buildPythonPackage {
      pname = "jupyter-server-ydoc";
      version = "0.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f2/e4/b9e9b31e5b0d91b4ef749195a19a02f565f6edfb3a845d8dd457031578e3/jupyter_server_ydoc-0.8.0-py3-none-any.whl";
        hash="sha256-lpo6GnftTplIfWCnQEjcn6fTsNzTLmCIXYNbv3unvhE=";
      };
      doCheck = false;
    };
    ipython_8_26_0 = buildPythonPackage {
      pname = "ipython";
      version = "8.26.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/73/48/4d2818054671bb272d1b12ca65748a4145dc602a463683b5c21b260becee/ipython-8.26.0-py3-none-any.whl";
        hash="sha256-5rNHwnvfnDLunTGuhd78UldVoYafFAV+kAZ1uejW5v8=";
      };
      doCheck = false;
    };
    jupyterlab_3_6_7 = buildPythonPackage {
      pname = "jupyterlab";
      version = "3.6.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/18/e8/0bd6fefac07f976c745a9fc037ddcfb501dde670a97230489ab550a2b966/jupyterlab-3.6.7-py3-none-any.whl";
        hash="sha256-2S1X1AL1OSK8pQkGVIQ6oI5REpDf8p/bCAnq+76235g=";
      };
      doCheck = false;
    };
    setuptools-scm_7_1_0 = buildPythonPackage {
      pname = "setuptools-scm";
      version = "7.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1d/66/8f42c941be949ef2b22fe905d850c794e7c170a526023612aad5f3a121ad/setuptools_scm-7.1.0-py3-none-any.whl";
        hash="sha256-c5iLbYSHCeKvFCqkjJhuopWSu8/KU3VngGRwggUlPY4=";
      };
      doCheck = false;
    };
    maturin_1_7_0 = buildPythonPackage {
      pname = "maturin";
      version = "1.7.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e1/5d/4ecbfa6acdfa178d1a54042d22cff8efdfafe1275f9e396722feb4b11792/maturin-1.7.0-py3-none-manylinux_2_17_ppc64le.manylinux2014_ppc64le.musllinux_1_1_ppc64le.whl";
        hash="sha256-KRh9XD4eFmwU6q3GOorcJba7s+WwVdG8h/bKkrS24zE=";
      };
      doCheck = false;
    };
    versioneer_0_29 = buildPythonPackage {
      pname = "versioneer";
      version = "0.29";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b0/79/f0f1ca286b78f6f33c521a36b5cbd5bd697c0d66217d8856f443aeb9dd77/versioneer-0.29-py3-none-any.whl";
        hash="sha256-DxoTe7XWgR6Wp5uwSGeYrq6bnG78JLOJZZzrsO45bLk=";
      };
      doCheck = false;
    };
    meson_1_2_1 = buildPythonPackage {
      pname = "meson";
      version = "1.2.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e5/74/a1f1c6ba14e11e0fb050d2c61a78b6db108dd38383b6c0ab51c1becbbeff/meson-1.2.1-py3-none-any.whl";
        hash="sha256-CPg/wXUT6ZzW6Cx1VMH1ivcEJSEYh/j5xzY7KpAglGI=";
      };
      doCheck = false;
    };
    meson-python_0_13_1 = buildPythonPackage {
      pname = "meson-python";
      version = "0.13.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9f/af/5f941f57dc516e72b018183a38fbcfb018a7e83afd3c756ecfba82f21c65/meson_python-0.13.1-py3-none-any.whl";
        hash="sha256-4z6j77rezBV2jCBdA7kFx7O/cq+uHh69hLQ4xKPtM5M=";
      };
      doCheck = false;
    };
    cython_3_0_5 = buildPythonPackage {
      pname = "cython";
      version = "3.0.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fb/fe/e213d8e9cb21775bb8f9c92ff97861504129e23e33d118be1a90ca26a13e/Cython-3.0.5-py2.py3-none-any.whl";
        hash="sha256-dSBjaVBPxELBCobs9XuRWS3KdE5Fkq8ipH6ad01T3RA=";
      };
      doCheck = false;
    };
    hatch-vcs_0_4_0 = buildPythonPackage {
      pname = "hatch-vcs";
      version = "0.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/82/0f/6cbd9976160bc334add63bc2e7a58b1433a31b34b7cda6c5de6dd983d9a7/hatch_vcs-0.4.0-py3-none-any.whl";
        hash="sha256-uKK2vuVM9vn8k3YttziQAXrlnJCB0QOKQfFiNc6viyw=";
      };
      doCheck = false;
    };
    cmake_3_30_1 = buildPythonPackage {
      pname = "cmake";
      version = "3.30.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/35/34/180972b77a17f21956778f0a23c8bd254ec64284e5d2d6961523189895c3/cmake-3.30.1-py3-none-manylinux_2_17_ppc64le.manylinux2014_ppc64le.whl";
        hash="sha256-iMVh4pr2oh+03ID5Q4dnr4ulCB0sWM/CoWKYB21zFTk=";
      };
      doCheck = false;
    };
    distro_1_9_0 = buildPythonPackage {
      pname = "distro";
      version = "1.9.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/12/b3/231ffd4ab1fc9d679809f356cebee130ac7daa00d6d6f3206dd4fd137e9e/distro-1.9.0-py3-none-any.whl";
        hash="sha256-e//ZJdZRaPhQJ9jamva92rZYE1uEBnCiI1ibwMjvArI=";
      };
      doCheck = false;
    };
    scikit-build_0_18_0 = buildPythonPackage {
      pname = "scikit-build";
      version = "0.18.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/24/43/a0b5837cf30db1561a04187edd262bdefaffcb61222cb441eadef35f9103/scikit_build-0.18.0-py3-none-any.whl";
        hash="sha256-6hcfVSnm4LW2Zhk0M4Ma9hoo1+35c7M4hOyMeCoV7jg=";
      };
      doCheck = false;
    };
    setuptools_59_2_0 = buildPythonPackage {
      pname = "setuptools";
      version = "59.2.0";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a8/e7/1440b0d19054a5616e9e5beeaa22f68485aa9de20d187f04e52880b7ae7a/setuptools-59.2.0.tar.gz";
        hash="sha256-FX0h3p0FWrno6jGG2R5/T4ZeEfQt6vqVLZCEJnH8JXY=";
      };
      build-system = with packages;
      with buildPackages;
      [];
      doCheck = false;
    };
    pip_24_1_2 = buildPythonPackage {
      pname = "pip";
      version = "24.1.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e7/54/0c1c068542cee73d8863336e974fc881e608d0170f3af15d0c0f28644531/pip-24.1.2-py3-none-any.whl";
        hash="sha256-fNIH7tTGCw9BG0RM0UZBmP4YZnHDI7bNbUM+2A/J0kc=";
      };
      doCheck = false;
    };
    poetry-core_1_9_0 = buildPythonPackage {
      pname = "poetry-core";
      version = "1.9.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a1/8d/85fcf9bcbfefcc53a1402450f28e5acf39dcfde3aabb996a1d98481ac829/poetry_core-1.9.0-py3-none-any.whl";
        hash="sha256-TgycatjPiZVvA7MIc22E6m3bRAidFvKtyUBQEI7B9aE=";
      };
      doCheck = false;
    };
    cython_0_29_37 = buildPythonPackage {
      pname = "cython";
      version = "0.29.37";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/7e/26/9d8de10005fedb1eceabe713348d43bae1dbab1786042ca0751a2e2b0f8c/Cython-0.29.37-py2.py3-none-any.whl";
        hash="sha256-lfHWqD7ycp5ns/pzGMgpzlsHrGTAhM1q8RwijgNkZiw=";
      };
      doCheck = false;
    };
    setuptools_68_1_2 = buildPythonPackage {
      pname = "setuptools";
      version = "68.1.2";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/19/20/d8dd9d8becaf3e2d6fdc17cc41870d5ada5ceda518996cf5968c2ca71bd8/setuptools-68.1.2.tar.gz";
        hash="sha256-PU36bZXxsQHWlaYWCnYm4VWDr3Gl9SF276XTmgVNR10=";
      };
      build-system = with packages;
      with buildPackages;
      [];
      doCheck = false;
    };
    numpy_2_0_0 = buildPythonPackage {
      pname = "numpy";
      version = "2.0.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/05/35/fb1ada118002df3fe91b5c3b28bc0d90f879b881a5d8f68b1f9b79c44bfe/numpy-2.0.0.tar.gz";
        hash="sha256-z10cnmg3+K+fkra9PobVE83BH2D9YhhcxJ7H0aujSGQ=";
      };
      build-system = with packages;
      with buildPackages;
      [cython_3_0_10 meson_1_5_0 meson-python_0_16_0 packaging pyproject-metadata_0_8_0];
      doCheck = false;
    };
    pkgconfig_1_5_5 = buildPythonPackage {
      pname = "pkgconfig";
      version = "1.5.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/32/af/89487c7bbf433f4079044f3dc32f9a9f887597fe04614a37a292e373e16b/pkgconfig-1.5.5-py3-none-any.whl";
        hash="sha256-0gAju+tC7m1Cig+sbgkEYx9UWYWhDN1xogqli8R6Qgk=";
      };
      doCheck = false;
    };
    meson-python_0_15_0 = buildPythonPackage {
      pname = "meson-python";
      version = "0.15.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1f/60/b10b11ab470a690d5777310d6cfd1c9bdbbb0a1313a78c34a1e82e0b9d27/meson_python-0.15.0-py3-none-any.whl";
        hash="sha256-OuOCU/8CsulHoF42Ki6vWpoJ0TPFZmtBIzme5fvy5ZE=";
      };
      doCheck = false;
    };
    pdm-backend_2_3_3 = buildPythonPackage {
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
  env = with packages;
  [regex sentry-sdk fonttools urllib3 certifi charset-normalizer idna requests smmap gitdb gitpython rpds-py referencing jsonschema-specifications attrs jsonschema tqdm protobuf psutil soupsieve pyparsing traitlets jupyter-core pytz python-xlib typing-extensions tenacity fastjsonschema fsspec click absl-py llvmlite numba contourpy packaging kiwisolver scipy evdev numcodecs toolz tzdata nbformat huggingface-hub plotly cycler matplotlib optax nest-asyncio tensorstore orbax-checkpoint msgpack flax pluggy ffmpegio-core ffmpegio python-dateutil safetensors mdurl pandas tokenizers filelock transformers termcolor pynput pillow opencv-python robosuite shapely pyopengl glfw trimesh mujoco etils mujoco-mjx fasteners asciitree zarr einops chex pygments markdown-it-py rich setuptools setproctitle platformdirs docker-pycreds wandb six pyyaml contextlib2 ml-collections opt-einsum ml-dtypes jaxlib h5py sentencepiece trajax beautifulsoup4 numpy jax language-model image-classifier cond-diffusion stanza-models stanza stanza-meta];
}