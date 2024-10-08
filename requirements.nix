{buildPythonPackage, fetchurl, nixpkgs, python, nixpy-custom ? {}}: rec {
  packages = rec {
    foundry-meta = {
      aarch64-darwin = buildPythonPackage {
        pname = "foundry-meta";
        version = "0.1.0";
        format="pyproject";
        src = ./.;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d25a40d24 foundry-models.dde5f89be policy-eval image-classifier language-model image-diffusion cond-diffusion-toy pytest];
        doCheck=false;
      } ;
      powerpc64le-linux = buildPythonPackage {
        pname = "foundry-meta";
        version = "0.1.0";
        format="pyproject";
        src = ./.;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d7e0cabb6 foundry-models.d66e1ada3 policy-eval image-classifier language-model image-diffusion cond-diffusion-toy pytest];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "foundry-meta";
        version = "0.1.0";
        format="pyproject";
        src = ./.;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d812c7c17 foundry-models.d2a9263c1 policy-eval image-classifier language-model image-diffusion cond-diffusion-toy pytest];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    foundry-core = {
      with_ipython = {
        powerpc64le-linux = buildPythonPackage {
          pname = "foundry-core";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/core;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d1f799c17 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py wandb boto3];
          doCheck=false;
        } ;
        x86_64-linux = buildPythonPackage {
          pname = "foundry-core";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/core;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d912aa156 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py wandb boto3];
          doCheck=false;
        } ;
        aarch64-darwin = buildPythonPackage {
          pname = "foundry-core";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/core;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d7821d309 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py wandb boto3];
          doCheck=false;
        } ;
      }.${
        nixpkgs.system
      };
      default = {
        x86_64-linux = buildPythonPackage {
          pname = "foundry-core";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/core;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d912aa156 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py wandb boto3];
          doCheck=false;
        } ;
        aarch64-darwin = buildPythonPackage {
          pname = "foundry-core";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/core;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d7821d309 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py wandb boto3];
          doCheck=false;
        } ;
        powerpc64le-linux = buildPythonPackage {
          pname = "foundry-core";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/core;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d1f799c17 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py wandb boto3];
          doCheck=false;
        } ;
      }.${
        nixpkgs.system
      };
    };
    foundry-systems = {
      aarch64-darwin = {
        d46fbf688 = buildPythonPackage {
          pname = "foundry-systems";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/systems;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.default];
          doCheck=false;
        } ;
        d25a40d24 = buildPythonPackage {
          pname = "foundry-systems";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/systems;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.with_ipython];
          doCheck=false;
        } ;
      };
      x86_64-linux = {
        df32716f4 = buildPythonPackage {
          pname = "foundry-systems";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/systems;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.default];
          doCheck=false;
        } ;
        d812c7c17 = buildPythonPackage {
          pname = "foundry-systems";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/systems;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.with_ipython];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        ddc581d2a = buildPythonPackage {
          pname = "foundry-systems";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/systems;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.default];
          doCheck=false;
        } ;
        d7e0cabb6 = buildPythonPackage {
          pname = "foundry-systems";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/systems;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.with_ipython];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    foundry-models = {
      x86_64-linux = {
        d961e93be = buildPythonPackage {
          pname = "foundry-models";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.default];
          doCheck=false;
        } ;
        d2a9263c1 = buildPythonPackage {
          pname = "foundry-models";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.with_ipython];
          doCheck=false;
        } ;
      };
      aarch64-darwin = {
        dfd2a6824 = buildPythonPackage {
          pname = "foundry-models";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.default];
          doCheck=false;
        } ;
        dde5f89be = buildPythonPackage {
          pname = "foundry-models";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.with_ipython];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        d993baadc = buildPythonPackage {
          pname = "foundry-models";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.default];
          doCheck=false;
        } ;
        d66e1ada3 = buildPythonPackage {
          pname = "foundry-models";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [foundry-core.with_ipython];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    policy-eval = {
      powerpc64le-linux = buildPythonPackage {
        pname = "policy-eval";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/policy-eval;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d7e0cabb6 foundry-models.d66e1ada3 omegaconf wandb];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "policy-eval";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/policy-eval;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d25a40d24 foundry-models.dde5f89be omegaconf wandb];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "policy-eval";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/policy-eval;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d812c7c17 foundry-models.d2a9263c1 omegaconf wandb];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    image-classifier = {
      powerpc64le-linux = buildPythonPackage {
        pname = "image-classifier";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-classifier;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-models.d66e1ada3];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "image-classifier";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-classifier;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-models.dde5f89be];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "image-classifier";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-classifier;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-models.d2a9263c1];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    language-model = {
      powerpc64le-linux = buildPythonPackage {
        pname = "language-model";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/language-model;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-models.d66e1ada3];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "language-model";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/language-model;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-models.dde5f89be];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "language-model";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/language-model;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-models.d2a9263c1];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    image-diffusion = {
      x86_64-linux = buildPythonPackage {
        pname = "image-diffusion";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-diffusion;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d812c7c17 foundry-models.d2a9263c1 omegaconf wandb];
        doCheck=false;
      } ;
      powerpc64le-linux = buildPythonPackage {
        pname = "image-diffusion";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-diffusion;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d7e0cabb6 foundry-models.d66e1ada3 omegaconf wandb];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "image-diffusion";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-diffusion;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [foundry-core.with_ipython foundry-systems.d25a40d24 foundry-models.dde5f89be omegaconf wandb];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    cond-diffusion-toy = buildPythonPackage {
      pname = "cond-diffusion-toy";
      version = "0.1.0";
      format="pyproject";
      src = ./projects/cond-diffusion-toy;
      build-system = with packages;
      [pdm-backend];
      dependencies = with packages;
      [foundry-core.with_ipython];
      doCheck=false;
    } ;
    jax = {
      aarch64-darwin = nixpy-custom.jax_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          scipy = scipy;
          opt-einsum = opt-einsum;
          ml-dtypes = ml-dtypes.d618fa062;
          jaxlib = jaxlib;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      powerpc64le-linux = nixpy-custom.jax_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          scipy = scipy;
          opt-einsum = opt-einsum;
          ml-dtypes = ml-dtypes.dd14ec181;
          jaxlib = jaxlib;
          jax-cuda12-plugin = jax-cuda12-plugin;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      x86_64-linux = nixpy-custom.jax_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          scipy = scipy;
          opt-einsum = opt-einsum;
          ml-dtypes = ml-dtypes.d513b6658;
          jaxlib = jaxlib;
          jax-cuda12-plugin = jax-cuda12-plugin;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
    };
    numpy = {
      v1_26_4 = nixpy-custom.numpy_1_26_4 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          cython = cython.v3_0_11;
          meson-python = meson-python.v0_15_0;
        };
        dependencies={
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      v2_0_1 = nixpy-custom.numpy_2_0_1 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          cython = cython.v3_0_11;
          meson-python = meson-python.v0_15_0;
        };
        dependencies={
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    };
    ffmpegio = nixpy-custom.ffmpegio_0_10_0_post0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        wheel = wheel;
      };
      dependencies=with packages;
      {
        ffmpegio-core = ffmpegio-core;
        numpy = numpy.v1_26_4;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    matplotlib = {
      powerpc64le-linux = nixpy-custom.matplotlib_3_9_1 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          meson-python = meson-python.v0_15_0;
          pybind11 = pybind11;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          contourpy = contourpy.d5c0bade7;
          cycler = cycler;
          fonttools = fonttools;
          kiwisolver = kiwisolver;
          numpy = numpy.v1_26_4;
          packaging = packaging;
          pillow = pillow;
          pyparsing = pyparsing;
          python-dateutil = python-dateutil;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      aarch64-darwin = nixpy-custom.matplotlib_3_9_1 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          meson-python = meson-python.v0_15_0;
          pybind11 = pybind11;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          contourpy = contourpy.dda94ea09;
          cycler = cycler;
          fonttools = fonttools;
          kiwisolver = kiwisolver;
          numpy = numpy.v1_26_4;
          packaging = packaging;
          pillow = pillow;
          pyparsing = pyparsing;
          python-dateutil = python-dateutil;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      x86_64-linux = nixpy-custom.matplotlib_3_9_1 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          meson-python = meson-python.v0_15_0;
          pybind11 = pybind11;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          contourpy = contourpy.d12135c46;
          cycler = cycler;
          fonttools = fonttools;
          kiwisolver = kiwisolver;
          numpy = numpy.v1_26_4;
          packaging = packaging;
          pillow = pillow;
          pyparsing = pyparsing;
          python-dateutil = python-dateutil;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
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
      [soupsieve];
      doCheck=false;
    } ;
    trajax = buildPythonPackage {
      pname = "trajax";
      version = "0.0.1";
      src = fetchurl {
        url="https://github.com/google/trajax/archive/c94a637c5a397b3d4100153f25b4b165507b5b20.tar.gz";
        hash="sha256-xN/LSQI/zvf367Ba9MFRIzpP/AmFbAOT1M1ShuW75pI=";
      };
      build-system = with packages;
      [setuptools.v74_1_1];
      dependencies = with packages;
      [absl-py jax jaxlib ml-collections scipy];
      doCheck=false;
    } ;
    shapely = nixpy-custom.shapely_2_0_5 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        cython = cython.v3_0_11;
        numpy = numpy.v1_26_4;
      };
      dependencies=with packages;
      {
        numpy = numpy.v1_26_4;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    robosuite = nixpy-custom.robosuite_1_4_1 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies=with packages;
      {
        numpy = numpy.v1_26_4;
        numba = numba;
        scipy = scipy;
        mujoco = mujoco;
        pillow = pillow;
        pynput = pynput;
        termcolor = termcolor;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    sentencepiece = nixpy-custom.sentencepiece_0_2_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    h5py = nixpy-custom.h5py_3_11_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        cython = cython.v3_0_11;
        numpy = numpy.v1_26_4;
        pkgconfig = pkgconfig;
        setuptools = setuptools.v74_1_1;
      };
      dependencies=with packages;
      {
        numpy = numpy.v1_26_4;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    wandb = nixpy-custom.wandb_0_17_9 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        hatchling = hatchling;
        typing-extensions = typing-extensions;
      };
      dependencies=with packages;
      {
        click = click;
        gitpython = gitpython;
        requests = requests;
        psutil = psutil;
        sentry-sdk = sentry-sdk;
        docker-pycreds = docker-pycreds;
        setuptools = setuptools.v74_1_1;
        protobuf = protobuf;
        pyyaml = pyyaml;
        setproctitle = setproctitle;
        platformdirs = platformdirs;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    boto3 = buildPythonPackage {
      pname = "boto3";
      version = "1.35.11";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/db/21/7535e4e04ac938c571a5ff3717a98142364f0c6c78ffd5723a5c8efce920/boto3-1.35.11-py3-none-any.whl";
        hash="sha256-9YNN2Qjt2lbD2oa5CGk8fNHBfC+BUN5zbp6QxW7MePY=";
      };
      dependencies = with packages;
      [botocore jmespath s3transfer];
      doCheck=false;
    } ;
    omegaconf = buildPythonPackage {
      pname = "omegaconf";
      version = "2.3.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e3/94/1843518e420fa3ed6919835845df698c7e27e183cb997394e4a670973a65/omegaconf-2.3.0-py3-none-any.whl";
        hash="sha256-e03xdc2wi6QA9FyuO9yue6g2XbTRZfxl/QSwUKtjtGs=";
      };
      dependencies = with packages;
      [antlr4-python3-runtime pyyaml];
      doCheck=false;
    } ;
    scipy = nixpy-custom.scipy_1_14_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        meson-python = meson-python.v0_15_0;
        cython = cython.v3_0_11;
        pybind11 = pybind11;
        pythran = pythran;
        numpy = numpy.v1_26_4;
        wheel = wheel;
        setuptools = setuptools.v74_1_1;
      };
      dependencies=with packages;
      {
        numpy = numpy.v1_26_4;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
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
      [numpy.v1_26_4];
      doCheck=false;
    } ;
    ml-dtypes = {
      aarch64-darwin = {
        d618fa062 = nixpy-custom.ml_dtypes_0_4_0 {
          buildPythonPackage=buildPythonPackage;
          build-system=with packages;
          {
            numpy = numpy.v2_0_1;
            setuptools = setuptools.v70_1_1;
          };
          dependencies=with packages;
          {
            numpy = numpy.v1_26_4;
          };
          fetchurl=fetchurl;
          nixpkgs=nixpkgs;
          python=python;
        };
        d7c40c4ed = nixpy-custom.ml_dtypes_0_4_0 {
          buildPythonPackage=buildPythonPackage;
          build-system=with packages;
          {
            numpy = numpy.v2_0_1;
            setuptools = setuptools.v70_1_1;
          };
          dependencies=with packages;
          {
            numpy = numpy.v2_0_1;
          };
          fetchurl=fetchurl;
          nixpkgs=nixpkgs;
          python=python;
        };
      };
      x86_64-linux = {
        dc81db21a = nixpy-custom.ml_dtypes_0_4_0 {
          buildPythonPackage=buildPythonPackage;
          build-system=with packages;
          {
            numpy = numpy.v2_0_1;
            setuptools = setuptools.v70_1_1;
          };
          dependencies=with packages;
          {
            numpy = numpy.v2_0_1;
          };
          fetchurl=fetchurl;
          nixpkgs=nixpkgs;
          python=python;
        };
        d513b6658 = nixpy-custom.ml_dtypes_0_4_0 {
          buildPythonPackage=buildPythonPackage;
          build-system=with packages;
          {
            numpy = numpy.v2_0_1;
            setuptools = setuptools.v70_1_1;
          };
          dependencies=with packages;
          {
            numpy = numpy.v1_26_4;
          };
          fetchurl=fetchurl;
          nixpkgs=nixpkgs;
          python=python;
        };
      };
      powerpc64le-linux = {
        d3b8ca945 = nixpy-custom.ml_dtypes_0_4_0 {
          buildPythonPackage=buildPythonPackage;
          build-system=with packages;
          {
            numpy = numpy.v2_0_1;
            setuptools = setuptools.v70_1_1;
          };
          dependencies=with packages;
          {
            numpy = numpy.v2_0_1;
          };
          fetchurl=fetchurl;
          nixpkgs=nixpkgs;
          python=python;
        };
        dd14ec181 = nixpy-custom.ml_dtypes_0_4_0 {
          buildPythonPackage=buildPythonPackage;
          build-system=with packages;
          {
            numpy = numpy.v2_0_1;
            setuptools = setuptools.v70_1_1;
          };
          dependencies=with packages;
          {
            numpy = numpy.v1_26_4;
          };
          fetchurl=fetchurl;
          nixpkgs=nixpkgs;
          python=python;
        };
      };
    }.${
      nixpkgs.system
    };
    jaxlib = {
      x86_64-linux = nixpy-custom.jaxlib_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d513b6658;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      aarch64-darwin = nixpy-custom.jaxlib_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d618fa062;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      powerpc64le-linux = nixpy-custom.jaxlib_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.dd14ec181;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
    };
    jax-cuda12-plugin = {
      powerpc64le-linux = nixpy-custom.jax_cuda12_plugin_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          jax-cuda12-prjt = jax-cuda12-prjt;
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.dd14ec181;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      x86_64-linux = nixpy-custom.jax_cuda12_plugin_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          jax-cuda12-prjt = jax-cuda12-prjt;
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d513b6658;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
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
      [packaging pluggy];
      doCheck=false;
    } ;
    kiwisolver = nixpy-custom.kiwisolver_1_4_5 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        wheel = wheel;
        setuptools-scm = setuptools-scm.with_toml;
        cppy = cppy;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pillow = nixpy-custom.pillow_10_4_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    ml-collections = buildPythonPackage {
      pname = "ml-collections";
      version = "0.1.1";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/aa/ea/853aa32dfa1006d3eb43384712f35b8f2d6f0a757b8c779d40c29e3e8515/ml_collections-0.1.1.tar.gz";
        hash="sha256-P+/McuxDOqHl0yMHo+R0u7Z/QFvoFOpSohZr/J2+aMw=";
      };
      build-system = with packages;
      [setuptools.v74_1_1];
      dependencies = with packages;
      [absl-py contextlib2 pyyaml six];
      doCheck=false;
    } ;
    numba = nixpy-custom.numba_0_60_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies=with packages;
      {
        numpy = numpy.v1_26_4;
        llvmlite = llvmlite;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    mujoco = nixpy-custom.mujoco_3_2_2 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        cmake = cmake;
        pybind11 = pybind11;
      };
      dependencies=with packages;
      {
        absl-py = absl-py;
        etils = etils.with_epath;
        glfw = glfw;
        numpy = numpy.v1_26_4;
        pyopengl = pyopengl;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pynput = {
      x86_64-linux = buildPythonPackage {
        pname = "pynput";
        version = "1.7.7";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/ef/1d/fdef3fdc9dc8dedc65898c8ad0e8922a914bb89c5308887e45f9aafaec36/pynput-1.7.7-py2.py3-none-any.whl";
          hash="sha256-r8Q/ZRaEyYgY3gSKvHat+fLT15cIPLB8H4K+dkotRMs=";
        };
        dependencies = with packages;
        [evdev python-xlib six];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "pynput";
        version = "1.7.7";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/ef/1d/fdef3fdc9dc8dedc65898c8ad0e8922a914bb89c5308887e45f9aafaec36/pynput-1.7.7-py2.py3-none-any.whl";
          hash="sha256-r8Q/ZRaEyYgY3gSKvHat+fLT15cIPLB8H4K+dkotRMs=";
        };
        dependencies = with packages;
        [pyobjc-framework-applicationservices pyobjc-framework-quartz six];
        doCheck=false;
      } ;
      powerpc64le-linux = buildPythonPackage {
        pname = "pynput";
        version = "1.7.7";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/ef/1d/fdef3fdc9dc8dedc65898c8ad0e8922a914bb89c5308887e45f9aafaec36/pynput-1.7.7-py2.py3-none-any.whl";
          hash="sha256-r8Q/ZRaEyYgY3gSKvHat+fLT15cIPLB8H4K+dkotRMs=";
        };
        dependencies = with packages;
        [evdev python-xlib six];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    termcolor = buildPythonPackage {
      pname = "termcolor";
      version = "2.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d9/5f/8c716e47b3a50cbd7c146f45881e11d9414def768b7cd9c5e6650ec2a80a/termcolor-2.4.0-py3-none-any.whl";
        hash="sha256-kpfA35yZRFwkEugy6IKniEA4olYXxgzqKtaUiNQEDWM=";
      };
      doCheck=false;
    } ;
    psutil = nixpy-custom.psutil_6_0_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
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
      [six];
      doCheck=false;
    } ;
    pyyaml = buildPythonPackage {
      pname = "pyyaml";
      version = "6.0.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/54/ed/79a089b6be93607fa5cdaedf301d7dfb23af5f25c398d5ead2525b063e17/pyyaml-6.0.2.tar.gz";
        hash="sha256-1YTZ7JGtZYYcwI1C6DQyTviQoILlkQN6vhFIUP97vD4=";
      };
      build-system = with packages;
      [cython.v3_0_11 setuptools.v74_1_1 wheel];
      doCheck=false;
    } ;
    setproctitle = buildPythonPackage {
      pname = "setproctitle";
      version = "1.3.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ff/e1/b16b16a1aa12174349d15b73fd4b87e641a8ae3fb1163e80938dbbf6ae98/setproctitle-1.3.3.tar.gz";
        hash="sha256-yRPhUefqAVZ4N/8DeiPKh0AZKIAZi3+7kLFtGBYHyq4=";
      };
      build-system = with packages;
      [setuptools.v74_1_1];
      doCheck=false;
    } ;
    platformdirs = buildPythonPackage {
      pname = "platformdirs";
      version = "4.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/68/13/2aa1f0e1364feb2c9ef45302f387ac0bd81484e9c9a4c5688a322fbdfd08/platformdirs-4.2.2-py3-none-any.whl";
        hash="sha256-LXoWV+NqgOqRHbgyqKbs5e5T2N4h7dXMWHmvZTCxv+4=";
      };
      doCheck=false;
    } ;
    botocore = buildPythonPackage {
      pname = "botocore";
      version = "1.35.11";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/16/27/1903dd4afd16530693476baf6cd9efc45706bc87e50af34dd64e1440e127/botocore-1.35.11-py3-none-any.whl";
        hash="sha256-6bZHts8fY/1wHCdDOALRxDQoOKN/0VK0D+U7ln/RmvQ=";
      };
      dependencies = with packages;
      [jmespath python-dateutil urllib3];
      doCheck=false;
    } ;
    jax-cuda12-prjt = {
      x86_64-linux = nixpy-custom.jax_cuda12_prjt_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d513b6658;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      powerpc64le-linux = nixpy-custom.jax_cuda12_prjt_0_4_34 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          build = build;
          cython = cython.v3_0_11;
          pybind11 = pybind11;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.dd14ec181;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
    };
    pluggy = buildPythonPackage {
      pname = "pluggy";
      version = "1.5.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/88/5f/e351af9a41f866ac3f1fac4ca0613908d9a41741cfcf2228f4ad853b697d/pluggy-1.5.0-py3-none-any.whl";
        hash="sha256-ROGtksjKAC3mN34WXz4PG+YyZqtNVUdAUyM1uddepmk=";
      };
      doCheck=false;
    } ;
    contextlib2 = buildPythonPackage {
      pname = "contextlib2";
      version = "21.6.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/76/56/6d6872f79d14c0cb02f1646cbb4592eef935857c0951a105874b7b62a0c3/contextlib2-21.6.0-py2.py3-none-any.whl";
        hash="sha256-P722RGav0jq69sl3Ynt1thOaWj6M44QFxbQTrtegRx8=";
      };
      doCheck=false;
    } ;
    six = buildPythonPackage {
      pname = "six";
      version = "1.16.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d9/5a/e7c31adbe875f2abbb91bd84cf2dc52d792b5a01506781dbcf25c91daf11/six-1.16.0-py2.py3-none-any.whl";
        hash="sha256-irsvHYaJCi37mJ+ad8/P0+R8KjVLAREXcTJviqJuAlQ=";
      };
      doCheck=false;
    } ;
    llvmlite = nixpy-custom.llvmlite_0_43_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    etils = {
      with_epath = buildPythonPackage {
        pname = "etils";
        version = "1.9.4";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/2e/35/7f8fcc9c23a504cf09e2795164eeb39a39ade1b2c7c8724ee207b2019ae6/etils-1.9.4-py3-none-any.whl";
          hash="sha256-Q4fnpJEaO1zEuSuZqSEThtF2tDuuHayOL+NF/Cy5Xks=";
        };
        dependencies = with packages;
        [fsspec importlib-resources typing-extensions zipp];
        doCheck=false;
      } ;
      with_epy = buildPythonPackage {
        pname = "etils";
        version = "1.9.4";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/2e/35/7f8fcc9c23a504cf09e2795164eeb39a39ade1b2c7c8724ee207b2019ae6/etils-1.9.4-py3-none-any.whl";
          hash="sha256-Q4fnpJEaO1zEuSuZqSEThtF2tDuuHayOL+NF/Cy5Xks=";
        };
        dependencies = with packages;
        [typing-extensions];
        doCheck=false;
      } ;
      with_epath_epy = buildPythonPackage {
        pname = "etils";
        version = "1.9.4";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/2e/35/7f8fcc9c23a504cf09e2795164eeb39a39ade1b2c7c8724ee207b2019ae6/etils-1.9.4-py3-none-any.whl";
          hash="sha256-Q4fnpJEaO1zEuSuZqSEThtF2tDuuHayOL+NF/Cy5Xks=";
        };
        dependencies = with packages;
        [fsspec importlib-resources typing-extensions zipp];
        doCheck=false;
      } ;
    };
    glfw = nixpy-custom.glfw_2_7_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pyopengl = nixpy-custom.pyopengl_3_1_7 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    evdev = nixpy-custom.evdev_1_7_1 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    fsspec = buildPythonPackage {
      pname = "fsspec";
      version = "2024.6.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5e/44/73bea497ac69bafde2ee4269292fa3b41f1198f4bb7bbaaabde30ad29d4a/fsspec-2024.6.1-py3-none-any.whl";
        hash="sha256-PLRD+LzS77MSlaW5/bAq7oHYRSyA0o+XptCVnmzuEB4=";
      };
      doCheck=false;
    } ;
    importlib-resources = buildPythonPackage {
      pname = "importlib-resources";
      version = "6.4.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/db/2a/728c8ae66011600fac5731a7db030d23c42f1321fd9547654f0c3b2b32d7/importlib_resources-6.4.4-py3-none-any.whl";
        hash="sha256-3aJCYD0cnNg2wzaLEXTtdMtASezSCeehoBBGIMGMXBE=";
      };
      doCheck=false;
    } ;
    typing-extensions = buildPythonPackage {
      pname = "typing-extensions";
      version = "4.12.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/26/9f/ad63fc0248c5379346306f8668cda6e2e2e9c95e01216d2b8ffd9ff037d0/typing_extensions-4.12.2-py3-none-any.whl";
        hash="sha256-BOXKA1Hg8/hcaFOVQHLfZZ0NE/rDJNAHIxa2fXeUcA0=";
      };
      doCheck=false;
    } ;
    zipp = buildPythonPackage {
      pname = "zipp";
      version = "3.20.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/07/9e/c96f7a4cd0bf5625bb409b7e61e99b1130dc63a98cb8b24aeabae62d43e8/zipp-3.20.1-py3-none-any.whl";
        hash="sha256-mWDNiWfI+FpW+SDV1QcnTnT5/4E6CriImltb4tr0QGQ=";
      };
      doCheck=false;
    } ;
    chex = buildPythonPackage {
      pname = "chex";
      version = "0.1.86";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e6/ed/625d545d08c6e258d2d63a93a0bf8ed8a296c09d254208e73f9d4fb0b746/chex-0.1.86-py3-none-any.whl";
        hash="sha256-JRwgghCSMjo9nCjhz4DkpYGAl4vsNo9TGUm9mEfu5Wg=";
      };
      dependencies = with packages;
      [absl-py jax jaxlib numpy.v1_26_4 toolz typing-extensions];
      doCheck=false;
    } ;
    einops = buildPythonPackage {
      pname = "einops";
      version = "0.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/44/5a/f0b9ad6c0a9017e62d4735daaeb11ba3b6c009d69a26141b258cd37b5588/einops-0.8.0-py3-none-any.whl";
        hash="sha256-lXL7YwRiZKhiaTsKhwiK873IwGj94D3mNFPLveJFRl8=";
      };
      doCheck=false;
    } ;
    zarr = buildPythonPackage {
      pname = "zarr";
      version = "2.18.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/5d/bd/8d881d8ca6d80fcb8da2b2f94f8855384daf649499ddfba78ffd1ee2caa3/zarr-2.18.2-py3-none-any.whl";
        hash="sha256-pjh1SQL5fvqZtAYIP9yAeg4szxKpSRFzidKkupsF3zg=";
      };
      dependencies = with packages;
      [asciitree fasteners numcodecs numpy.v1_26_4];
      doCheck=false;
    } ;
    asciitree = buildPythonPackage {
      pname = "asciitree";
      version = "0.3.3";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/2d/6a/885bc91484e1aa8f618f6f0228d76d0e67000b0fdd6090673b777e311913/asciitree-0.3.3.tar.gz";
        hash="sha256-SqS5tkn4Xj/LNDNj2XVkqh+2LiSWd/LhipZ2UUXMD24=";
      };
      build-system = with packages;
      [setuptools.v74_1_1];
      doCheck=false;
    } ;
    fasteners = buildPythonPackage {
      pname = "fasteners";
      version = "0.19";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/61/bf/fd60001b3abc5222d8eaa4a204cd8c0ae78e75adc688f33ce4bf25b7fafa/fasteners-0.19-py3-none-any.whl";
        hash="sha256-dYgZy12Uze306DaYi3TeOWzqy44nlNIfgtEx/Z7ncjc=";
      };
      doCheck=false;
    } ;
    rich = buildPythonPackage {
      pname = "rich";
      version = "13.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c7/d9/c2a126eeae791e90ea099d05cb0515feea3688474b978343f3cdcfe04523/rich-13.8.0-py3-none-any.whl";
        hash="sha256-LoUwagY7lJLf/IYngZemDL7Odby3ZgIvNDb1Z8rhG9w=";
      };
      dependencies = with packages;
      [markdown-it-py pygments];
      doCheck=false;
    } ;
    markdown-it-py = buildPythonPackage {
      pname = "markdown-it-py";
      version = "3.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/42/d7/1ec15b46af6af88f19b8e5ffea08fa375d433c998b8a7639e76935c14f1f/markdown_it_py-3.0.0-py3-none-any.whl";
        hash="sha256-NVIWhFxgvZYjLNjYxA6Pl2XMhvRogOQ6j9ItwaGoyrE=";
      };
      dependencies = with packages;
      [mdurl];
      doCheck=false;
    } ;
    pandas = buildPythonPackage {
      pname = "pandas";
      version = "2.2.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/88/d9/ecf715f34c73ccb1d8ceb82fc01cd1028a65a5f6dbc57bfa6ea155119058/pandas-2.2.2.tar.gz";
        hash="sha256-nnkBmrpDy0/ank2YP46IygNzrbtpeunGxDCTIY3ii1Q=";
      };
      build-system = with packages;
      [cython.v3_0_5 meson-python.v0_13_1 meson.v1_2_1 numpy.v2_0_1 versioneer wheel];
      dependencies = with packages;
      [numpy.v1_26_4 python-dateutil pytz tzdata];
      doCheck=false;
    } ;
    python-dateutil = buildPythonPackage {
      pname = "python-dateutil";
      version = "2.9.0.post0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl";
        hash="sha256-qLK8e/+uKCKByBQKl9OqnBTaCxNt/oP4UO6ppfdHBCc=";
      };
      dependencies = with packages;
      [six];
      doCheck=false;
    } ;
    s3transfer = buildPythonPackage {
      pname = "s3transfer";
      version = "0.10.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3c/4a/b221409913760d26cf4498b7b1741d510c82d3ad38381984a3ddc135ec66/s3transfer-0.10.2-py3-none-any.whl";
        hash="sha256-7KHCDecKOdruWArvSYaZZiDzZcTg/aaoYQAjHWLxv2k=";
      };
      dependencies = with packages;
      [botocore];
      doCheck=false;
    } ;
    mdurl = buildPythonPackage {
      pname = "mdurl";
      version = "0.1.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/38/89ba8ad64ae25be8de66a6d463314cf1eb366222074cfda9ee839c56a4b4/mdurl-0.1.2-py3-none-any.whl";
        hash="sha256-hACKQeUWFaSfyZZhkf+RUJ48QLk5F25kP9UKXCGWuPg=";
      };
      doCheck=false;
    } ;
    mujoco-mjx = buildPythonPackage {
      pname = "mujoco-mjx";
      version = "3.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/de/5b/26a20a24672b9ba7b16bebe2c9afb28ae432960272fa7e8d27d8be8bea76/mujoco_mjx-3.2.2-py3-none-any.whl";
        hash="sha256-p6RxiCvwDvIRi84lSpw4J5hwRnr5TKP2KrXPoVksAik=";
      };
      dependencies = with packages;
      [absl-py etils.with_epath jax jaxlib mujoco scipy trimesh];
      doCheck=false;
    } ;
    trimesh = buildPythonPackage {
      pname = "trimesh";
      version = "4.4.8";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d7/bc/636613b49265748a81f7d88d488f16260ee78a19ee92182c3ae722415a03/trimesh-4.4.8-py3-none-any.whl";
        hash="sha256-H+sAbhP626soUn5PmY4YFmiuvcbOVVG9JfoNIspazew=";
      };
      dependencies = with packages;
      [numpy.v1_26_4];
      doCheck=false;
    } ;
    cycler = buildPythonPackage {
      pname = "cycler";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e7/05/c19819d5e3d95294a6f5947fb9b9629efb316b96de511b418c53d245aae6/cycler-0.12.1-py3-none-any.whl";
        hash="sha256-hc73z/Ii2GRBYVKYCEZZcuUTQFmUWbisPMusWoVODTA=";
      };
      doCheck=false;
    } ;
    antlr4-python3-runtime = buildPythonPackage {
      pname = "antlr4-python3-runtime";
      version = "4.9.3";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz";
        hash="sha256-8iRGm0FoKUkCux76gKi/eFXyTJmu+Zy+/BvNPM53iBs=";
      };
      build-system = with packages;
      [setuptools.v74_1_1];
      doCheck=false;
    } ;
    pytest = buildPythonPackage {
      pname = "pytest";
      version = "8.3.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6b/77/7440a06a8ead44c7757a64362dd22df5760f9b12dc5f11b6188cd2fc27a0/pytest-8.3.3-py3-none-any.whl";
        hash="sha256-poU8c3WyZjFVB5RD0uRd6ROpEaEdZp3wKlCBSUTbV7I=";
      };
      dependencies = with packages;
      [exceptiongroup iniconfig packaging pluggy tomli];
      doCheck=false;
    } ;
    iniconfig = buildPythonPackage {
      pname = "iniconfig";
      version = "2.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ef/a6/62565a6e1cf69e10f5727360368e451d4b7f58beeac6173dc9db836a5b46/iniconfig-2.0.0-py3-none-any.whl";
        hash="sha256-tqhYcaedLjsi0tG5SsKCQiamPGt0HIj3rpdfGLZ3g3Q=";
      };
      doCheck=false;
    } ;
    optax = {
      x86_64-linux = {
        d3a5bf6ee = buildPythonPackage {
          pname = "optax";
          version = "0.2.3";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
            hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
          };
          dependencies = with packages;
          [absl-py chex etils.with_epath_epy jax jaxlib numpy.v1_26_4];
          doCheck=false;
        } ;
        d912aa156 = buildPythonPackage {
          pname = "optax";
          version = "0.2.3";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
            hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
          };
          dependencies = with packages;
          [absl-py chex etils.with_epath jax jaxlib numpy.v1_26_4];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        de614ffb8 = buildPythonPackage {
          pname = "optax";
          version = "0.2.3";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
            hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
          };
          dependencies = with packages;
          [absl-py chex etils.with_epath_epy jax jaxlib numpy.v1_26_4];
          doCheck=false;
        } ;
        d1f799c17 = buildPythonPackage {
          pname = "optax";
          version = "0.2.3";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
            hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
          };
          dependencies = with packages;
          [absl-py chex etils.with_epath jax jaxlib numpy.v1_26_4];
          doCheck=false;
        } ;
      };
      aarch64-darwin = {
        d4951cf1e = buildPythonPackage {
          pname = "optax";
          version = "0.2.3";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
            hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
          };
          dependencies = with packages;
          [absl-py chex etils.with_epath_epy jax jaxlib numpy.v1_26_4];
          doCheck=false;
        } ;
        d7821d309 = buildPythonPackage {
          pname = "optax";
          version = "0.2.3";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/a3/8b/7032a6788205e9da398a8a33e1030ee9a22bd9289126e5afed9aac33bcde/optax-0.2.3-py3-none-any.whl";
            hash="sha256-CD5gPc1zHX502Z9xwS93k33VP3kAG0wJwpDk9H3S6U8=";
          };
          dependencies = with packages;
          [absl-py chex etils.with_epath jax jaxlib numpy.v1_26_4];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    flax = {
      powerpc64le-linux = buildPythonPackage {
        pname = "flax";
        version = "0.9.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/e0/e8/e0aa0c81a4b2c14bcaf7566d865039d4ae39ed604b8ba90708f8faedbda5/flax-0.9.0-py3-none-any.whl";
          hash="sha256-Es2PcWIWXd1Wh3+xzZpPy0ejFWnkxTQ+61mjY2n6LP4=";
        };
        dependencies = with packages;
        [jax msgpack optax.d1f799c17 orbax-checkpoint.dcc546a6d pyyaml rich tensorstore typing-extensions];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "flax";
        version = "0.9.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/e0/e8/e0aa0c81a4b2c14bcaf7566d865039d4ae39ed604b8ba90708f8faedbda5/flax-0.9.0-py3-none-any.whl";
          hash="sha256-Es2PcWIWXd1Wh3+xzZpPy0ejFWnkxTQ+61mjY2n6LP4=";
        };
        dependencies = with packages;
        [jax msgpack optax.d7821d309 orbax-checkpoint.d710ea424 pyyaml rich tensorstore typing-extensions];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "flax";
        version = "0.9.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/e0/e8/e0aa0c81a4b2c14bcaf7566d865039d4ae39ed604b8ba90708f8faedbda5/flax-0.9.0-py3-none-any.whl";
          hash="sha256-Es2PcWIWXd1Wh3+xzZpPy0ejFWnkxTQ+61mjY2n6LP4=";
        };
        dependencies = with packages;
        [jax msgpack optax.d912aa156 orbax-checkpoint.dbd2dbc15 pyyaml rich tensorstore typing-extensions];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    msgpack = nixpy-custom.msgpack_1_0_8 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        cython = cython.v3_0_11;
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    orbax-checkpoint = {
      aarch64-darwin = {
        d710ea424 = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.6.1";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/60/be/378ca2a60c2fb368f9aafe463af583fd3cd6c60e0d39941ac09755647686/orbax_checkpoint-0.6.1-py3-none-any.whl";
            hash="sha256-9/yx71KM7ilOokTnaeruF94jecaKANbfTDpGPlz3FqE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath humanize jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
        db029cf14 = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.6.1";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/60/be/378ca2a60c2fb368f9aafe463af583fd3cd6c60e0d39941ac09755647686/orbax_checkpoint-0.6.1-py3-none-any.whl";
            hash="sha256-9/yx71KM7ilOokTnaeruF94jecaKANbfTDpGPlz3FqE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath_epy humanize jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
      };
      x86_64-linux = {
        d6175137a = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.6.1";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/60/be/378ca2a60c2fb368f9aafe463af583fd3cd6c60e0d39941ac09755647686/orbax_checkpoint-0.6.1-py3-none-any.whl";
            hash="sha256-9/yx71KM7ilOokTnaeruF94jecaKANbfTDpGPlz3FqE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath_epy humanize jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
        dbd2dbc15 = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.6.1";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/60/be/378ca2a60c2fb368f9aafe463af583fd3cd6c60e0d39941ac09755647686/orbax_checkpoint-0.6.1-py3-none-any.whl";
            hash="sha256-9/yx71KM7ilOokTnaeruF94jecaKANbfTDpGPlz3FqE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath humanize jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        d740edf33 = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.6.1";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/60/be/378ca2a60c2fb368f9aafe463af583fd3cd6c60e0d39941ac09755647686/orbax_checkpoint-0.6.1-py3-none-any.whl";
            hash="sha256-9/yx71KM7ilOokTnaeruF94jecaKANbfTDpGPlz3FqE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath_epy humanize jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
        dcc546a6d = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.6.1";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/60/be/378ca2a60c2fb368f9aafe463af583fd3cd6c60e0d39941ac09755647686/orbax_checkpoint-0.6.1-py3-none-any.whl";
            hash="sha256-9/yx71KM7ilOokTnaeruF94jecaKANbfTDpGPlz3FqE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath humanize jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    tensorstore = {
      x86_64-linux = nixpy-custom.tensorstore_0_1_64 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d513b6658;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      aarch64-darwin = nixpy-custom.tensorstore_0_1_64 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d618fa062;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      powerpc64le-linux = nixpy-custom.tensorstore_0_1_64 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v74_1_1;
          wheel = wheel;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.dd14ec181;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
    };
    humanize = buildPythonPackage {
      pname = "humanize";
      version = "4.10.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/8f/49/a29c79bea335e52fb512a43faf84998c184c87fef82c65f568f8c56f2642/humanize-4.10.0-py3-none-any.whl";
        hash="sha256-OefMuWkj5zK1wuJ66qOxCo3+66Prllunt0o+sOMAQKY=";
      };
      doCheck=false;
    } ;
    nest-asyncio = buildPythonPackage {
      pname = "nest-asyncio";
      version = "1.6.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a0/c4/c2971a3ba4c6103a3d10c4b0f24f461ddc027f0f09763220cf35ca1401b3/nest_asyncio-1.6.0-py3-none-any.whl";
        hash="sha256-h69u/WteiXyBBQR372XGLisvNdUXA8rgGv8pBbGFLhw=";
      };
      doCheck=false;
    } ;
    nbformat = buildPythonPackage {
      pname = "nbformat";
      version = "5.10.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a9/82/0340caa499416c78e5d8f5f05947ae4bc3cba53c9f038ab6e9ed964e22f1/nbformat-5.10.4-py3-none-any.whl";
        hash="sha256-O0jWyPvKSymb85gup9sa8hWA5P7Caa0Ie56BWIiRIAs=";
      };
      dependencies = with packages;
      [fastjsonschema jsonschema jupyter-core traitlets];
      doCheck=false;
    } ;
    tzdata = buildPythonPackage {
      pname = "tzdata";
      version = "2024.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/65/58/f9c9e6be752e9fcb8b6a0ee9fb87e6e7a1f6bcab2cdc73f02bb7ba91ada0/tzdata-2024.1-py2.py3-none-any.whl";
        hash="sha256-kGi8GWE2Rj9SReUe/ag4r6FarsqZA/SQUN+iZ5200lI=";
      };
      doCheck=false;
    } ;
    plotly = buildPythonPackage {
      pname = "plotly";
      version = "5.24.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b3/f6/64fe388089588a0c364b03809443cd628fe0261d1364313fb7aec6bb0ebc/plotly-5.24.0-py3-none-any.whl";
        hash="sha256-DlTv5SyM74mffapBvp7Zfftr5iJhOiqPVqhqBjSytn4=";
      };
      dependencies = with packages;
      [packaging tenacity];
      doCheck=false;
    } ;
    toolz = buildPythonPackage {
      pname = "toolz";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/8a/d82202c9f89eab30f9fc05380daae87d617e2ad11571ab23d7c13a29bb54/toolz-0.12.1-py3-none-any.whl";
        hash="sha256-0icxNkwH1y7qCgrUW6+ywpN6tv04o1B79V6uh0SqfYU=";
      };
      doCheck=false;
    } ;
    numcodecs = buildPythonPackage {
      pname = "numcodecs";
      version = "0.13.0";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f8/22/e5cba9013403186906390c0efb0ab0db60d4e580a8966650b2372ab967e1/numcodecs-0.13.0.tar.gz";
        hash="sha256-uk+scDbqWgeMev4dTf/rloUIDULxnJwWsS2thmcDqi4=";
      };
      build-system = with packages;
      [cython.v3_0_11 numpy.v1_26_4 py-cpuinfo setuptools-scm.with_toml setuptools.v74_1_1];
      dependencies = with packages;
      [numpy.v1_26_4];
      doCheck=false;
    } ;
    pygments = buildPythonPackage {
      pname = "pygments";
      version = "2.18.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f7/3f/01c8b82017c199075f8f788d0d906b9ffbbc5a47dc9918a945e13d5a2bda/pygments-2.18.0-py3-none-any.whl";
        hash="sha256-uOasoFI/Ordv7lF5nEiOOHgqwG6vz5XnuoMphcjnsTo=";
      };
      doCheck=false;
    } ;
    jmespath = buildPythonPackage {
      pname = "jmespath";
      version = "1.0.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/31/b4/b9b800c45527aadd64d5b442f9b932b00648617eb5d63d2c7a6587b7cafc/jmespath-1.0.1-py3-none-any.whl";
        hash="sha256-AuLkzHG1vKuIMy7r+QdRkZDdnm6CEH+n+DsQA6YlKYA=";
      };
      doCheck=false;
    } ;
    contourpy = {
      x86_64-linux = {
        d12135c46 = buildPythonPackage {
          pname = "contourpy";
          version = "1.3.0";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/f5/f6/31a8f28b4a2a4fa0e01085e542f3081ab0588eff8e589d39d775172c9792/contourpy-1.3.0.tar.gz";
            hash="sha256-f/oNsXcXqP+xJ+/QyVpDYtmWuJLCkE23JCjVtS4ZOKQ=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11];
          dependencies = with packages;
          [numpy.v1_26_4];
          doCheck=false;
        } ;
        dd28ac591 = buildPythonPackage {
          pname = "contourpy";
          version = "1.3.0";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/f5/f6/31a8f28b4a2a4fa0e01085e542f3081ab0588eff8e589d39d775172c9792/contourpy-1.3.0.tar.gz";
            hash="sha256-f/oNsXcXqP+xJ+/QyVpDYtmWuJLCkE23JCjVtS4ZOKQ=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11];
          dependencies = with packages;
          [numpy.v2_0_1];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        d5c0bade7 = buildPythonPackage {
          pname = "contourpy";
          version = "1.3.0";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/f5/f6/31a8f28b4a2a4fa0e01085e542f3081ab0588eff8e589d39d775172c9792/contourpy-1.3.0.tar.gz";
            hash="sha256-f/oNsXcXqP+xJ+/QyVpDYtmWuJLCkE23JCjVtS4ZOKQ=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11];
          dependencies = with packages;
          [numpy.v1_26_4];
          doCheck=false;
        } ;
        d31f6e430 = buildPythonPackage {
          pname = "contourpy";
          version = "1.3.0";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/f5/f6/31a8f28b4a2a4fa0e01085e542f3081ab0588eff8e589d39d775172c9792/contourpy-1.3.0.tar.gz";
            hash="sha256-f/oNsXcXqP+xJ+/QyVpDYtmWuJLCkE23JCjVtS4ZOKQ=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11];
          dependencies = with packages;
          [numpy.v2_0_1];
          doCheck=false;
        } ;
      };
      aarch64-darwin = {
        dda94ea09 = buildPythonPackage {
          pname = "contourpy";
          version = "1.3.0";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/f5/f6/31a8f28b4a2a4fa0e01085e542f3081ab0588eff8e589d39d775172c9792/contourpy-1.3.0.tar.gz";
            hash="sha256-f/oNsXcXqP+xJ+/QyVpDYtmWuJLCkE23JCjVtS4ZOKQ=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11];
          dependencies = with packages;
          [numpy.v1_26_4];
          doCheck=false;
        } ;
        d4f897f7a = buildPythonPackage {
          pname = "contourpy";
          version = "1.3.0";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/f5/f6/31a8f28b4a2a4fa0e01085e542f3081ab0588eff8e589d39d775172c9792/contourpy-1.3.0.tar.gz";
            hash="sha256-f/oNsXcXqP+xJ+/QyVpDYtmWuJLCkE23JCjVtS4ZOKQ=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11];
          dependencies = with packages;
          [numpy.v2_0_1];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    exceptiongroup = buildPythonPackage {
      pname = "exceptiongroup";
      version = "1.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/02/cc/b7e31358aac6ed1ef2bb790a9746ac2c69bcb3c8588b41616914eb106eaf/exceptiongroup-1.2.2-py3-none-any.whl";
        hash="sha256-MRG50THCOL7C+PUW4SPhS6JDVj+xNdP+iFmQWFqneVs=";
      };
      doCheck=false;
    } ;
    tomli = buildPythonPackage {
      pname = "tomli";
      version = "2.0.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/97/75/10a9ebee3fd790d20926a90a2547f0bf78f371b2f13aa822c759680ca7b9/tomli-2.0.1-py3-none-any.whl";
        hash="sha256-k53j56YWGvDIh++Rt9QaU+fFocqXYyX0KctG6pvDDsw=";
      };
      doCheck=false;
    } ;
    absl-py = buildPythonPackage {
      pname = "absl-py";
      version = "2.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a2/ad/e0d3c824784ff121c03cc031f944bc7e139a8f1870ffd2845cc2dd76f6c4/absl_py-2.1.0-py3-none-any.whl";
        hash="sha256-UmoE6tq4tO5xnOaPIEFy6tECdUkIlwLZm5BZ8Sn/Ewg=";
      };
      doCheck=false;
    } ;
    click = buildPythonPackage {
      pname = "click";
      version = "8.1.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/00/2e/d53fa4befbf2cfa713304affc7ca780ce4fc1fd8710527771b58311a3229/click-8.1.7-py3-none-any.whl";
        hash="sha256-rnT7lsIKAneh1hXx5Nc8hBT1qY24t5mnkx0VgvM5DCg=";
      };
      doCheck=false;
    } ;
    fastjsonschema = buildPythonPackage {
      pname = "fastjsonschema";
      version = "2.20.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6d/ca/086311cdfc017ec964b2436fe0c98c1f4efcb7e4c328956a22456e497655/fastjsonschema-2.20.0-py3-none-any.whl";
        hash="sha256-WHXwsPp6AEOpHpOpuPeTvLu6lpHn/YPcqVwouibSHwo=";
      };
      doCheck=false;
    } ;
    setuptools = {
      v74_1_1 = buildPythonPackage {
        pname = "setuptools";
        version = "74.1.1";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/48/f3/e30ee63caefa90716afdffd7d9ae959cd8d0dbd2d0a0eb9fe1d73ddf806b/setuptools-74.1.1-py3-none-any.whl";
          hash="sha256-/JG1+J45LvW3f+FDsX4y9l0wJHRPumbcOv4HIBaE12Y=";
        };
        doCheck=false;
      } ;
      v70_1_1 = buildPythonPackage {
        pname = "setuptools";
        version = "70.1.1";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/b3/7a/629889a5d76200287aa5483d753811bd247bbd1b03175186f759e0c7d3a7/setuptools-70.1.1-py3-none-any.whl";
          hash="sha256-pYqP3gVB2rBBl1C8xSH734WF9uXLQZCd86Ry73uBypU=";
        };
        doCheck=false;
      } ;
    };
    python-xlib = buildPythonPackage {
      pname = "python-xlib";
      version = "0.33";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fc/b8/ff33610932e0ee81ae7f1269c890f697d56ff74b9f5b2ee5d9b7fa2c5355/python_xlib-0.33-py2.py3-none-any.whl";
        hash="sha256-w1NAONQuDfLxOSobMKFaT/X9wrhs+pTwcr8RsQoWQ5g=";
      };
      dependencies = with packages;
      [six];
      doCheck=false;
    } ;
    tenacity = buildPythonPackage {
      pname = "tenacity";
      version = "9.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b6/cb/b86984bed139586d01532a587464b5805f12e397594f19f931c4c2fbfa61/tenacity-9.0.0-py3-none-any.whl";
        hash="sha256-k94MmHhbJ/z2WYVqqfVL+9OZ4plpsGIbx/divUQbRTk=";
      };
      doCheck=false;
    } ;
    pytz = buildPythonPackage {
      pname = "pytz";
      version = "2024.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9c/3d/a121f284241f08268b21359bd425f7d4825cffc5ac5cd0e1b3d82ffd2b10/pytz-2024.1-py2.py3-none-any.whl";
        hash="sha256-MoFx9ONiMTnaSYNFGVCyjpWscG4T8/JjCoeXSeeosxk=";
      };
      doCheck=false;
    } ;
    jupyter-core = buildPythonPackage {
      pname = "jupyter-core";
      version = "5.7.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c9/fb/108ecd1fe961941959ad0ee4e12ee7b8b1477247f30b1fdfd83ceaf017f0/jupyter_core-5.7.2-py3-none-any.whl";
        hash="sha256-T3MV0va0vPLj58tuRncuunYK5FnNH1nSnrV7CgG9dAk=";
      };
      dependencies = with packages;
      [platformdirs traitlets];
      doCheck=false;
    } ;
    packaging = buildPythonPackage {
      pname = "packaging";
      version = "24.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/08/aa/cc0199a5f0ad350994d660967a8efb233fe0416e4639146c089643407ce6/packaging-24.1-py3-none-any.whl";
        hash="sha256-W48iF9vb0vfzhMQcYoVE5tUvLQ9TxtDD6mGqXR1/8SQ=";
      };
      doCheck=false;
    } ;
    traitlets = buildPythonPackage {
      pname = "traitlets";
      version = "5.14.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/00/c0/8f5d070730d7836adc9c9b6408dec68c6ced86b304a9b26a14df072a6e8c/traitlets-5.14.3-py3-none-any.whl";
        hash="sha256-t06J45ex7SjMgx23rqdZumZAyz3hMJDKFFQmaI/xrE8=";
      };
      doCheck=false;
    } ;
    pyparsing = buildPythonPackage {
      pname = "pyparsing";
      version = "3.1.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e5/0c/0e3c05b1c87bb6a1c76d281b0f35e78d2d80ac91b5f8f524cebf77f51049/pyparsing-3.1.4-py3-none-any.whl";
        hash="sha256-pqfuQjWj+USqH6Ikkwdwj4k/5XF9xgNQPGx5acBw+3w=";
      };
      doCheck=false;
    } ;
    soupsieve = buildPythonPackage {
      pname = "soupsieve";
      version = "2.6";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/d1/c2/fe97d779f3ef3b15f05c94a2f1e3d21732574ed441687474db9d342a7315/soupsieve-2.6-py3-none-any.whl";
        hash="sha256-5yxP8G5PtuS1qfD1X+boFRRYH8oVFQKGJdDymcYCzMk=";
      };
      doCheck=false;
    } ;
    urllib3 = buildPythonPackage {
      pname = "urllib3";
      version = "2.2.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ca/1c/89ffc63a9605b583d5df2be791a27bc1a42b7c32bab68d3c8f2f73a98cd4/urllib3-2.2.2-py3-none-any.whl";
        hash="sha256-pEiy9k1oYVVGgDfhrOny0hmXduF/CkZhBIDTEfc+NHI=";
      };
      doCheck=false;
    } ;
    protobuf = buildPythonPackage {
      pname = "protobuf";
      version = "5.28.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e3/b2/4df9958122a0377e571972c71692420bafd623d1df3ce506d88c2aba7e12/protobuf-5.28.0-py3-none-any.whl";
        hash="sha256-UQ7XjNCYD20yGAmeh0cUzfDYqVWC57BZsGyrrYVe0KA=";
      };
      doCheck=false;
    } ;
    jsonschema = buildPythonPackage {
      pname = "jsonschema";
      version = "4.23.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/69/4a/4f9dbeb84e8850557c02365a0eee0649abe5eb1d84af92a25731c6c0f922/jsonschema-4.23.0-py3-none-any.whl";
        hash="sha256-+622+LFEqPjPnwuJupRQHRQ+UEEaEnhjP1anrPf9VWY=";
      };
      dependencies = with packages;
      [attrs jsonschema-specifications referencing rpds-py];
      doCheck=false;
    } ;
    rpds-py = nixpy-custom.rpds_py_0_20_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    attrs = buildPythonPackage {
      pname = "attrs";
      version = "24.2.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/6a/21/5b6702a7f963e95456c0de2d495f67bf5fd62840ac655dc451586d23d39a/attrs-24.2.0-py3-none-any.whl";
        hash="sha256-gZIeuW3jGRyCWMGZYYEE3SesYI2TZvXjXQEerhhn7eI=";
      };
      doCheck=false;
    } ;
    jsonschema-specifications = buildPythonPackage {
      pname = "jsonschema-specifications";
      version = "2023.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ee/07/44bd408781594c4d0a027666ef27fab1e441b109dc3b76b4f836f8fd04fe/jsonschema_specifications-2023.12.1-py3-none-any.whl";
        hash="sha256-h+T986lIWLiiuid42bpX2KnK/KfHSJxGug0wqLxqnDw=";
      };
      dependencies = with packages;
      [referencing];
      doCheck=false;
    } ;
    referencing = buildPythonPackage {
      pname = "referencing";
      version = "0.35.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b7/59/2056f61236782a2c86b33906c025d4f4a0b17be0161b63b70fd9e8775d36/referencing-0.35.1-py3-none-any.whl";
        hash="sha256-7abTI01igU0cZOMFwTMcmjphMtpHWrY4LqqZeyHudd4=";
      };
      dependencies = with packages;
      [attrs rpds-py];
      doCheck=false;
    } ;
    gitpython = buildPythonPackage {
      pname = "gitpython";
      version = "3.1.43";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e9/bd/cc3a402a6439c15c3d4294333e13042b915bbeab54edc457c723931fed3f/GitPython-3.1.43-py3-none-any.whl";
        hash="sha256-7sfsVrkqrXUfmRKnNAS8ArohKiOtsscJjuZoQXBRof8=";
      };
      dependencies = with packages;
      [gitdb];
      doCheck=false;
    } ;
    gitdb = buildPythonPackage {
      pname = "gitdb";
      version = "4.0.11";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/fd/5b/8f0c4a5bb9fd491c277c21eff7ccae71b47d43c4446c9d0c6cff2fe8c2c4/gitdb-4.0.11-py3-none-any.whl";
        hash="sha256-gaNAfd0u6N9ETLrOoA4tA45AFQrPowAWlv4Nzx0636Q=";
      };
      dependencies = with packages;
      [smmap];
      doCheck=false;
    } ;
    smmap = buildPythonPackage {
      pname = "smmap";
      version = "5.0.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a7/a5/10f97f73544edcdef54409f1d839f6049a0d79df68adbc1ceb24d1aaca42/smmap-5.0.1-py3-none-any.whl";
        hash="sha256-5thmj6X5PnBpNKYte02xnI2euM8q27de8bZ1qjMrado=";
      };
      doCheck=false;
    } ;
    requests = buildPythonPackage {
      pname = "requests";
      version = "2.32.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/f9/9b/335f9764261e915ed497fcdeb11df5dfd6f7bf257d4a6a2a686d80da4d54/requests-2.32.3-py3-none-any.whl";
        hash="sha256-cHYc/gPHc86yKqL2cbR1eXYUUXXN/KA4wCZU0GHW3MY=";
      };
      dependencies = with packages;
      [certifi charset-normalizer idna urllib3];
      doCheck=false;
    } ;
    idna = buildPythonPackage {
      pname = "idna";
      version = "3.8";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/22/7e/d71db821f177828df9dea8c42ac46473366f191be53080e552e628aad991/idna-3.8-py3-none-any.whl";
        hash="sha256-BQtOW6rc1E12DO29K45jny/4m7x6VzD8xmKVQwM3eqw=";
      };
      doCheck=false;
    } ;
    charset-normalizer = buildPythonPackage {
      pname = "charset-normalizer";
      version = "3.3.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/28/76/e6222113b83e3622caa4bb41032d0b1bf785250607392e1b778aca0b8a7d/charset_normalizer-3.3.2-py3-none-any.whl";
        hash="sha256-Pk0fZYcyLSeIg2qZxpBi+7CRMx7JQOAtEtF5wdU+Jfw=";
      };
      doCheck=false;
    } ;
    certifi = buildPythonPackage {
      pname = "certifi";
      version = "2024.8.30";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/12/90/3c9ff0512038035f59d279fddeb79f5f1eccd8859f06d6163c58798b9487/certifi-2024.8.30-py3-none-any.whl";
        hash="sha256-kiggtT23pyV/+9o/WXJm1DUkWQPYBzfjT4pF/z4yMNg=";
      };
      doCheck=false;
    } ;
    fonttools = buildPythonPackage {
      pname = "fonttools";
      version = "4.53.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e4/b9/0394d67056d4ad36a3807b439571934b318f1df925593a95e9ec0516b1a7/fonttools-4.53.1-py3-none-any.whl";
        hash="sha256-8fh1iirREL1kMiA6NEJp9EWikH3CTva8z9CsThTg1x0=";
      };
      doCheck=false;
    } ;
    sentry-sdk = buildPythonPackage {
      pname = "sentry-sdk";
      version = "2.13.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ad/7e/e9ca09f24a6c334286631a2d32c267cdc5edad5ac03fd9d20a01a82f1c35/sentry_sdk-2.13.0-py2.py3-none-any.whl";
        hash="sha256-a+7ej8KrQEPaf2nZVTTjIJRGkGgN2aljF4pJ3nHXJsY=";
      };
      dependencies = with packages;
      [certifi urllib3];
      doCheck=false;
    } ;
    pdm-backend = buildPythonPackage {
      pname = "pdm-backend";
      version = "2.3.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/eb/fe/483cf0918747a32800795f430319ec292f833eb871ba6da3ebed4553a575/pdm_backend-2.3.3-py3-none-any.whl";
        hash="sha256-226G3oyoTkJkw1piCHexSrqAkq16NN45VxVVMURmiCM=";
      };
      doCheck=false;
    } ;
    wheel = buildPythonPackage {
      pname = "wheel";
      version = "0.44.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1b/d1/9babe2ccaecff775992753d8686970b1e2755d21c8a63be73aba7a4e7d77/wheel-0.44.0-py3-none-any.whl";
        hash="sha256-I3apDJjMM30YYjUnqXwxeXvQK60AM9QVRwQ6HL++RI8=";
      };
      doCheck=false;
    } ;
    meson-python = {
      v0_15_0 = nixpy-custom.meson_python_0_15_0 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          meson = meson.v1_5_1;
          ninja = ninja;
          packaging = packaging;
          pyproject-metadata = pyproject-metadata;
          tomli = tomli;
        };
        dependencies=with packages;
        {
          meson = meson.v1_5_1;
          packaging = packaging;
          pyproject-metadata = pyproject-metadata;
          tomli = tomli;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      v0_13_1 = nixpy-custom.meson_python_0_13_1 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          meson = meson.v1_2_1;
          ninja = ninja;
          packaging = packaging;
          pyproject-metadata = pyproject-metadata;
          tomli = tomli;
        };
        dependencies=with packages;
        {
          meson = meson.v1_2_1;
          packaging = packaging;
          pyproject-metadata = pyproject-metadata;
          tomli = tomli;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    };
    pyproject-metadata = buildPythonPackage {
      pname = "pyproject-metadata";
      version = "0.8.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/aa/5f/bb5970d3d04173b46c9037109f7f05fc8904ff5be073ee49bb6ff00301bc/pyproject_metadata-0.8.0-py3-none-any.whl";
        hash="sha256-rYWNRI4dOh+0CKxbrJ6ndD56i7tHLyaTqqM00ttC9SY=";
      };
      dependencies = with packages;
      [packaging];
      doCheck=false;
    } ;
    cython = {
      v3_0_11 = buildPythonPackage {
        pname = "cython";
        version = "3.0.11";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/43/39/bdbec9142bc46605b54d674bf158a78b191c2b75be527c6dcf3e6dfe90b8/Cython-3.0.11-py2.py3-none-any.whl";
          hash="sha256-DiX2QlrUpwDX93zUaNqRYeY2WIN9G8NIYamGGk72NG0=";
        };
        doCheck=false;
      } ;
      v3_0_5 = buildPythonPackage {
        pname = "cython";
        version = "3.0.5";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/fb/fe/e213d8e9cb21775bb8f9c92ff97861504129e23e33d118be1a90ca26a13e/Cython-3.0.5-py2.py3-none-any.whl";
          hash="sha256-dSBjaVBPxELBCobs9XuRWS3KdE5Fkq8ipH6ad01T3RA=";
        };
        doCheck=false;
      } ;
    };
    meson = {
      v1_5_1 = buildPythonPackage {
        pname = "meson";
        version = "1.5.1";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/b7/33/513a9ca4fd5892463abb38592105b78fd425214f7983033633e2e48cbd30/meson-1.5.1-py3-none-any.whl";
          hash="sha256-VTHiTmz9YAC/HHEnk88o3/AyhBNwsaO5QaiU5P3kblo=";
        };
        doCheck=false;
      } ;
      v1_2_1 = buildPythonPackage {
        pname = "meson";
        version = "1.2.1";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/e5/74/a1f1c6ba14e11e0fb050d2c61a78b6db108dd38383b6c0ab51c1becbbeff/meson-1.2.1-py3-none-any.whl";
          hash="sha256-CPg/wXUT6ZzW6Cx1VMH1ivcEJSEYh/j5xzY7KpAglGI=";
        };
        doCheck=false;
      } ;
    };
    pybind11 = nixpy-custom.pybind11_2_13_1 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        cmake = cmake;
        ninja = ninja;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    setuptools-scm = {
      default = buildPythonPackage {
        pname = "setuptools-scm";
        version = "8.1.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/a0/b9/1906bfeb30f2fc13bb39bf7ddb8749784c05faadbd18a21cf141ba37bff2/setuptools_scm-8.1.0-py3-none-any.whl";
          hash="sha256-iXoyJqb9Sm6y8Gh0XklzMmGiH3Cxuyj84DOf65eNmvM=";
        };
        dependencies = with packages;
        [packaging setuptools.v74_1_1 tomli];
        doCheck=false;
      } ;
      with_toml = buildPythonPackage {
        pname = "setuptools-scm";
        version = "8.1.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/a0/b9/1906bfeb30f2fc13bb39bf7ddb8749784c05faadbd18a21cf141ba37bff2/setuptools_scm-8.1.0-py3-none-any.whl";
          hash="sha256-iXoyJqb9Sm6y8Gh0XklzMmGiH3Cxuyj84DOf65eNmvM=";
        };
        dependencies = with packages;
        [packaging setuptools.v74_1_1 tomli];
        doCheck=false;
      } ;
    };
    pkgconfig = buildPythonPackage {
      pname = "pkgconfig";
      version = "1.5.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/32/af/89487c7bbf433f4079044f3dc32f9a9f887597fe04614a37a292e373e16b/pkgconfig-1.5.5-py3-none-any.whl";
        hash="sha256-0gAju+tC7m1Cig+sbgkEYx9UWYWhDN1xogqli8R6Qgk=";
      };
      doCheck=false;
    } ;
    hatchling = buildPythonPackage {
      pname = "hatchling";
      version = "1.25.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/0c/8b/90e80904fdc24ce33f6fc6f35ebd2232fe731a8528a22008458cf197bc4d/hatchling-1.25.0-py3-none-any.whl";
        hash="sha256-tHlI5F1NlzA0WE3UyznBS2pwInzyh6t+wK15g0CKiCw=";
      };
      dependencies = with packages;
      [packaging pathspec pluggy tomli trove-classifiers];
      doCheck=false;
    } ;
    trove-classifiers = buildPythonPackage {
      pname = "trove-classifiers";
      version = "2024.7.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/0f/b0/09794439a62a7dc18bffdbf145aaf50297fd994890b11da27a13e376b947/trove_classifiers-2024.7.2-py3-none-any.whl";
        hash="sha256-zMV6M3F2RN9NrKAY5+w+9XqDXEjpah5x/AfrftrGevY=";
      };
      doCheck=false;
    } ;
    pathspec = buildPythonPackage {
      pname = "pathspec";
      version = "0.12.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/cc/20/ff623b09d963f88bfde16306a54e12ee5ea43e9b597108672ff3a408aad6/pathspec-0.12.1-py3-none-any.whl";
        hash="sha256-oNUD4TikwSOydJCk977aagHG8ojfDkqLecfrDce0zAg=";
      };
      doCheck=false;
    } ;
    pythran = buildPythonPackage {
      pname = "pythran";
      version = "0.16.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/73/32/f892675c5009cd4c1895ded3d6153476bf00adb5ad1634d03635620881f5/pythran-0.16.1.tar.gz";
        hash="sha256-hhdIwPnH1CKzJySxFLOBfYGO1Oq4bAl4GqCj986rt/k=";
      };
      build-system = with packages;
      [setuptools.v74_1_1];
      dependencies = with packages;
      [beniget gast numpy.v1_26_4 ply setuptools.v74_1_1];
      doCheck=false;
    } ;
    beniget = buildPythonPackage {
      pname = "beniget";
      version = "0.4.2.post1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/44/e4/6e8731d4d10dd09942a6f5015b2148ae612bf13e49629f33f9fade3c8253/beniget-0.4.2.post1-py3-none-any.whl";
        hash="sha256-4bM257XyriAebMIfUzSGZp8bnsy6AY3P9Zac1S8cILo=";
      };
      dependencies = with packages;
      [gast];
      doCheck=false;
    } ;
    gast = buildPythonPackage {
      pname = "gast";
      version = "0.5.5";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/60/be/fb26a2ca22f26264cf18f8ac9f42ce5400910f6c1c3cb82342b61445eb88/gast-0.5.5-py3-none-any.whl";
        hash="sha256-hEhgFdmtkJZfcxv5266UJ8hyjaDJ3R8p/37PPBS8m2g=";
      };
      doCheck=false;
    } ;
    ply = buildPythonPackage {
      pname = "ply";
      version = "3.11";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/a3/58/35da89ee790598a0700ea49b2a66594140f44dec458c07e8e3d4979137fc/ply-3.11-py2.py3-none-any.whl";
        hash="sha256-CW+bg1C2Xr0v0TRrEkUu/luWB/dIKBP/ylDCJyKoB84=";
      };
      doCheck=false;
    } ;
    build = buildPythonPackage {
      pname = "build";
      version = "1.2.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e2/03/f3c8ba0a6b6e30d7d18c40faab90807c9bb5e9a1e3b2fe2008af624a9c97/build-1.2.1-py3-none-any.whl";
        hash="sha256-deEPdnpDPZqG5Q2D9BjoPvwY7ekj7l/335O2ywMGxdQ=";
      };
      dependencies = with packages;
      [importlib-metadata packaging pyproject-hooks tomli];
      doCheck=false;
    } ;
    pyproject-hooks = buildPythonPackage {
      pname = "pyproject-hooks";
      version = "1.1.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ae/f3/431b9d5fe7d14af7a32340792ef43b8a714e7726f1d7b69cc4e8e7a3f1d7/pyproject_hooks-1.1.0-py3-none-any.whl";
        hash="sha256-fO7v6a7GOhBkwY2Tm9w63y2KoZiKUQr+wVFRV4sjKqI=";
      };
      doCheck=false;
    } ;
    importlib-metadata = buildPythonPackage {
      pname = "importlib-metadata";
      version = "8.4.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/c0/14/362d31bf1076b21e1bcdcb0dc61944822ff263937b804a79231df2774d28/importlib_metadata-8.4.0-py3-none-any.whl";
        hash="sha256-ZvNCzGrJgY/G/zQFdqzSTWW6Cz76uytKwItZiWWkovE=";
      };
      dependencies = with packages;
      [zipp];
      doCheck=false;
    } ;
    cppy = buildPythonPackage {
      pname = "cppy";
      version = "1.2.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/31/5e/b8faf2b2aeb679c0f4359fd1a4716fe90d65f72f72639413ffb95f3c3b46/cppy-1.2.1-py3-none-any.whl";
        hash="sha256-xbXqw9P0JZOgfTUnWwvCf0R7drmtjyfGLjz6KG3BmIo=";
      };
      doCheck=false;
    } ;
    cmake = nixpy-custom.cmake_3_30_2 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    versioneer = buildPythonPackage {
      pname = "versioneer";
      version = "0.29";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/b0/79/f0f1ca286b78f6f33c521a36b5cbd5bd697c0d66217d8856f443aeb9dd77/versioneer-0.29-py3-none-any.whl";
        hash="sha256-DxoTe7XWgR6Wp5uwSGeYrq6bnG78JLOJZZzrsO45bLk=";
      };
      dependencies = with packages;
      [tomli];
      doCheck=false;
    } ;
    py-cpuinfo = buildPythonPackage {
      pname = "py-cpuinfo";
      version = "9.0.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e0/a9/023730ba63db1e494a271cb018dcd361bd2c917ba7004c3e49d5daf795a2/py_cpuinfo-9.0.0-py3-none-any.whl";
        hash="sha256-hZYlvCUfZOIfB30JnUFiaJx2K11qTDyXVT1WJByWdNU=";
      };
      doCheck=false;
    } ;
    ninja = nixpy-custom.ninja_1_12_1_1 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pyobjc-framework-applicationservices = nixpy-custom.pyobjc_framework_applicationservices_9_2 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        wheel = wheel;
      };
      dependencies=with packages;
      {
        pyobjc-core = pyobjc-core;
        pyobjc-framework-cocoa = pyobjc-framework-cocoa;
        pyobjc-framework-quartz = pyobjc-framework-quartz;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pyobjc-framework-quartz = nixpy-custom.pyobjc_framework_quartz_9_2 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        wheel = wheel;
      };
      dependencies=with packages;
      {
        pyobjc-core = pyobjc-core;
        pyobjc-framework-cocoa = pyobjc-framework-cocoa;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pyobjc-core = nixpy-custom.pyobjc_core_10_3_1 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        wheel = wheel;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    pyobjc-framework-cocoa = nixpy-custom.pyobjc_framework_cocoa_9_2 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v74_1_1;
        wheel = wheel;
      };
      dependencies=with packages;
      {
        pyobjc-core = pyobjc-core;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
  };
  envs = {
    x86_64-linux = with packages;
    {
      pygments = pygments;
      foundry-meta = foundry-meta;
      foundry-core = foundry-core.with_ipython;
      tzdata = tzdata;
      pytest = pytest;
      pyparsing = pyparsing;
      trajax = trajax;
      typing-extensions = typing-extensions;
      boto3 = boto3;
      matplotlib = matplotlib;
      requests = requests;
      policy-eval = policy-eval;
      ml-collections = ml-collections;
      omegaconf = omegaconf;
      tensorstore = tensorstore;
      ffmpegio-core = ffmpegio-core;
      setproctitle = setproctitle;
      mujoco-mjx = mujoco-mjx;
      contourpy = contourpy.d12135c46;
      pynput = pynput;
      msgpack = msgpack;
      fonttools = fonttools;
      tomli = tomli;
      s3transfer = s3transfer;
      tenacity = tenacity;
      jax-cuda12-prjt = jax-cuda12-prjt;
      jsonschema = jsonschema;
      contextlib2 = contextlib2;
      fsspec = fsspec;
      nest-asyncio = nest-asyncio;
      einops = einops;
      shapely = shapely;
      docker-pycreds = docker-pycreds;
      soupsieve = soupsieve;
      protobuf = protobuf;
      platformdirs = platformdirs;
      numcodecs = numcodecs;
      certifi = certifi;
      python-dateutil = python-dateutil;
      image-classifier = image-classifier;
      iniconfig = iniconfig;
      six = six;
      sentencepiece = sentencepiece;
      wandb = wandb;
      foundry-systems = foundry-systems.d812c7c17;
      ml-dtypes = ml-dtypes.d513b6658;
      mujoco = mujoco;
      ffmpegio = ffmpegio;
      toolz = toolz;
      jupyter-core = jupyter-core;
      antlr4-python3-runtime = antlr4-python3-runtime;
      beautifulsoup4 = beautifulsoup4;
      nbformat = nbformat;
      evdev = evdev;
      gitpython = gitpython;
      kiwisolver = kiwisolver;
      pytz = pytz;
      language-model = language-model;
      idna = idna;
      zarr = zarr;
      trimesh = trimesh;
      glfw = glfw;
      pyopengl = pyopengl;
      termcolor = termcolor;
      jsonschema-specifications = jsonschema-specifications;
      flax = flax;
      click = click;
      importlib-resources = importlib-resources;
      botocore = botocore;
      setuptools = setuptools.v74_1_1;
      urllib3 = urllib3;
      humanize = humanize;
      rich = rich;
      jax = jax;
      fasteners = fasteners;
      jax-cuda12-plugin = jax-cuda12-plugin;
      pluggy = pluggy;
      cycler = cycler;
      jmespath = jmespath;
      opt-einsum = opt-einsum;
      traitlets = traitlets;
      zipp = zipp;
      scipy = scipy;
      packaging = packaging;
      llvmlite = llvmlite;
      referencing = referencing;
      fastjsonschema = fastjsonschema;
      image-diffusion = image-diffusion;
      python-xlib = python-xlib;
      pyyaml = pyyaml;
      sentry-sdk = sentry-sdk;
      numpy = numpy.v1_26_4;
      rpds-py = rpds-py;
      psutil = psutil;
      attrs = attrs;
      h5py = h5py;
      numba = numba;
      cond-diffusion-toy = cond-diffusion-toy;
      pillow = pillow;
      chex = chex;
      foundry-models = foundry-models.d2a9263c1;
      orbax-checkpoint = orbax-checkpoint.dbd2dbc15;
      smmap = smmap;
      gitdb = gitdb;
      optax = optax.d912aa156;
      jaxlib = jaxlib;
      mdurl = mdurl;
      exceptiongroup = exceptiongroup;
      plotly = plotly;
      pandas = pandas;
      markdown-it-py = markdown-it-py;
      etils = etils.with_epath;
      charset-normalizer = charset-normalizer;
      robosuite = robosuite;
      absl-py = absl-py;
      asciitree = asciitree;
    };
    powerpc64le-linux = with packages;
    {
      pandas = pandas;
      pytest = pytest;
      glfw = glfw;
      click = click;
      tensorstore = tensorstore;
      typing-extensions = typing-extensions;
      nest-asyncio = nest-asyncio;
      setuptools = setuptools.v74_1_1;
      packaging = packaging;
      boto3 = boto3;
      traitlets = traitlets;
      matplotlib = matplotlib;
      ffmpegio-core = ffmpegio-core;
      pyparsing = pyparsing;
      numpy = numpy.v1_26_4;
      cond-diffusion-toy = cond-diffusion-toy;
      asciitree = asciitree;
      jupyter-core = jupyter-core;
      etils = etils.with_epath;
      msgpack = msgpack;
      jaxlib = jaxlib;
      jsonschema-specifications = jsonschema-specifications;
      pytz = pytz;
      exceptiongroup = exceptiongroup;
      platformdirs = platformdirs;
      soupsieve = soupsieve;
      llvmlite = llvmlite;
      scipy = scipy;
      kiwisolver = kiwisolver;
      trajax = trajax;
      einops = einops;
      attrs = attrs;
      six = six;
      flax = flax;
      jax-cuda12-plugin = jax-cuda12-plugin;
      h5py = h5py;
      jsonschema = jsonschema;
      mujoco-mjx = mujoco-mjx;
      rich = rich;
      rpds-py = rpds-py;
      fastjsonschema = fastjsonschema;
      psutil = psutil;
      trimesh = trimesh;
      ml-collections = ml-collections;
      humanize = humanize;
      sentencepiece = sentencepiece;
      pluggy = pluggy;
      beautifulsoup4 = beautifulsoup4;
      opt-einsum = opt-einsum;
      pygments = pygments;
      robosuite = robosuite;
      image-classifier = image-classifier;
      setproctitle = setproctitle;
      urllib3 = urllib3;
      foundry-models = foundry-models.d66e1ada3;
      tzdata = tzdata;
      certifi = certifi;
      orbax-checkpoint = orbax-checkpoint.dcc546a6d;
      numcodecs = numcodecs;
      python-xlib = python-xlib;
      ml-dtypes = ml-dtypes.dd14ec181;
      gitdb = gitdb;
      python-dateutil = python-dateutil;
      fonttools = fonttools;
      jax-cuda12-prjt = jax-cuda12-prjt;
      termcolor = termcolor;
      tomli = tomli;
      numba = numba;
      fasteners = fasteners;
      mdurl = mdurl;
      importlib-resources = importlib-resources;
      protobuf = protobuf;
      plotly = plotly;
      markdown-it-py = markdown-it-py;
      zarr = zarr;
      pillow = pillow;
      botocore = botocore;
      nbformat = nbformat;
      chex = chex;
      image-diffusion = image-diffusion;
      idna = idna;
      charset-normalizer = charset-normalizer;
      gitpython = gitpython;
      pyopengl = pyopengl;
      foundry-meta = foundry-meta;
      wandb = wandb;
      pyyaml = pyyaml;
      pynput = pynput;
      foundry-systems = foundry-systems.d7e0cabb6;
      smmap = smmap;
      s3transfer = s3transfer;
      toolz = toolz;
      antlr4-python3-runtime = antlr4-python3-runtime;
      shapely = shapely;
      tenacity = tenacity;
      absl-py = absl-py;
      jax = jax;
      contextlib2 = contextlib2;
      omegaconf = omegaconf;
      evdev = evdev;
      zipp = zipp;
      docker-pycreds = docker-pycreds;
      iniconfig = iniconfig;
      policy-eval = policy-eval;
      fsspec = fsspec;
      optax = optax.d1f799c17;
      cycler = cycler;
      jmespath = jmespath;
      mujoco = mujoco;
      referencing = referencing;
      ffmpegio = ffmpegio;
      sentry-sdk = sentry-sdk;
      language-model = language-model;
      foundry-core = foundry-core.with_ipython;
      requests = requests;
      contourpy = contourpy.d5c0bade7;
    };
    aarch64-darwin = with packages;
    {
      chex = chex;
      jmespath = jmespath;
      shapely = shapely;
      pyobjc-framework-cocoa = pyobjc-framework-cocoa;
      foundry-meta = foundry-meta;
      omegaconf = omegaconf;
      pyopengl = pyopengl;
      foundry-core = foundry-core.with_ipython;
      jaxlib = jaxlib;
      termcolor = termcolor;
      humanize = humanize;
      exceptiongroup = exceptiongroup;
      numpy = numpy.v1_26_4;
      markdown-it-py = markdown-it-py;
      attrs = attrs;
      rpds-py = rpds-py;
      typing-extensions = typing-extensions;
      ml-dtypes = ml-dtypes.d618fa062;
      idna = idna;
      fasteners = fasteners;
      cond-diffusion-toy = cond-diffusion-toy;
      referencing = referencing;
      h5py = h5py;
      setuptools = setuptools.v74_1_1;
      contourpy = contourpy.dda94ea09;
      ffmpegio-core = ffmpegio-core;
      mujoco = mujoco;
      zipp = zipp;
      psutil = psutil;
      einops = einops;
      boto3 = boto3;
      trajax = trajax;
      setproctitle = setproctitle;
      traitlets = traitlets;
      packaging = packaging;
      ffmpegio = ffmpegio;
      nest-asyncio = nest-asyncio;
      opt-einsum = opt-einsum;
      ml-collections = ml-collections;
      pygments = pygments;
      numba = numba;
      plotly = plotly;
      image-classifier = image-classifier;
      robosuite = robosuite;
      trimesh = trimesh;
      kiwisolver = kiwisolver;
      jax = jax;
      certifi = certifi;
      pyobjc-framework-applicationservices = pyobjc-framework-applicationservices;
      nbformat = nbformat;
      jsonschema = jsonschema;
      iniconfig = iniconfig;
      llvmlite = llvmlite;
      policy-eval = policy-eval;
      sentry-sdk = sentry-sdk;
      tenacity = tenacity;
      pyobjc-core = pyobjc-core;
      importlib-resources = importlib-resources;
      msgpack = msgpack;
      fonttools = fonttools;
      orbax-checkpoint = orbax-checkpoint.d710ea424;
      gitdb = gitdb;
      urllib3 = urllib3;
      python-dateutil = python-dateutil;
      requests = requests;
      cycler = cycler;
      jsonschema-specifications = jsonschema-specifications;
      language-model = language-model;
      glfw = glfw;
      scipy = scipy;
      jupyter-core = jupyter-core;
      six = six;
      pytz = pytz;
      matplotlib = matplotlib;
      sentencepiece = sentencepiece;
      foundry-models = foundry-models.dde5f89be;
      fsspec = fsspec;
      botocore = botocore;
      toolz = toolz;
      rich = rich;
      soupsieve = soupsieve;
      asciitree = asciitree;
      tomli = tomli;
      pyyaml = pyyaml;
      tzdata = tzdata;
      click = click;
      tensorstore = tensorstore;
      wandb = wandb;
      pillow = pillow;
      foundry-systems = foundry-systems.d25a40d24;
      platformdirs = platformdirs;
      pandas = pandas;
      optax = optax.d7821d309;
      mujoco-mjx = mujoco-mjx;
      zarr = zarr;
      absl-py = absl-py;
      etils = etils.with_epath;
      gitpython = gitpython;
      image-diffusion = image-diffusion;
      pynput = pynput;
      numcodecs = numcodecs;
      fastjsonschema = fastjsonschema;
      antlr4-python3-runtime = antlr4-python3-runtime;
      pluggy = pluggy;
      charset-normalizer = charset-normalizer;
      pyparsing = pyparsing;
      pytest = pytest;
      smmap = smmap;
      mdurl = mdurl;
      pyobjc-framework-quartz = pyobjc-framework-quartz;
      contextlib2 = contextlib2;
      protobuf = protobuf;
      beautifulsoup4 = beautifulsoup4;
      s3transfer = s3transfer;
      docker-pycreds = docker-pycreds;
      flax = flax;
    };
  };
  env = envs.${
    nixpkgs.system
  };
}