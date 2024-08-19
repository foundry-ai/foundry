{buildPythonPackage, fetchurl, nixpkgs, python, nixpy-custom ? {}}: rec {
  packages = rec {
    stanza-meta = {
      powerpc64le-linux = buildPythonPackage {
        pname = "stanza-meta";
        version = "0.1.0";
        format="pyproject";
        src = ./.;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.d8c05cd08 cond-diffusion image-classifier language-model wandb];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "stanza-meta";
        version = "0.1.0";
        format="pyproject";
        src = ./.;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.df8630732 cond-diffusion image-classifier language-model wandb];
        doCheck=false;
      } ;
      aarch64-darwin = buildPythonPackage {
        pname = "stanza-meta";
        version = "0.1.0";
        format="pyproject";
        src = ./.;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.d01969111 cond-diffusion image-classifier language-model wandb];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    stanza = {
      with_docs_ipython = {
        powerpc64le-linux = buildPythonPackage {
          pname = "stanza";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/stanza;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d4eb27649 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
          doCheck=false;
        } ;
        x86_64-linux = buildPythonPackage {
          pname = "stanza";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/stanza;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d4f5774f3 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
          doCheck=false;
        } ;
        aarch64-darwin = buildPythonPackage {
          pname = "stanza";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/stanza;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.dedf1c546 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
          doCheck=false;
        } ;
      }.${
        nixpkgs.system
      };
      default = {
        x86_64-linux = buildPythonPackage {
          pname = "stanza";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/stanza;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d4f5774f3 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
          doCheck=false;
        } ;
        aarch64-darwin = buildPythonPackage {
          pname = "stanza";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/stanza;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.dedf1c546 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
          doCheck=false;
        } ;
        powerpc64le-linux = buildPythonPackage {
          pname = "stanza";
          version = "0.1.0";
          format="pyproject";
          src = ./packages/stanza;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [jax rich flax optax.d4eb27649 pandas chex numpy.v1_26_4 ffmpegio einops matplotlib plotly nbformat beautifulsoup4 trajax zarr mujoco-mjx shapely robosuite sentencepiece h5py];
          doCheck=false;
        } ;
      }.${
        nixpkgs.system
      };
    };
    stanza-models = {
      powerpc64le-linux = {
        d8c05cd08 = buildPythonPackage {
          pname = "stanza-models";
          version = "0.1.0";
          format="pyproject";
          src = ./projects/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [stanza.with_docs_ipython];
          doCheck=false;
        } ;
        d3f268798 = buildPythonPackage {
          pname = "stanza-models";
          version = "0.1.0";
          format="pyproject";
          src = ./projects/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [stanza.default];
          doCheck=false;
        } ;
      };
      x86_64-linux = {
        dfc2ad4d3 = buildPythonPackage {
          pname = "stanza-models";
          version = "0.1.0";
          format="pyproject";
          src = ./projects/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [stanza.default];
          doCheck=false;
        } ;
        df8630732 = buildPythonPackage {
          pname = "stanza-models";
          version = "0.1.0";
          format="pyproject";
          src = ./projects/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [stanza.with_docs_ipython];
          doCheck=false;
        } ;
      };
      aarch64-darwin = {
        d01969111 = buildPythonPackage {
          pname = "stanza-models";
          version = "0.1.0";
          format="pyproject";
          src = ./projects/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [stanza.with_docs_ipython];
          doCheck=false;
        } ;
        d20b0617b = buildPythonPackage {
          pname = "stanza-models";
          version = "0.1.0";
          format="pyproject";
          src = ./projects/models;
          build-system = with packages;
          [pdm-backend];
          dependencies = with packages;
          [stanza.default];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    cond-diffusion = buildPythonPackage {
      pname = "cond-diffusion";
      version = "0.1.0";
      format="pyproject";
      src = ./projects/cond-diffusion;
      build-system = with packages;
      [pdm-backend];
      dependencies = with packages;
      [stanza.with_docs_ipython];
      doCheck=false;
    } ;
    image-classifier = {
      aarch64-darwin = buildPythonPackage {
        pname = "image-classifier";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-classifier;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.d01969111];
        doCheck=false;
      } ;
      powerpc64le-linux = buildPythonPackage {
        pname = "image-classifier";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/image-classifier;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.d8c05cd08];
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
        [stanza.with_docs_ipython stanza-models.df8630732];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    language-model = {
      x86_64-linux = buildPythonPackage {
        pname = "language-model";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/language-model;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.df8630732];
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
        [stanza.with_docs_ipython stanza-models.d01969111];
        doCheck=false;
      } ;
      powerpc64le-linux = buildPythonPackage {
        pname = "language-model";
        version = "0.1.0";
        format="pyproject";
        src = ./projects/language-model;
        build-system = with packages;
        [pdm-backend];
        dependencies = with packages;
        [stanza.with_docs_ipython stanza-models.d8c05cd08];
        doCheck=false;
      } ;
    }.${
      nixpkgs.system
    };
    jax = {
      x86_64-linux = nixpy-custom.jax_0_4_28 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          scipy = scipy;
          opt-einsum = opt-einsum;
          ml-dtypes = ml-dtypes.d869c1b01;
          jaxlib = jaxlib;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      powerpc64le-linux = nixpy-custom.jax_0_4_28 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          scipy = scipy;
          opt-einsum = opt-einsum;
          ml-dtypes = ml-dtypes.d2f76c2b2;
          jaxlib = jaxlib;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      aarch64-darwin = nixpy-custom.jax_0_4_28 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          scipy = scipy;
          opt-einsum = opt-einsum;
          ml-dtypes = ml-dtypes.d3d465209;
          jaxlib = jaxlib;
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
    matplotlib = {
      powerpc64le-linux = nixpy-custom.matplotlib_3_9_1 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          meson-python = meson-python.v0_15_0;
          pybind11 = pybind11.v2_13_4;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          contourpy = contourpy.d363fff9a;
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
          pybind11 = pybind11.v2_13_4;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          contourpy = contourpy.df33f8675;
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
          pybind11 = pybind11.v2_13_4;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          contourpy = contourpy.dfcb691a8;
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
      [setuptools.v72_2_0];
      dependencies = with packages;
      [absl-py jax jaxlib ml-collections scipy];
      doCheck=false;
    } ;
    shapely = nixpy-custom.shapely_2_0_5 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v72_2_0;
        cython = cython.v3_0_11;
        numpy = numpy.v2_0_1;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
      };
      dependencies=with packages;
      {
        numpy = numpy.v1_26_4;
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    scipy = nixpy-custom.scipy_1_14_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        meson-python = meson-python.v0_15_0;
        cython = cython.v3_0_11;
        pybind11 = pybind11.v2_12_0;
        pythran = pythran;
        numpy = numpy.v2_0_1;
        wheel = wheel;
        setuptools = setuptools.v72_2_0;
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
        dbaf8c40f = nixpy-custom.ml_dtypes_0_4_0 {
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
        d3d465209 = nixpy-custom.ml_dtypes_0_4_0 {
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
        d2f76c2b2 = nixpy-custom.ml_dtypes_0_4_0 {
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
        d2ddc4d0c = nixpy-custom.ml_dtypes_0_4_0 {
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
        d869c1b01 = nixpy-custom.ml_dtypes_0_4_0 {
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
        d6450fc4b = nixpy-custom.ml_dtypes_0_4_0 {
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
    }.${
      nixpkgs.system
    };
    jaxlib = {
      powerpc64le-linux = nixpy-custom.jaxlib_0_4_28 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d2f76c2b2;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      x86_64-linux = nixpy-custom.jaxlib_0_4_28 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d869c1b01;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      aarch64-darwin = nixpy-custom.jaxlib_0_4_28 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
        };
        dependencies=with packages;
        {
          scipy = scipy;
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d3d465209;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
    };
    kiwisolver = nixpy-custom.kiwisolver_1_4_5 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
      [setuptools.v72_2_0];
      dependencies = with packages;
      [absl-py contextlib2 pyyaml six];
      doCheck=false;
    } ;
    numba = nixpy-custom.numba_0_60_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
        cmake = cmake;
        pybind11 = pybind11.v2_13_4;
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
    pyyaml = buildPythonPackage {
      pname = "pyyaml";
      version = "6.0.2";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/54/ed/79a089b6be93607fa5cdaedf301d7dfb23af5f25c398d5ead2525b063e17/pyyaml-6.0.2.tar.gz";
        hash="sha256-1YTZ7JGtZYYcwI1C6DQyTviQoILlkQN6vhFIUP97vD4=";
      };
      build-system = with packages;
      [cython.v3_0_11 setuptools.v72_2_0 wheel];
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
        setuptools = setuptools.v72_2_0;
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
        version = "1.9.2";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/a0/f4/305f3ea85aecd23422c606c179fb6d00bd7d255b10d55b4c797a3a680144/etils-1.9.2-py3-none-any.whl";
          hash="sha256-7Ned4fv+qbDWkkdWz6kisF7TNgxFzyFwdn2kvuAAHSA=";
        };
        dependencies = with packages;
        [fsspec importlib-resources typing-extensions zipp];
        doCheck=false;
      } ;
      with_epath_epy = buildPythonPackage {
        pname = "etils";
        version = "1.9.2";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/a0/f4/305f3ea85aecd23422c606c179fb6d00bd7d255b10d55b4c797a3a680144/etils-1.9.2-py3-none-any.whl";
          hash="sha256-7Ned4fv+qbDWkkdWz6kisF7TNgxFzyFwdn2kvuAAHSA=";
        };
        dependencies = with packages;
        [fsspec importlib-resources typing-extensions zipp];
        doCheck=false;
      } ;
      with_epy = buildPythonPackage {
        pname = "etils";
        version = "1.9.2";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/a0/f4/305f3ea85aecd23422c606c179fb6d00bd7d255b10d55b4c797a3a680144/etils-1.9.2-py3-none-any.whl";
          hash="sha256-7Ned4fv+qbDWkkdWz6kisF7TNgxFzyFwdn2kvuAAHSA=";
        };
        dependencies = with packages;
        [typing-extensions];
        doCheck=false;
      } ;
    };
    glfw = nixpy-custom.glfw_2_7_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
      version = "6.4.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/13/da/ec59848cc9d785fc0f55823efd84d213418e14e2c97ae87c15b5ac46a6ab/importlib_resources-6.4.2-py3-none-any.whl";
        hash="sha256-i7qMVKijr6oUGZEIRfom69cG3HFt0gjZsVi0tpZvXFw=";
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
      version = "3.20.0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/da/cc/b9958af9f9c86b51f846d8487440af495ecf19b16e426fce1ed0b0796175/zipp-3.20.0-py3-none-any.whl";
        hash="sha256-WNphaL6J8L5ZvrGU2hJQUW/aoGLM69MBJ6xl0wBF4Q0=";
      };
      doCheck=false;
    } ;
    rich = buildPythonPackage {
      pname = "rich";
      version = "13.7.1";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/87/67/a37f6214d0e9fe57f6ae54b2956d550ca8365857f42a1ce0392bb21d9410/rich-13.7.1-py3-none-any.whl";
        hash="sha256-TtuuMU9Z60gvVOnjC/ANMzUKqpT0v81OnjEQ5k0NciI=";
      };
      dependencies = with packages;
      [markdown-it-py pygments];
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
      [setuptools.v72_2_0];
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
    wandb = buildPythonPackage {
      pname = "wandb";
      version = "0.17.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ab/5f/502508715966a471e9c06ab26ecdc427e5f4f593d3828acd6250d6bf3da2/wandb-0.17.7-py3-none-any.whl";
        hash="sha256-QvN9fU8ZNPxaMyM74d4PDI47v/BKRAOzrGAw5XfMhOE=";
      };
      dependencies = with packages;
      [click docker-pycreds gitpython platformdirs protobuf psutil pyyaml requests sentry-sdk setproctitle setuptools.v72_2_0];
      doCheck=false;
    } ;
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
    psutil = nixpy-custom.psutil_6_0_0 {
      buildPythonPackage=buildPythonPackage;
      build-system=with packages;
      {
        setuptools = setuptools.v72_2_0;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    setproctitle = buildPythonPackage {
      pname = "setproctitle";
      version = "1.3.3";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/ff/e1/b16b16a1aa12174349d15b73fd4b87e641a8ae3fb1163e80938dbbf6ae98/setproctitle-1.3.3.tar.gz";
        hash="sha256-yRPhUefqAVZ4N/8DeiPKh0AZKIAZi3+7kLFtGBYHyq4=";
      };
      build-system = with packages;
      [setuptools.v72_2_0];
      doCheck=false;
    } ;
    setuptools = {
      v72_2_0 = buildPythonPackage {
        pname = "setuptools";
        version = "72.2.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/6e/ec/06715d912351edc453e37f93f3fc80dcffd5ca0e70386c87529aca296f05/setuptools-72.2.0-py3-none-any.whl";
          hash="sha256-8R3ZS3uuOhVqlewVHyTkY3+0+hnIeOTRkb+4stgnKMQ=";
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
    ffmpegio = buildPythonPackage {
      pname = "ffmpegio";
      version = "0.10.0.post0";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/52/fd/b85fec7fc96cb6e0aa8b307e1ec0de9986826b3574d67b60762868e7feea/ffmpegio-0.10.0.post0-py3-none-any.whl";
        hash="sha256-br1OgDQ+cnqjGQQRHrXFc7QXWwQ57kEsZSs5LVnM+gU=";
      };
      dependencies = with packages;
      [ffmpegio-core numpy.v1_26_4];
      doCheck=false;
    } ;
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
      version = "4.4.6";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/3c/b5/5772d5e3d059f9949f4f4cf96248c6e69d035f50c548c2c145ed6eb980f0/trimesh-4.4.6-py3-none-any.whl";
        hash="sha256-95W6x2DviHtdhbKJ6cUn+Mkp0ulEKWiFhpBWYAV6FWI=";
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
    flax = {
      aarch64-darwin = buildPythonPackage {
        pname = "flax";
        version = "0.8.5";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/1c/a9/6978d2547b1d8ca0ce75b534c0ba5c60e8e7b918c5c1800225aa0169cb7f/flax-0.8.5-py3-none-any.whl";
          hash="sha256-yW5G0cSKMA0BDr9cSEbxY73XrMbv/1/yv7HLWwiqZdg=";
        };
        dependencies = with packages;
        [jax msgpack numpy.v1_26_4 optax.dedf1c546 orbax-checkpoint.d836b5ef8 pyyaml rich tensorstore typing-extensions];
        doCheck=false;
      } ;
      x86_64-linux = buildPythonPackage {
        pname = "flax";
        version = "0.8.5";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/1c/a9/6978d2547b1d8ca0ce75b534c0ba5c60e8e7b918c5c1800225aa0169cb7f/flax-0.8.5-py3-none-any.whl";
          hash="sha256-yW5G0cSKMA0BDr9cSEbxY73XrMbv/1/yv7HLWwiqZdg=";
        };
        dependencies = with packages;
        [jax msgpack numpy.v1_26_4 optax.d4f5774f3 orbax-checkpoint.dbd5f266f pyyaml rich tensorstore typing-extensions];
        doCheck=false;
      } ;
      powerpc64le-linux = buildPythonPackage {
        pname = "flax";
        version = "0.8.5";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/1c/a9/6978d2547b1d8ca0ce75b534c0ba5c60e8e7b918c5c1800225aa0169cb7f/flax-0.8.5-py3-none-any.whl";
          hash="sha256-yW5G0cSKMA0BDr9cSEbxY73XrMbv/1/yv7HLWwiqZdg=";
        };
        dependencies = with packages;
        [jax msgpack numpy.v1_26_4 optax.d4eb27649 orbax-checkpoint.d9e3d4bc9 pyyaml rich tensorstore typing-extensions];
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
        setuptools = setuptools.v72_2_0;
      };
      dependencies={
      };
      fetchurl=fetchurl;
      nixpkgs=nixpkgs;
      python=python;
    };
    orbax-checkpoint = {
      aarch64-darwin = {
        d836b5ef8 = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.5.23";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/6a/2e/0a2efe062084f477113ea07ba2e0e7298b05b3ca8c5d3568bf5ce6991b6e/orbax_checkpoint-0.5.23-py3-none-any.whl";
            hash="sha256-DecT4kKuKVrGEUdv+4MIfNsKrSIefVS8H+qk29ExjEE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
        d1846cbcc = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.5.23";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/6a/2e/0a2efe062084f477113ea07ba2e0e7298b05b3ca8c5d3568bf5ce6991b6e/orbax_checkpoint-0.5.23-py3-none-any.whl";
            hash="sha256-DecT4kKuKVrGEUdv+4MIfNsKrSIefVS8H+qk29ExjEE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath_epy jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
      };
      x86_64-linux = {
        d7f2a7bee = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.5.23";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/6a/2e/0a2efe062084f477113ea07ba2e0e7298b05b3ca8c5d3568bf5ce6991b6e/orbax_checkpoint-0.5.23-py3-none-any.whl";
            hash="sha256-DecT4kKuKVrGEUdv+4MIfNsKrSIefVS8H+qk29ExjEE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath_epy jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
        dbd5f266f = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.5.23";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/6a/2e/0a2efe062084f477113ea07ba2e0e7298b05b3ca8c5d3568bf5ce6991b6e/orbax_checkpoint-0.5.23-py3-none-any.whl";
            hash="sha256-DecT4kKuKVrGEUdv+4MIfNsKrSIefVS8H+qk29ExjEE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        df77811cc = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.5.23";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/6a/2e/0a2efe062084f477113ea07ba2e0e7298b05b3ca8c5d3568bf5ce6991b6e/orbax_checkpoint-0.5.23-py3-none-any.whl";
            hash="sha256-DecT4kKuKVrGEUdv+4MIfNsKrSIefVS8H+qk29ExjEE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath_epy jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
        d9e3d4bc9 = buildPythonPackage {
          pname = "orbax-checkpoint";
          version = "0.5.23";
          format="wheel";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/6a/2e/0a2efe062084f477113ea07ba2e0e7298b05b3ca8c5d3568bf5ce6991b6e/orbax_checkpoint-0.5.23-py3-none-any.whl";
            hash="sha256-DecT4kKuKVrGEUdv+4MIfNsKrSIefVS8H+qk29ExjEE=";
          };
          dependencies = with packages;
          [absl-py etils.with_epath jax jaxlib msgpack nest-asyncio numpy.v1_26_4 protobuf pyyaml tensorstore typing-extensions];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
    tensorstore = {
      aarch64-darwin = nixpy-custom.tensorstore_0_1_64 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d3d465209;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      x86_64-linux = nixpy-custom.tensorstore_0_1_64 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d869c1b01;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
      powerpc64le-linux = nixpy-custom.tensorstore_0_1_64 {
        buildPythonPackage=buildPythonPackage;
        build-system=with packages;
        {
          setuptools = setuptools.v72_2_0;
          wheel = wheel;
          setuptools-scm = setuptools-scm.default;
          numpy = numpy.v2_0_1;
        };
        dependencies=with packages;
        {
          numpy = numpy.v1_26_4;
          ml-dtypes = ml-dtypes.d2f76c2b2;
        };
        fetchurl=fetchurl;
        nixpkgs=nixpkgs;
        python=python;
      };
    }.${
      nixpkgs.system
    };
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
    optax = {
      aarch64-darwin = {
        d495cbdc8 = buildPythonPackage {
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
        dedf1c546 = buildPythonPackage {
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
      x86_64-linux = {
        d4f5774f3 = buildPythonPackage {
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
        de94c5a1d = buildPythonPackage {
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
      };
      powerpc64le-linux = {
        d4eb27649 = buildPythonPackage {
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
        d9c2a4279 = buildPythonPackage {
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
      };
    }.${
      nixpkgs.system
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
      [packaging tenacity];
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
      [cython.v3_0_11 numpy.v1_26_4 py-cpuinfo setuptools-scm.with_toml setuptools.v72_2_0];
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
    contourpy = {
      x86_64-linux = {
        df33f8675 = buildPythonPackage {
          pname = "contourpy";
          version = "1.2.1";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
            hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11.v2_13_4];
          dependencies = with packages;
          [numpy.v1_26_4];
          doCheck=false;
        } ;
        dcd244290 = buildPythonPackage {
          pname = "contourpy";
          version = "1.2.1";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
            hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11.v2_13_4];
          dependencies = with packages;
          [numpy.v2_0_1];
          doCheck=false;
        } ;
      };
      aarch64-darwin = {
        d2d52fd33 = buildPythonPackage {
          pname = "contourpy";
          version = "1.2.1";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
            hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11.v2_13_4];
          dependencies = with packages;
          [numpy.v2_0_1];
          doCheck=false;
        } ;
        dfcb691a8 = buildPythonPackage {
          pname = "contourpy";
          version = "1.2.1";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
            hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11.v2_13_4];
          dependencies = with packages;
          [numpy.v1_26_4];
          doCheck=false;
        } ;
      };
      powerpc64le-linux = {
        d363fff9a = buildPythonPackage {
          pname = "contourpy";
          version = "1.2.1";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
            hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11.v2_13_4];
          dependencies = with packages;
          [numpy.v1_26_4];
          doCheck=false;
        } ;
        d7562710b = buildPythonPackage {
          pname = "contourpy";
          version = "1.2.1";
          format="pyproject";
          src = fetchurl {
            url="https://files.pythonhosted.org/packages/8d/9e/e4786569b319847ffd98a8326802d5cf8a5500860dbfc2df1f0f4883ed99/contourpy-1.2.1.tar.gz";
            hash="sha256-TYkIs77hyInlR4Z8pM3FTlq2vm0+B4VWgUoiRX9JQjw=";
          };
          build-system = with packages;
          [meson-python.v0_15_0 meson.v1_5_1 pybind11.v2_13_4];
          dependencies = with packages;
          [numpy.v2_0_1];
          doCheck=false;
        } ;
      };
    }.${
      nixpkgs.system
    };
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
      version = "3.1.2";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/9d/ea/6d76df31432a0e6fdf81681a895f009a4bb47b3c39036db3e1b528191d52/pyparsing-3.1.2-py3-none-any.whl";
        hash="sha256-+dt1kRgB7XeP5hu2Qwef+GYBrKmfyuY0WqZykgOPt0I=";
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
    protobuf = buildPythonPackage {
      pname = "protobuf";
      version = "5.27.3";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e1/94/d77bd282d3d53155147166c2bbd156f540009b0d7be24330f76286668b90/protobuf-5.27.3-py3-none-any.whl";
        hash="sha256-hXLGUz5UTr9omcNg6R1ry77iVJJRZD0yxSz4pd4pW6U=";
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
        setuptools = setuptools.v72_2_0;
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
      version = "3.7";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/e5/3e/741d8c82801c347547f8a2a06aa57dbb1992be9e948df2ea0eda2c8b79e8/idna-3.7-py3-none-any.whl";
        hash="sha256-gv7h/Hit1DSS06GJi/ptipBMyX2EJ/aD7Y55jQd2GqA=";
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
      version = "2024.7.4";
      format="wheel";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/1c/d5/c84e1a17bf61d4df64ca866a1c9a913874b4e9bdc131ec689a0ad013fb36/certifi-2024.7.4-py3-none-any.whl";
        hash="sha256-wZjiGxKJwquF7k5nu0tO8+rQiSBZkBqNW2IvJKEQHpA=";
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
        [packaging setuptools.v72_2_0 tomli];
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
        [packaging setuptools.v72_2_0 tomli];
        doCheck=false;
      } ;
    };
    pybind11 = {
      v2_13_4 = buildPythonPackage {
        pname = "pybind11";
        version = "2.13.4";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/d9/0d/639dbbe0122b9359185877da57ad421a2cac9ab3a5c766833963354c4331/pybind11-2.13.4-py3-none-any.whl";
          hash="sha256-WTLWPVcLOhLs4vZnits4RswcIp3B+FGKRtW1QPJA+Vk=";
        };
        doCheck=false;
      } ;
      v2_12_0 = buildPythonPackage {
        pname = "pybind11";
        version = "2.12.0";
        format="wheel";
        src = fetchurl {
          url="https://files.pythonhosted.org/packages/26/55/e776489172f576b782e616f58273e1f3de56a91004b0d20504169dd345af/pybind11-2.12.0-py3-none-any.whl";
          hash="sha256-341guU+ecU2BAT2yMzk9Qw6/nzVRZCuCKRzxsU0a/b0=";
        };
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
    pythran = buildPythonPackage {
      pname = "pythran";
      version = "0.16.1";
      format="pyproject";
      src = fetchurl {
        url="https://files.pythonhosted.org/packages/73/32/f892675c5009cd4c1895ded3d6153476bf00adb5ad1634d03635620881f5/pythran-0.16.1.tar.gz";
        hash="sha256-hhdIwPnH1CKzJySxFLOBfYGO1Oq4bAl4GqCj986rt/k=";
      };
      build-system = with packages;
      [setuptools.v72_2_0];
      dependencies = with packages;
      [beniget gast numpy.v2_0_1 ply setuptools.v72_2_0];
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
        setuptools = setuptools.v72_2_0;
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
      python-xlib = python-xlib;
      gitdb = gitdb;
      trimesh = trimesh;
      packaging = packaging;
      wandb = wandb;
      zipp = zipp;
      contourpy = contourpy.df33f8675;
      fasteners = fasteners;
      platformdirs = platformdirs;
      pillow = pillow;
      fonttools = fonttools;
      psutil = psutil;
      jaxlib = jaxlib;
      importlib-resources = importlib-resources;
      attrs = attrs;
      pytz = pytz;
      python-dateutil = python-dateutil;
      toolz = toolz;
      orbax-checkpoint = orbax-checkpoint.dbd5f266f;
      sentencepiece = sentencepiece;
      robosuite = robosuite;
      h5py = h5py;
      typing-extensions = typing-extensions;
      glfw = glfw;
      numcodecs = numcodecs;
      termcolor = termcolor;
      msgpack = msgpack;
      docker-pycreds = docker-pycreds;
      gitpython = gitpython;
      referencing = referencing;
      mujoco-mjx = mujoco-mjx;
      nbformat = nbformat;
      plotly = plotly;
      traitlets = traitlets;
      pyyaml = pyyaml;
      rich = rich;
      evdev = evdev;
      ffmpegio-core = ffmpegio-core;
      flax = flax;
      jsonschema = jsonschema;
      cycler = cycler;
      jsonschema-specifications = jsonschema-specifications;
      pygments = pygments;
      certifi = certifi;
      tensorstore = tensorstore;
      cond-diffusion = cond-diffusion;
      llvmlite = llvmlite;
      beautifulsoup4 = beautifulsoup4;
      stanza-models = stanza-models.df8630732;
      fastjsonschema = fastjsonschema;
      rpds-py = rpds-py;
      trajax = trajax;
      scipy = scipy;
      setproctitle = setproctitle;
      jupyter-core = jupyter-core;
      shapely = shapely;
      zarr = zarr;
      pandas = pandas;
      click = click;
      stanza-meta = stanza-meta;
      fsspec = fsspec;
      ml-dtypes = ml-dtypes.d869c1b01;
      ffmpegio = ffmpegio;
      contextlib2 = contextlib2;
      six = six;
      ml-collections = ml-collections;
      protobuf = protobuf;
      numba = numba;
      charset-normalizer = charset-normalizer;
      absl-py = absl-py;
      stanza = stanza.with_docs_ipython;
      einops = einops;
      soupsieve = soupsieve;
      kiwisolver = kiwisolver;
      jax = jax;
      etils = etils.with_epath;
      opt-einsum = opt-einsum;
      matplotlib = matplotlib;
      numpy = numpy.v1_26_4;
      chex = chex;
      requests = requests;
      idna = idna;
      pluggy = pluggy;
      markdown-it-py = markdown-it-py;
      mujoco = mujoco;
      tenacity = tenacity;
      mdurl = mdurl;
      sentry-sdk = sentry-sdk;
      asciitree = asciitree;
      tzdata = tzdata;
      optax = optax.d4f5774f3;
      pyopengl = pyopengl;
      language-model = language-model;
      pyparsing = pyparsing;
      pynput = pynput;
      image-classifier = image-classifier;
      smmap = smmap;
      setuptools = setuptools.v72_2_0;
      nest-asyncio = nest-asyncio;
      urllib3 = urllib3;
    };
    powerpc64le-linux = with packages;
    {
      psutil = psutil;
      soupsieve = soupsieve;
      pluggy = pluggy;
      pillow = pillow;
      setuptools = setuptools.v72_2_0;
      toolz = toolz;
      trimesh = trimesh;
      sentry-sdk = sentry-sdk;
      ml-collections = ml-collections;
      tensorstore = tensorstore;
      trajax = trajax;
      cond-diffusion = cond-diffusion;
      certifi = certifi;
      matplotlib = matplotlib;
      stanza-meta = stanza-meta;
      rich = rich;
      h5py = h5py;
      nbformat = nbformat;
      robosuite = robosuite;
      wandb = wandb;
      urllib3 = urllib3;
      asciitree = asciitree;
      python-dateutil = python-dateutil;
      zarr = zarr;
      zipp = zipp;
      pyyaml = pyyaml;
      stanza-models = stanza-models.d8c05cd08;
      evdev = evdev;
      contourpy = contourpy.d363fff9a;
      ffmpegio = ffmpegio;
      six = six;
      importlib-resources = importlib-resources;
      packaging = packaging;
      pytz = pytz;
      shapely = shapely;
      mdurl = mdurl;
      flax = flax;
      sentencepiece = sentencepiece;
      glfw = glfw;
      smmap = smmap;
      absl-py = absl-py;
      jaxlib = jaxlib;
      language-model = language-model;
      gitdb = gitdb;
      pandas = pandas;
      msgpack = msgpack;
      jupyter-core = jupyter-core;
      ffmpegio-core = ffmpegio-core;
      contextlib2 = contextlib2;
      markdown-it-py = markdown-it-py;
      beautifulsoup4 = beautifulsoup4;
      optax = optax.d4eb27649;
      typing-extensions = typing-extensions;
      cycler = cycler;
      tzdata = tzdata;
      stanza = stanza.with_docs_ipython;
      numba = numba;
      protobuf = protobuf;
      nest-asyncio = nest-asyncio;
      python-xlib = python-xlib;
      termcolor = termcolor;
      pynput = pynput;
      requests = requests;
      numcodecs = numcodecs;
      traitlets = traitlets;
      kiwisolver = kiwisolver;
      fsspec = fsspec;
      jsonschema = jsonschema;
      click = click;
      setproctitle = setproctitle;
      numpy = numpy.v1_26_4;
      charset-normalizer = charset-normalizer;
      image-classifier = image-classifier;
      idna = idna;
      plotly = plotly;
      pygments = pygments;
      opt-einsum = opt-einsum;
      pyopengl = pyopengl;
      pyparsing = pyparsing;
      fonttools = fonttools;
      fastjsonschema = fastjsonschema;
      tenacity = tenacity;
      platformdirs = platformdirs;
      scipy = scipy;
      docker-pycreds = docker-pycreds;
      fasteners = fasteners;
      rpds-py = rpds-py;
      gitpython = gitpython;
      referencing = referencing;
      ml-dtypes = ml-dtypes.d2f76c2b2;
      etils = etils.with_epath;
      orbax-checkpoint = orbax-checkpoint.d9e3d4bc9;
      llvmlite = llvmlite;
      attrs = attrs;
      mujoco-mjx = mujoco-mjx;
      einops = einops;
      chex = chex;
      jsonschema-specifications = jsonschema-specifications;
      mujoco = mujoco;
      jax = jax;
    };
    aarch64-darwin = with packages;
    {
      fastjsonschema = fastjsonschema;
      gitdb = gitdb;
      python-dateutil = python-dateutil;
      trajax = trajax;
      etils = etils.with_epath;
      termcolor = termcolor;
      zipp = zipp;
      glfw = glfw;
      pynput = pynput;
      jupyter-core = jupyter-core;
      pillow = pillow;
      fonttools = fonttools;
      numcodecs = numcodecs;
      robosuite = robosuite;
      setuptools = setuptools.v72_2_0;
      stanza = stanza.with_docs_ipython;
      contextlib2 = contextlib2;
      jax = jax;
      pyyaml = pyyaml;
      markdown-it-py = markdown-it-py;
      tenacity = tenacity;
      plotly = plotly;
      zarr = zarr;
      docker-pycreds = docker-pycreds;
      mujoco = mujoco;
      ffmpegio-core = ffmpegio-core;
      requests = requests;
      shapely = shapely;
      rpds-py = rpds-py;
      chex = chex;
      stanza-meta = stanza-meta;
      tensorstore = tensorstore;
      mdurl = mdurl;
      toolz = toolz;
      optax = optax.dedf1c546;
      gitpython = gitpython;
      contourpy = contourpy.dfcb691a8;
      numba = numba;
      trimesh = trimesh;
      matplotlib = matplotlib;
      traitlets = traitlets;
      importlib-resources = importlib-resources;
      jsonschema = jsonschema;
      rich = rich;
      cond-diffusion = cond-diffusion;
      sentry-sdk = sentry-sdk;
      sentencepiece = sentencepiece;
      ml-collections = ml-collections;
      ffmpegio = ffmpegio;
      pytz = pytz;
      pyobjc-core = pyobjc-core;
      kiwisolver = kiwisolver;
      tzdata = tzdata;
      referencing = referencing;
      fasteners = fasteners;
      six = six;
      protobuf = protobuf;
      pyopengl = pyopengl;
      platformdirs = platformdirs;
      charset-normalizer = charset-normalizer;
      click = click;
      fsspec = fsspec;
      attrs = attrs;
      language-model = language-model;
      msgpack = msgpack;
      wandb = wandb;
      urllib3 = urllib3;
      pyobjc-framework-quartz = pyobjc-framework-quartz;
      einops = einops;
      pygments = pygments;
      absl-py = absl-py;
      pluggy = pluggy;
      soupsieve = soupsieve;
      image-classifier = image-classifier;
      flax = flax;
      orbax-checkpoint = orbax-checkpoint.d836b5ef8;
      llvmlite = llvmlite;
      ml-dtypes = ml-dtypes.d3d465209;
      h5py = h5py;
      jaxlib = jaxlib;
      certifi = certifi;
      psutil = psutil;
      numpy = numpy.v1_26_4;
      packaging = packaging;
      stanza-models = stanza-models.d01969111;
      opt-einsum = opt-einsum;
      setproctitle = setproctitle;
      typing-extensions = typing-extensions;
      nbformat = nbformat;
      cycler = cycler;
      pandas = pandas;
      smmap = smmap;
      idna = idna;
      mujoco-mjx = mujoco-mjx;
      jsonschema-specifications = jsonschema-specifications;
      pyobjc-framework-cocoa = pyobjc-framework-cocoa;
      beautifulsoup4 = beautifulsoup4;
      nest-asyncio = nest-asyncio;
      pyparsing = pyparsing;
      scipy = scipy;
      asciitree = asciitree;
      pyobjc-framework-applicationservices = pyobjc-framework-applicationservices;
    };
  };
  env = envs.${
    nixpkgs.system
  };
}