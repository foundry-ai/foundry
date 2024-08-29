Getting Started
===============

Adding foundry to Your Project
-----------------------------

We recommend using `PDM <https://pdm-project.org/>`_ to manage your project's dependencies.

Setting up a project with PDM is easy. Ensure you have ``python>=3.10``. Install PDM:

   curl -sSL https://pdm-project.org/install-pdm.py | python3 -

Then, create a new project in the top-level project directory:

   pdm init

Make sure to choose "yes" when pdm asks if the project should be "built for distribution"

.. warning::
   If you select "no" the ``launch`` script may not be able to find your main module.

1. Adding foundry as a git submodule
"""""""""""""""""""""""""""""""""""

   git submodule add git@github.com:pfrommerd/foundry.git

   pdm add ./foundry

2. Adding foundry as a git dependency
""""""""""""""""""""""""""""""""""""

   pdm add "foundry @ git+https://github.com/pfrommerd/foundry.git"

Development
-----------

We use `PDM <https://pdm-project.org/>`_ to manage foundry.

1. Install Python 3.10 or later
2. Install PDM

      curl -sSL https://pdm-project.org/install-pdm.py | python3 -

3. For CPU-only installation, run

      pdm install -d

   For cuda support, run

      pdm install -d -G cuda12_pip

4. Run an example with

      pdm run python examples/train.py
   
   The recommended way to run a main script is to use

      pdm run launch my_module:my_function

Building documentation
----------------------

1. Install the required packages

      pdm install -d -G docs

2. Build the documentation. In the ``docs`` folder, run:

      make html