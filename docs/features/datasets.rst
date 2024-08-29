Data and Datasets
=================

There are two main modules for working with data and datasets respectively:

:py:mod:`foundry.data` contains utilities for working with
:py:class:`Data <foundry.data.Data>` objects
containing structured pytrees which can be loaded
from disk via :py:func:`DataLoader <foundry.data.DataLoader>`
similar to PyTorch.

:py:mod:`foundry.datasets` contains tools for downloading and working 
with common datasets, as well as a :py:class:`DatasetRegistry <foundry.datasets.DatasetRegistry>`
which can be used to register new datasets.