Data and Datasets
=================

There are two main modules for working with data and datasets respectively:

:py:mod:`stanza.data` contains utilities for working with
:py:class:`Data <stanza.data.Data>` objects
containing structured pytrees which can be loaded
from disk via :py:func:`DataLoader <stanza.data.DataLoader>`
similar to PyTorch.

:py:mod:`stanza.datasets` contains tools for downloading and working 
with common datasets, as well as a :py:class:`DatasetRegistry <stanza.datasets.DatasetRegistry>`
which can be used to register new datasets.