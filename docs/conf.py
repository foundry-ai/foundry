# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
from sphinx.util import logging # type: ignore
logging.getLogger("sphinx.ext.autosummary").setLevel("CRITICAL")

sys.path.insert(0, os.path.abspath("../src"))

project = 'Stanza'
copyright = '2024, Daniel Pfrommer'
author = 'Daniel Pfrommer'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

master_doc = "index"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_design',
]

suppress_warnings = [
    'autosummary'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = None
autosummary_generate = True
napolean_use_rtype = False


typehints_use_signature = True
typehints_use_signature_return = True
typehints_document_rtype = False

# autosummary config

autosummary_generate = True
autosummary_generate_overwrite = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_css_files = [
    "style.css"
]

html_theme_options = {
    'navigation_with_keys': False,
}