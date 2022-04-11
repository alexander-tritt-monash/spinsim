# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


"""
To init rst from comments:
sphinx-apidoc -o source/ ../

To build html from rst:
make html
(must run from cmd, not powershell)
"""

import os
import sys
import sphinx_rtd_theme
# from import exts.numbadoc
sys.path.insert(0, os.path.abspath('../../'))
sys.path.append(os.path.abspath("./_ext"))
sys.path.append(os.path.abspath("../../../"))
# sys.path.insert(1, os.path.abspath('exts\\'))
# from exts import numbadoc
# sys.path.append(os.path.abspath('exts\\'))


# -- Project information -----------------------------------------------------

project = 'spinsim'
copyright = '2022, Monash University'
author = 'Alexander Tritt'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",       # Generate documentation from code
    "sphinx.ext.napoleon",      # Use numpy standard formatting
    "sphinx_rtd_theme",         # Documentation theme
    "numbadoc",                 # Is able to document numba
    "sphinx.ext.intersphinx",   # Link to other documentation
    "sphinxcontrib.bibtex"      # Bibtex support
]

autoclass_content = 'both'
numfig = True

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'haiku'
# html_theme = "default"
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numba': ('http://numba.pydata.org/numba-doc/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('http://matplotlib.sourceforge.net/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'qutip': ('http://qutip.org/docs/latest/', None),
    'neural-sense': ('https://neural-sense.readthedocs.io/en/latest/', None)
}

# add_module_names = False