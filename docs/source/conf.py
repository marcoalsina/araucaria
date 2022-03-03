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
import os
import sys
from datetime import datetime
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath(os.path.join(os.pardir, os.pardir)))

# retrieving the version
f   = open(os.path.join(os.pardir, os.pardir, 'araucaria', 'version'), 'r')
ver = f.readline()
f.close()

# retrieving date
now = datetime.now()

# -- Project information -----------------------------------------------------

project   = 'araucaria'
copyright = '2020-%s, Marco A. Alsina' % now.year
author    = 'Marco A. Alsina'

# The full version, including alpha/beta/rc tags
release   = ver


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['matplotlib.sphinxext.plot_directive',
              'sphinx.ext.napoleon',  # this needs to be loaded first
              'sphinx.ext.autodoc',
              'sphinx_autodoc_typehints',
              'sphinx.ext.intersphinx',
              'sphinx.ext.autosectionlabel',
              'sphinx_rtd_theme',
              'nbsphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.coverage',
              'sphinx_copybutton',
              'sphinx.ext.viewcode',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme      = 'sphinx_rtd_theme'
#add_module_names = False

# style for docstring Example code
pygments_style = 'colorful'

# type hints parameters
set_type_checking_flag   = True
typehints_fully_qualified= False
autodoc_typehints        = 'description'  # show type hints in doc body instead of signature
autoclass_content        = 'both'         # get docstring from class level and init simultaneously

# plot parameters
plot_include_source = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
autodoc_default_options = {
    'member-order': 'bysource',
}

# intersphinx maps
intersphinx_mapping = {'python'    : ('https://docs.python.org/3', None),
                        'numpy'    : ('http://docs.scipy.org/doc/numpy/', None),
                       'scipy'     : ('https://docs.scipy.org/doc/scipy/reference/', None),
                       'matplotlib': ('https://matplotlib.org/', None),
                       'h5py'      : ('https://docs.h5py.org/en/latest/', None),
                       'lmfit'     : ('https://lmfit.github.io/lmfit-py/', None),
                       }

#copybutton options
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True