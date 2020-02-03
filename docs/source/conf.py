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
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'HAWKS'
copyright = '2020, Cameron Shand'
author = 'Cameron Shand'

# The full version, including alpha/beta/rc tags
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Remove the "View page source" link
html_show_sourcelink = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' # may need to install, see https://sphinx-rtd-theme.readthedocs.io/en/stable/

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
        ],
    }

# -- Extension options -------------------------------------------------

# sphinx_gallery_conf = {
#     "examples_dirs": "examples/",   # path to your example scripts
#     "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
#     "filename_pattern": ""
# }

autodoc_member_order = 'alphabetical' # switch to 'bysource'?

autoclass_content = "class"
apidoc_module_dir = "../hawks"
# apidoc_output_dir = "hawks_api"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True

# autodoc_default_options = {
#     'undoc-members': False
# }

napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "deap": ("https://deap.readthedocs.io/en/master/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "seaborn": ("https://seaborn.pydata.org/", None)
}

todo_include_todos = True
