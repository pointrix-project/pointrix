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
import sys
from pathlib import Path

# find project


# -- Project information -----------------------------------------------------

project = 'Pointrix 中文文档'
copyright = '2024, Pointrix-Group'
author = 'Pointrix-Group'

# The full version, including alpha/beta/rc tags
# release = '0.0.1'

html_logo = "../images/pointrix_landscape_b.png"
html_favicon = "../images/pointrix.ico"
html_title = "Pointrix"

html_context = {
   # ...
   "default_mode": "light"
}
autodoc_mock_imports = ["simple_knn", "dptr", "msplat", "diff_gaussian_rasterization"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'myst_parser',
    'sphinx_design'
    ]

myst_enable_extensions = ["colon_fence"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'
master_doc = 'index_cn'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# # sphinx-gallery configuration
# sphinx_gallery_conf = {
#     # path to your example scripts
#     'examples_dirs': ['../tutorial'],
#     # path to where to save gallery generated output
#     'gallery_dirs': ['auto_gallery-1'],
#     # specify that examples should be ordered according to filename
#     'within_subsection_order': FileNameSortKey,
#     # directory where function granular galleries are stored
#     'backreferences_dir': 'gen_modules/backreferences',
#     # Modules for which function level galleries are created.  In
#     # this case sphinx_gallery and numpy in a tuple of strings.
#     'doc_module': ('SampleModule'),
# }

# configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(sys.version_info), None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/', None),
}
