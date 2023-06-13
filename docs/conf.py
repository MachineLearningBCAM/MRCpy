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
# sys.path.insert(0, os.path.abspath('../MRCpy/'))
import sphinx_rtd_theme
from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------

project = u'MRCpy'
copyright = (u'2021, Kartheek Bondugula, Claudia Guerrero, Santiago Mazuelas and Aritz Perez')
author = (u'Kartheek Bondugula, Claudia Guerrero, Santiago Mazuelas and Aritz Perez')

# The full version, including alpha/beta/rc tags
release = '0.1.0'
language = 'en'

# -- General configuration ---------------------------------------------------



# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'

exclude_patterns = ['_build']
pygments_style = 'sphinx'
todo_include_todos = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']

# generate autosummary even if no references
autosummary_generate = True

# Option to only need single backticks to refer to symbols
default_role = 'any'

# Option to hide doctests comments in the documentation (like # doctest:
# +NORMALIZE_WHITESPACE for instance)
trim_doctest_flags = True

# intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'scikit-learn': ('https://scikit-learn.org/stable/', None)
}

# sphinx-gallery configuration
sphinx_gallery_conf = {
    # to generate mini-galleries at the end of each docstring in the API
    # section: (see https://sphinx-gallery.github.io/configuration.html
    # #references-to-examples)
    'doc_module': 'MRCpy',
    'examples_dirs': ['../examples'],
    'backreferences_dir': os.path.join('generated'),
    'within_subsection_order': FileNameSortKey, # You can also use ExplicitOrder if needed
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Switch to old behavior with html4, for a good display of references,
# as described in https://github.com/sphinx-doc/sphinx/issues/6705
# html4_writer = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
htmlhelp_basename = 'MRCpydoc'

# Temporary work-around for spacing problem between parameter and parameter
# type in the doc, see https://github.com/numpy/numpydoc/issues/215. The bug
# has been fixed in sphinx (https://github.com/sphinx-doc/sphinx/pull/5976) but
# through a change in sphinx basic.css except rtd_theme does not use basic.css.
# In an ideal world, this would get fixed in this PR:
# https://github.com/readthedocs/sphinx_rtd_theme/pull/747/files
def setup(app):
    app.add_js_file('js/copybutton.js')
    app.add_css_file("basic.css")
    # app.add_css_file("my-styles.css")

