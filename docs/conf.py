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
sys.path.insert(0, os.path.abspath('..'))
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
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx_design',
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

# Use 'obj' as default role so single backticks render as inline code
# rather than trying to resolve as cross-references (which causes
# many warnings for shape parameters like `n_samples`, `n_classes`, etc.)
default_role = 'obj'

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
html_theme = 'pydata_sphinx_theme'

html_title = 'MRCpy'

html_theme_options = {
    "github_url": "https://github.com/MachineLearningBCAM/MRCpy",
    "navigation_with_keys": True,
    "show_prev_next": False,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
    "logo": {
        "text": "MRCpy",
    },
}

html_context = {
    "default_mode": "auto",
}

html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['js/copybutton.js']
htmlhelp_basename = 'MRCpydoc'

