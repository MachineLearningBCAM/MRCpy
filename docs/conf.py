"""Sphinx configuration file."""

import minimax_risk_classifiers


project = 'MRCpy'
copyright = '2020-2021, MRCpy, Inc.'
author = 'MRCpy Team'
release = version = minimax_risk_classifiers.__version__

extensions = [
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

# Configure nbsphinx for notebooks execution
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

templates_path = ['_templates']

source_suffix = ['.md', '.ipynb']

master_doc = 'index'

language = None

exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

pygments_style = None

html_theme = 'sphinx_rtd_theme'
html_baseurl = 'https://machinelearningbcam.github.io/Minimax-Risk-Classifiers/'
htmlhelp_basename = 'MRCpydoc'
html_last_updated_fmt = '%c'

latex_elements = {
}


latex_documents = [
    (master_doc, 'MRCpy.tex', 'MRCpy Documentation',
     'MRCpy Team', 'manual'),
]

man_pages = [
    (master_doc, 'MRCpy', 'MRCpy Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'MRCpy', 'MRCpy Documentation',
     author, 'MRCpy', 'One line description of project.',
     'Miscellaneous'),
]

epub_title = project
epub_exclude_files = ['search.html']
