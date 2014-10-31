import os
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:
    import sphinx_rtd_theme

extensions = [
    'sphinx.ext.mathjax',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'osprey'
copyright = u'2014, Stanford University'

# The short X.Y version.
version = '0.2'
# The full version, including alpha/beta/rc tags.
release = '0.2'

# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

if not on_rtd:
    html_theme = "sphinx_rtd_theme"

if not on_rtd:
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'ospreydoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'osprey.tex', u'osprey Documentation',
     u'Robert T. McGibbon', 'manual'),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'osprey', u'osprey Documentation',
     [u'Robert T. McGibbon'], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'osprey', u'osprey Documentation',
     u'Robert T. McGibbon', 'osprey', 'One line description of project.',
     'Miscellaneous'),
]
