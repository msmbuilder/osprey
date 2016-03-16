import os
import shutil
from osprey import version

if osprey.release:
    docversion = version.short_version
else:
    docversion = 'development'

os.mkdir("docs/_deploy")
shutil.copytree("docs/_build/html", "docs/_deploy/{docversion}"
                .format(docversion=docversion))
