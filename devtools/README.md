How to release
===================

Pre-release + Github
--------------------
- Update the `docs/whatsnew.rst` document. Use the github view that shows all the
  commits to master since the last release to write it
   * You can also try using [this tool](https://github.com/rmcgibbo/gh-util), which should list all
     of the PRs that have been merged since the laster release.
- Update the version number in `devtools/conda-recipe/meta.yaml`
- Commit to master, and tag the
  release on github.
- Run git pull to pull the newly created tag locally. Versioneer depends on
  this to get the version string right.

PyPI
----
The next step is to add the release to the python package index.

- Git pull, and make sure it pulls the recent tag.
- Run `git clean -fdx` to clean the source directory.
- Create the cannoncal "sdist" (source distribution) using `python setup.py sdist --formats=gztar,zip`.
  You ran git clean, right?
- Inspect the sdist files (they're placed in `dist/`), and make sure they look right.
  You can try installing them into your environment with pip, unzipping or untaring them, etc.
- Once you're satisfied that the sdist is correct, push the source to PyPI using
  `twine upload [path to sdist files]`. This requires being registered on PyPI as a owner or maintainer
  of the project.

Immediately after creating the sdist
------------------------------------
- Update the version number in `devtools/conda-recipy/meta.yaml`
  to `1.(x+1).0.dev0` per PEP440.
- Add a new section in `docs/whatsnew.rst` and mark it "(Development)".
- Commit to master.

Conda
-----
- File a PR against [omnia-md/conda-recipes](https://github.com/omnia-md/conda-recipes) that
  updates the recipe's version string and source URL to pull the new sdist from PyPI. Travis
  and Appveyor will then build binary conda packages.

Wheels
------

PyPI hosts *wheels*, pre-compiled binary packages, like conda packages, for OS X and
Windows. (At the time of this writing, they are still ironing out issues w.r.t.
linux.) To create and upload wheels, download the sdist and unpack the (or check out
the exact tag from git), and run `python setup.py bdist_wheel`.

For example, to build wheels for Python 2.7, 3.4 and 3.5 on OS X, I ran
```
conda env remove -y -n _build
versions=("2.7" "3.4" "3.5")
for v in "${versions[@]}"; do
    conda create -y -n _build python=$v numpy cython
    source activate _build
    python setup.py bdist_wheel
    source deactivate
    conda env remove -y -n _build
done
```
Then, if these all look good, you can upload them to PyPI with twine, as was done with the
sdist.


Docs Building & Hosting
=======================

After a travis build succeeds, the docs are built with sphinx and pushed to
the msmbuilder.org amazon s3 account in the osprey/ subdirectory.
The credentials for that account are stored,
encrypted, in the .travis.yml file.

Multiple versions of the docs are hosted
online. When a build happens on a version with ISRELEASED==False, it's put into
the "development" folder on the S3 bucket. If ISRELEASED==True, it's put into a
subfolder with the name of the short release. The relevant logic is in
`devtools/travis-ci/set_doc_version.py`.


Tools License
=============
Copyright (c) 2012-2016 Stanford University and the Authors
All rights reserved.

Redistribution and use of all files in this folder (devtools) and (../.travis.yml,
../basesetup.py, ../setup.py) files in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
