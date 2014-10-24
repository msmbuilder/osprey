from setuptools import setup, find_packages

import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'osprey/_version.py'
versioneer.versionfile_build = 'osprey/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'osprey-' # dirname like 'myproject-1.2.0'

setup(
    name='osprey',
    author='Robert T. McGibbon',
    author_email='rmcgibbo@gmail.com',
    url='https://github.com/rmcgibbo/osprey',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'osprey = osprey.main:main',
        ],
    }
)
