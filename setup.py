import sys
import versioneer
import subprocess
from distutils.spawn import find_executable
from setuptools import setup, find_packages

versioneer.VCS = 'git'
versioneer.versionfile_source = 'osprey/_version.py'
versioneer.versionfile_build = 'osprey/_version.py'
versioneer.tag_prefix = ''  # tags are like 1.2.0
versioneer.parentdir_prefix = 'osprey-'  # dirname like 'myproject-1.2.0'


def main(**kwargs):
    classifiers = """\
    Development Status :: 3 - Alpha
    Intended Audience :: Science/Research
    License :: OSI Approved :: Apache Software License
    Programming Language :: Python
    Programming Language :: Python :: 3
    Operating System :: Unix
    Operating System :: MacOS
    Operating System :: Microsoft :: Windows
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Information Analysis"""
    setup(
        name='osprey',
        author='Robert T. McGibbon',
        author_email='rmcgibbo@gmail.com',
        url='https://github.com/rmcgibbo/osprey',
        classifiers=[e.strip() for e in classifiers.splitlines()],
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=find_packages(),
        zip_safe=False,
        package_data={'osprey': ['data/*']},
        entry_points={
            'console_scripts': [
                'osprey = osprey.main:main',
            ],
        },
        **kwargs
    )


def readme_to_rst():
    pandoc = find_executable('pandoc')
    if pandoc is None:
        return {}
    long_description = subprocess.check_output(
        [pandoc, 'README.md', '-t', 'rst'])
    short_description = long_description.split('\n\n')[1]
    return {
        'description': short_description,
        'long_description': long_description,
    }


if __name__ == '__main__':
    kwargs = {}
    if any(e in sys.argv for e in ('upload', 'register', 'sdist')):
        kwargs = readme_to_rst()
    main(**kwargs)
