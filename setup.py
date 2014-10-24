from setuptools import setup, find_packages

setup(
    name='adaptive-search',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'mixtape-worker = adaptive_search.main_worker:main'],
    }
)
