# This is for compatibility with msmbuilder/mdtraj doc building code
# which requires the following variables in osprey.version
#
# This file adapts the output of versioneer

from ._version import get_versions
versions = get_versions()
short_version = versions['version']
version = short_version

# Try to determine if this is a release
# Make sure the source is not dirty
# Make sure the source is at a tag. The version is something like 1.0+4 for
# things that are 4 commits past the last tag
release = not ('dirty' in versions and versions['dirty']) \
        and '+' not in version \
        and 'dirty' not in version
