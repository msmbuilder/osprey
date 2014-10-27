#!/bin/bash
ref_header="from __future__ import print_function, absolute_import, division"
files=$(find . -name '*.py'  ! -name build ! -path './build/*' ! \
            -name versioneer.py ! -name _version.py ! -name __init__.py | xargs)

for file in $files; do
    header=$(head -n 1 $file)
    [ "$header" != "$ref_header" ] && echo "missing __future__ on $file"
done;

echo
echo $ref_header
