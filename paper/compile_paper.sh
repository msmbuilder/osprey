#!/usr/bin/env bash

pandoc -S -o paper.pdf -V geometry:margin=1in --filter pandoc-citeproc paper.md --template paper.template
echo "Done!"
