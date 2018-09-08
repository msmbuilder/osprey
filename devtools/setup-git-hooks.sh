#!/bin/bash
cd "$(git rev-parse --show-toplevel)"
echo "setting git pre-commit hook"
# ln -s -f ../../devtools/pre-commit-hook .git/hooks/pre-commit
# ln -s -f ../../devtools/remove-trailing-whitepace-hook.sh .git/hooks/pre-commit
ln -s -f ../../devtools/pre-commit .git/hooks/pre-commit
