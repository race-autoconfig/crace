#!/bin/bash
set -e
CURRENT=`pwd`

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

cd "$REPO_ROOT"

# # README → docs/index
# if [ -f README.md ]; then
#     rsync -a README.md docs/index.md 
# fi

# # LICENSE.md → docs/LICENSE.md
# if [ -f LICENSE.md ]; then
#     rsync -a LICENSE.md docs/LICENSE.md 
# fi

# crace/vignettes/user_guide → docs/crace/vignettes/user_guide
if [ -f crace/vignettes/crace-package.pdf ]; then
    rsync -a crace/vignettes/crace-package.pdf docs/references/crace-package.pdf 
fi

cd "$CURRENT"