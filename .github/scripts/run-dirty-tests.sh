#!/usr/bin/env bash
# TODO(@dakol): We should wrap this script into action for convinient use with
# GitHub Actions.

function name-status() {
    git diff --name-only main..$branch -- "$@"
}

branch=$1
if [ -z "$branch" ]; then
    branch=$(git branch --show-current)
    echo "no branch specified: assume current branch is $branch"
fi

# Is native extension (aka nntile_core*.so) affected?
name-status '**/CMakeLists.txt' > cmake-lists.txt
name-status 'include/**/*.cuh' \
    'include/**/*.h' 'include/**/*.hh' 'include/**/*.hpp' > headers.txt
name-status \
    'src/**/*.cc' 'src/**/*.cpp' 'src/**/*.cu' \
    'wrappers/python/*.cc' 'wrappers/python/**/*.cpp' > sources.txt
echo "$(wc -l cmake-lists.txt) cmake-lists changed"
echo "$(wc -l headers.txt) headers changed"
echo "$(wc -l sources.txt) sources changed"
cat cmake-lists.txt headers.txt sources.txt > total.txt
if [ -s total.txt ]; then
    echo wrappers/python/nntile/nntile_code*.so > changed.txt
fi

# What pure python modules are changed in this PR?
name-status 'wrappers/python/**/*.py' >> changed.txt

# What python modules are changed in this PR?
echo ':: Changed python modules'
cat changed.txt

# What python tests are affected in this PR?
: > affected.txt
pytest-dirty changed.txt affected.txt
echo ':: Affected python tests'
cat affected.txt

# Run tests affected by this PR.
if [ -s affected.txt ]; then
    pytest -vv \
        --junitxml=junit/test-results.xml \
        --cov=com --cov-report=xml --cov-report=html \
        @affected.txt
else
    echo ':: No tests affected'
fi
