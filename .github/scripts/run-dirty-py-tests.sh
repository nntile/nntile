#!/usr/bin/env bash
# Run only those Python (pytest) tests whose corresponding sources were
# updated in the current PR.

set -e

branch=$1
if [ -z "$branch" ]; then
    branch=$(git branch --show-current)
    echo "no branch specified: assume current branch is $branch"
fi

function changed-files() {
    git diff --name-only main..$branch -- "$@"
}

# Start with an empty manifest so later appends always have a target file.
: > changed.txt

# Is native extension (aka nntile_core*.so) affected?
changed-files '**/CMakeLists.txt' > cmake-lists.txt
changed-files 'include/**/*.cuh' \
    'include/**/*.h' 'include/**/*.hh' 'include/**/*.hpp' > headers.txt
changed-files \
    'src/**/*.cc' 'src/**/*.cpp' 'src/**/*.cu' \
    'wrappers/python/*.cc' 'wrappers/python/**/*.cpp' > sources.txt
echo "$(wc -l cmake-lists.txt) cmake-lists changed"
echo "$(wc -l headers.txt) headers changed"
echo "$(wc -l sources.txt) sources changed"
cat cmake-lists.txt headers.txt sources.txt > total.txt
if [ -s total.txt ]; then
    echo wrappers/python/nntile/nntile_core*.so >> changed.txt
fi

# What pure python modules are changed in this PR?
changed-files 'wrappers/python/**/*.py' >> changed.txt

echo ':: Changed python modules'
cat changed.txt

# Unknown changes (e.g. only workflow files) — run full suite, like run-dirty-cpp-tests.sh.
if [ ! -s changed.txt ]; then
    all_changed=$(git diff --name-only main..$branch)
    if [ -n "$all_changed" ]; then
        echo ':: Unknown changes (no pattern matched), running all Python tests'
        pytest -vv \
            --cov=wrappers/python/nntile \
            --cov-report=html:coverage/html/${PYTHON_TAG} \
            --cov-report=xml:coverage/xml/report.${PYTHON_TAG}.xml \
            --junitxml=pytest/report.${PYTHON_TAG}.xml
        exit
    fi
    echo ':: No tests affected'
    exit 0
fi

# What python tests are affected in this PR?
: > affected.txt
pytest-dirty changed.txt affected.txt
echo ':: Affected python tests'
cat affected.txt

# Run tests affected by this PR.
if [ -s affected.txt ]; then
    pytest -vv \
        --cov=wrappers/python/nntile \
        --cov-report=html:coverage/html/${PYTHON_TAG} \
        --cov-report=xml:coverage/xml/report.${PYTHON_TAG}.xml \
        --junitxml=pytest/report.${PYTHON_TAG}.xml \
        @affected.txt
else
    echo ':: No tests affected'
fi
