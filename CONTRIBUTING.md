# Contributing to NNTile

We do all of our development using git, so basic knowledge is assumed. Then
follow these steps to contribute code.

1. Fork the NNTile repository by clicking the **Fork** button on the
   [repository page](http://github.com/nntile/nntile). This creates a copy of
   the NNTile repository in your own account.

2. Install Python 3.10+ locally in order to run tests.

3. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/nntile
   cd nntile
   pip install -e wrapers/python  # Install NNTile in editable mode.
   ```

4. Add the NNTile repo as an upstream remote, so you can use it to sync with
   main development line.

   ```bash
   git remote add upstream https://github.com/nntile/nntile
   ```

5. Create a branch where you will develop as follows.

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor (we recommend [Visual
   Studio Code](https://code.visualstudio.com/)).

6. Make sure your code passes NNTile's lint and type checks, by running the
   following from the top of the repository.

   ```bash
   pip install pre-commit
   pre-commit run --all
   ```

   See Linting and Type-Checking section for more details.

7. Make sure the tests pass by running the following command from
   `wrappers/python` directory.

   ```bash
   cd wrappers/python
   pytest
   ```

   NNTile's test suite is quite large, so if you know the specific test file
   that covers your changes, you can limit the tests to that. For example, run
   only `optimizer/test_adam.py`

   ```bash
   pytest optimizer/test_adam.py
   ```

   You can narrow the tests further by using the `pytest -k` flag to match
   particular test names.

   ```bash
   pytest optimizer/test_adam.py -k adam
   ```

   NNTile also offers more fine-grained control over which particular tests are
   run; see Running Tests for more information.

8. Once you are satisfied with your change, create a commit as follows ([how
   to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m 'Your commit message'
   ```

   Then sync your code with the main repo.

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request as follows.

   ```bash
   git push --set-upstream origin name-of-change
   ```

   Please ensure your contribution is a single commit (see Single Change
   Commits section).

9. Create a pull request from the NNTile repository and send it for review.
   Check the Pull Request Checklist for considerations when preparing your PR,
   and consult [GitHub Help][gh-help] if you need more information on using
   pull requests.

[gh-help]: https://help.github.com/articles/about-pull-requests/

## NNTile Pull Request Checklist

As you prepare a NNTile pull request, here are a few things to keep in mind.

### Single-Change Commits and Pull Requests

A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests typically comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may need to squash together
multiple commits. We ask that you do this prior to sending the PR for review if
possible. The `git rebase -i` command might be useful to this end.

### Linting and Type-Checking

NNTile uses [mypy][mypy] and [ruff][ruff] to statically test code quality; the
easiest way to run these checks locally is via the [pre-commit][pre-commit]
framework.

```bash
pip install pre-commit
pre-commit run --all
```

If your pull request touches documentation notebooks, this will also run some
checks on those.

[mypy]: https://mypy.readthedocs.io/
[ruff]: https://docs.astral.sh/ruff/
[pre-commit]: https://pre-commit.com/

### GitHub CI

Your PR will automatically be run through a full test suite on GitHub CI, which
covers a range of Python versions, dependency versions, and configuration
options. It's normal for these tests to turn up failures that you didn't catch
locally; to fix the issues you can push new commits to your branch.

# Alternative Way to Build Development Environment

Sometimes development of low-level components requires hermetic builds. In this
case, containerization approach can help. Specifically, the following builds a
docker images which can be used to build and test low-level CPU-only routines.

```shell
docker build -t ghcr.io/nntile/nntile/sandbox:cpu -f ci/Dockerfile . \
    --build-arg=BASE_IMAGE=ubuntu:24.04 \
    --target=sandbox
```

As your image is built, you can run a container from it as follows.

```shell
docker run --rm -ti -v $HOME:$HOME -w $PWD ghcr.io/nntile/nntile/sandbox:cpu
```

Then run tests or build NNTile from scratch.

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithInfo -DUSE_CUDA=OFF
cmake --build build
```
