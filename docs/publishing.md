# Publishing torch-tvae

Target GitHub repository: https://github.com/xuzhikangnba/torch-tvae

The 0.2.0 candidate includes the paper's original multi-cluster models and a
completed audit. Exact paper reproduction remains unresolved; see
reproduction.md. Do not describe installation or unit-test success as full
benchmark reproduction. The current release is staged locally pending the
author's decision on those documented discrepancies.

The arXiv manuscript is handled separately, using the accepted final source
supplied by the authors. Add its identifier to README and CITATION.cff after
the preprint is available. Do not invent an arXiv ID or DOI.

## GitHub

Create an empty public repository named `torch-tvae` under `xuzhikangnba`.
Push the tested source and tag the release after verifying CI. Keep datasets,
checkpoints, local environments, and credentials outside the repository.

```bash
git remote set-url origin https://github.com/xuzhikangnba/torch-tvae.git
git push -u origin main
git tag v0.2.0
git push origin v0.2.0
```

Use the source install command in README until the PyPI release is published.
For reproducibility, users may pin the Git revision or the v0.2.0 tag.

## Build and inspect

```bash
python -m pip install '.[dev]'
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Install the wheel into a fresh environment with the appropriate PyTorch build.
Run both a forward/backward pass and the custom-backbone example outside the
source directory before publication.

## TestPyPI and PyPI

The package name must still be available when publishing. The PyPI and TestPyPI
projects and credentials are separate. First publish to TestPyPI; install
PyTorch separately and install the test package with --no-deps, so the test
index is only used for this distribution.

```bash
python -m twine upload --repository testpypi dist/*
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps torch-tvae==0.2.0
```

After that validation, the author can publish the same artifacts to PyPI:

```bash
python -m twine upload dist/*
```

Do not commit API tokens. For future automation, configure PyPI Trusted
Publishing for the repository rather than keeping long-lived tokens in code.
Once the first public release is verified, update README with:

```bash
python -m pip install torch-tvae
```

Official references:
- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://docs.pypi.org/trusted-publishers/
