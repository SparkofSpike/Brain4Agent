# Releasing

Publishing happens from CI, not from a laptop. `.github/workflows/release.yml`
builds the sdist and wheel, checks the version against the tag, installs the
wheel into a clean environment and runs the CLI, then publishes.

No token is stored anywhere. PyPI issues the publishing job a short-lived OIDC
credential that expires with the job, through a mechanism called
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). A stored
`PYPI_API_TOKEN` is one forgotten rotation away from being a leaked credential;
this has nothing to rotate.

## One-time setup

Both steps are done once, in the PyPI web UI. They cannot be automated from here.

1. **Claim the project.** Sign in at <https://pypi.org> and go to
   *Publishing* → *Add a new pending publisher* with these values:

   | field            | value                            |
   | ---------------- | -------------------------------- |
   | PyPI project name | `brain4agent`                    |
   | Owner            | `SparkofSpike`                    |
   | Repository name  | `Brain4Agent`                     |
   | Workflow name    | `release.yml`                     |
   | Environment name | `pypi`                            |

   PyPI reserves nothing until a publisher exists, so do this before the first
   tag, not after a failed release.

2. **Create the `pypi` environment** in the GitHub repository settings
   (*Settings → Environments → New environment*). Optionally add required
   reviewers, which turns a release into a two-person decision.

## Releasing

```bash
# 1. Bump the version in two places. They are checked against each other by
#    tests/test_version.py, so forgetting one fails locally instead of on PyPI.
#      pyproject.toml            [project].version
#      src/brain4agent/__init__.py   __version__

# 2. Make sure the tree is clean and the gates pass.
python -m pytest tests/ -q
python -m ruff check src bench tests

# 3. Tag and push. The tag must match the version, or the workflow stops
#    before publishing anything.
git tag v0.1.0
git push origin v0.1.0
```

Watch the run with `gh run watch`. The publish step needs the `pypi`
environment, so if you added reviewers it will wait for approval there.

## If a release is wrong

**You cannot delete a version from PyPI and reuse its number.** PyPI refuses
re-uploads of a filename it has already seen, even after deletion, because
mirrors and caches may have the old copy. The only remedy is to publish a new
version. This is why the workflow refuses to publish when the tag and the
version disagree — a typo becomes a stuck version number otherwise.

Yanking, not deleting, is the right response to a bad release: `Yank` in the
project's *Releases* page hides it from resolvers while leaving it installable
for anyone who pinned it.

## Manual upload, for when CI is unavailable

Possible but discouraged, because it is the path that needs a token:

```bash
python -m build --sdist --wheel --outdir dist/
python -m twine upload dist/*
```

`twine` prompts for a username (`__token__`) and the token as a password. Do not
put the token in a file, a shell history, or an environment variable that
outlives the command.
