# Contributing to FramerAI

Thanks for your interest in improving FramerAI. This guide explains how to set up
the project, propose changes, and get them merged.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Project layout](#project-layout)
- [Development setup](#development-setup)
- [Branching model](#branching-model)
- [Commit messages](#commit-messages)
- [Coding standards](#coding-standards)
- [Running checks locally](#running-checks-locally)
- [Pull requests](#pull-requests)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)

## Ways to contribute

- Fix bugs or improve performance in the model, backend, or website.
- Add or improve documentation.
- Expand test coverage.
- Triage issues and help reproduce reported problems.
- Propose new capabilities through a feature request before implementing large changes.

For anything larger than a small fix, open an issue first so the approach can be
discussed before you invest time.

## Project layout

| Path | Description |
|------|-------------|
| `model/` | Model architecture, tokenizer, configs, data loader, and inference worker. |
| `build.py` | Command line entry point to build, train, and export models. |
| `train.sh` | Convenience wrapper around common training runs. |
| `backend/` | Express API server and WebSocket streaming. |
| `website/` | React frontend built with Vite. |
| `.github/` | CI workflows, issue and pull request templates. |

See [GUIDE.md](GUIDE.md) for a deeper walkthrough of each component.

## Development setup

Python 3.10 or newer, Node 18 or newer for the backend, and Node 20.19 or newer for the
website (its build tooling requires it). CI runs the same versions.

### Model (Python)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Backend (Node)

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

### Website (React)

```bash
cd website
npm install
npm run dev
```

## Branching model

- `stable` is the default branch and holds released, production-ready code.
- `beta` holds pre-release changes staged for the next release.
- `dev` is the active integration branch for day to day work.

Create feature branches from `dev` using a short, descriptive name, for example
`fix/websocket-reconnect` or `feature/rope-scaling-presets`. Open pull requests
against `dev` unless a maintainer directs otherwise.

## Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): short summary

optional body explaining what and why
```

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
Keep the subject line under 72 characters and in the imperative mood.

Do not add `Co-Authored-By:` trailers naming an AI assistant or vendor. A pull
request ref freezes when the pull request merges, so a trailer that lands there
can only be removed by GitHub Support. CI fails the build on one; install the
local hook to catch it before the commit exists:

```bash
git config core.hooksPath scripts/hooks
```

## Coding standards

- Python: format and lint with [ruff](https://docs.astral.sh/ruff/). Target Python 3.10+.
  Prefer type hints and docstrings on public functions and classes.
- JavaScript and React: keep modules small and focused. Match the existing style
  in `backend/` and `website/`.
- Do not commit large binary artifacts, checkpoints, datasets, or secrets. The
  `.gitignore` already excludes common cases.

## Running checks locally

Match what CI runs before opening a pull request.

```bash
# Model (install once: pip install -r requirements.txt -r requirements-dev.txt)
ruff check model build.py scripts tests conftest.py
python -m compileall -q model build.py scripts tests conftest.py
python -m pytest -q

# Backend
cd backend && npm ci && for f in $(find src -name '*.js'); do node --check "$f"; done && npm test

# Website
cd website && npm ci && npm run lint && npm test && npm run build
```

The full `ruff.toml` rule set is blocking in CI, not advisory. `requirements-dev.txt`
pulls in `requirements.txt`, so one install covers both runtime and tooling.

## Pull requests

1. Keep pull requests focused on a single concern.
2. Fill out the pull request template completely.
3. Link the issue the change resolves, for example `Closes #123`.
4. Ensure all CI checks pass.
5. Be responsive to review feedback. Maintainers may request changes before merging.

## Reporting bugs and requesting features

Use the issue templates:

- Bug reports should include reproduction steps, expected behavior, logs, and environment details.
- Feature requests should describe the problem before the proposed solution.

For security issues, do not open a public issue. Follow [SECURITY.md](SECURITY.md).
