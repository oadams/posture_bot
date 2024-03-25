posture_bot
==============================

[![Tests](https://github.com/oadams/posture_bot/workflows/Tests/badge.svg)](https://github.com/oadams/posture_bot/actions?workflow=Tests)

Penalize bad posture.

Initial steps for the person who cookiecutter cloned this repo:
- [x] Once: Initialize git repo with `git init`.
- [ ] Optional, once: Initialize DVC with `dvc init`.
- [ ] Optional, once: Set up a [dvc remote](https://dvc.org/doc/command-reference/remote).
- [x] Go through and delete files we don't want.

Everyone who clones this repo:
- Set up a python virtual environment and automate its use when in this directory, using your preferred method. E.g., with pyenv-virtualenv: ``pyenv virtualenv 3.11 posture_bot `` then `echo posture_bot > .python-version`
- `pip install -e '.[dev]'` to install our project in editable mode with dev
  dependencies.
- `pip install pre-commit`.
- Set up pre-commit hooks `pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit`
- If using DVC: Set up [DVC merge driver](https://dvc.org/doc/user-guide/how-to/resolve-merge-conflicts#directories)
- If using DVC, set up DVC pre-commit hooks: `dvc install --use-pre-commit-tool`
- `nox`: Use this to run tests when you're refactoring or whatever. It's
  not linked to a pre-commit hook like the linting/DVC is as it's a bit slower,
  so use your own judgement about when to run it. Github workflows will also
  run it when we push to Github.

Questions to ask the client:
- Create this list.

Project Organization
------------

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── dvc.yaml           <- DVC file that specifies a pipeline's directed acyclic graph (DAG)
│
├── .github/workflows/tests.yaml  <- Github Actions configuration file.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── noxfile.py         <- Test managmenet configuration
│
├── params.yaml        <- Parameters for experiments managed by DVC.
│
├── .pre-commit-config.yaml  <- Configuration for pre-commit hooks.
│
├── pyproject.toml     <- Houses project metadata and dependencies. Used by
setuptools.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
