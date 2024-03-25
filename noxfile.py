import nox


@nox.session(python=["3.11"])
def tests(session):
    """Running tests."""
    args = session.posargs or ["--cov=src", "-m", "not slow"]
    session.run("pip", "install", ".[dev]")
    session.run("pytest", *args)


@nox.session(python=["3.11"])
def lint(session):
    """Linting."""
    args = session.posargs or ["src", "tests", "noxfile.py"]
    session.run("pip", "install", ".[dev]")
    session.run("ruff", "check", ".")
    session.run("mypy", "--install-types", "--non-interactive", *args)
