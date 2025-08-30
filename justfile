test:
    poetry run pytest -vv
    pre-commit run --all-files
