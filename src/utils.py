import inspect
import os
from pathlib import Path
from IPython.display import Markdown
import re

import optuna

import contextlib
import os, sys


root_dir = (Path(__file__).parents[1]).resolve()


def object_to_markdown(object):

    # Strip global indentation level
    leading_white_spaces = re.compile("$(\s+)")
    lines, _ = inspect.getsourcelines(object)
    n_spaces = min(
        re.search("[^ ]", line).start() for line in lines if not line.isspace()
    )
    stripped_lines = []
    for line in lines:
        if not line.isspace():
            stripped_lines.append(line[n_spaces:])
        else:
            stripped_lines.append(line)
    return Markdown(f"```python\n{''.join(stripped_lines)}```")


@contextlib.contextmanager
def cd(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def get_study(study_name, storage_file="optuna-storage.db"):



    with cd(root_dir):
        storage_name = f"sqlite:///{storage_file}"
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_name,
        )
    return study
