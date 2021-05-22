import inspect
from IPython.display import Markdown
import re

def object_to_markdown(object):

    # Strip global indentation level
    leading_white_spaces = re.compile("$(\s+)")
    lines, _ = inspect.getsourcelines(object)
    n_spaces = min(re.search('[^ ]', line).start() for  line in lines if not line.isspace())
    stripped_lines = []
    for line in lines:
        if not line.isspace():
            stripped_lines.append(line[n_spaces:])
        else:
            stripped_lines.append(line)
    return Markdown(f"```python\n{''.join(stripped_lines)}```" )

