"""Package marker for project tools.

Allows tools modules to be imported as `tools.<module>` and avoids mypy
module name conflicts when running static checks.
"""

from typing import List

__all__: List[str] = []
