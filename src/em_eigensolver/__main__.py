"""em_eigensolver 包入口。

支持 `python -m em_eigensolver` 直接运行 CLI。
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
