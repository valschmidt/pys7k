"""Allow ``python -m PYS7K -f FILE ...``."""

import sys

from .s7k import main

sys.exit(main())
