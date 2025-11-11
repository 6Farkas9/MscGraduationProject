import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

from collections import Counter

from Data.MySQLOperator import mysqlop

class CDRepository():

    def __init__(self):
        pass