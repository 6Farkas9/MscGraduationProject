import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from Data.MySQLOperator import mysqlop
from Data.BasicRepository import basicrepo

class CDRepository():

    def getLrnQus(self):
        lrn_untqus_uid = mysqlop.get_all_records(
            "learner_question",
            "lrn_uid, qus_uid, correct",
            order_by="lrn_uid, create_time"
        )
        return [(lrn_uid, qus_uid, correct) for (lrn_uid, qus_uid, correct) in lrn_untqus_uid]
    
cdrepo = CDRepository()