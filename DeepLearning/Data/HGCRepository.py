import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from Data.MySQLOperator import mysqlop
from Data.BasicRepository import basicrepo

class HGCRepository():

    def getLrnUntQusCount(self) -> list:
        lrn_untqus_uids = basicrepo.getUidUid("interaction", "lrn_uid, unt_uid")
        counter = Counter(lrn_untqus_uids)
        return [(lrn_uid, untqus_uid, count) for (lrn_uid, untqus_uid), count in counter.items()]
    
    def getLrnCrsCount(self) -> list:
        lrn_crs_uid = basicrepo.getUidUid("learner_course", "lrn_uid, crs_uid")
        counter = Counter(lrn_crs_uid)
        return [(lrn_uid, crs_uid, count) for (lrn_uid, crs_uid), count in counter.items()]
    
    def getLrnTpcCount(self) -> list:
        lrn_tpc_uid = basicrepo.getUidUid("learner_topic", "lrn_uid, tpc_uid")
        counter = Counter(lrn_tpc_uid)
        return [(lrn_uid, tpc_uid, count) for (lrn_uid, tpc_uid), count in counter.items()]

    def getUntQusCpt(self):
        unt_cpt_uids = basicrepo.getUidUid("unit_concept", "unt_uid, cpt_uid")
        qus_cpt_uids = basicrepo.getUidUid("question_concept", "qus_uid, cpt_uid")
        return [(unt_uid, cpt_uid) for (unt_uid, cpt_uid) in unt_cpt_uids] + [(qus_uid, cpt_uid) for (qus_uid, cpt_uid) in qus_cpt_uids]

    def getUntCrs(self):
        unt_crs_uid = basicrepo.getUidUid("course_unit", "unt_uid, crs_uid")
        return [(unt_uid, crs_uid) for (unt_uid, crs_uid) in unt_crs_uid]
    
    def getUntUnt(self):
        unt_unt_uid = basicrepo.getUidUid("unit_unit", "uid1, uid2")
        return [(uid1, uid2) for (uid1, uid2) in unt_unt_uid]

    def getCptUidName(self):
        cpt_uid_name = mysqlop.get_all_records(
            "concepts",
            "uid, name",
            order_by="id"
        )
        return [(uid, name) for (uid, name) in cpt_uid_name]
    
    def getCptTpc(self):
        cpt_tpc_uid = basicrepo.getUidUid("topic_concept", "cpt_uid, tpc_uid")
        return [(cpt_uid, tpc_uid) for (cpt_uid, tpc_uid) in cpt_tpc_uid]
    
    def getCptCpt(self):
        cpt_cpt_uid = basicrepo.getUidUid("concept_concept", "pre_uid, aft_uid")
        return [(pre_uid, aft_uid) for (pre_uid, aft_uid) in cpt_cpt_uid]

hgcrepo = HGCRepository()

