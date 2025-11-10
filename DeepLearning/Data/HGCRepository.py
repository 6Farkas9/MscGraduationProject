import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

from collections import Counter

from Data.MySQLOperator import mysqlop

class HGCRepository():

    def getUid(self, table) -> list:
        uids = mysqlop.get_all_records(
            table, 
            "uid",
            order_by="id"
        )
        return uids
    
    def getUidUid(self, table, uid) -> list:
        uid_uid = mysqlop.get_all_records(
            table,
            uid
        )
        return uid_uid

    def getLrnUid(self) -> list:
        uids = self.getUid("basiclearners")
        return [item[0] for item in uids]
    
    def getUntUid(self) -> list:
        uids = self.getUid("units")
        return [item[0] for item in uids]
    
    def getQusUid(self) -> list:
        uids = self.getUid("questions")
        return [item[0] for item in uids]
    
    def getTpcUid(self) -> list:
        uids = self.getUid("topics")
        return [item[0] for item in uids]
    
    def getCrsUid(self) -> list:
        uids = self.getUid("courses")
        return [item[0] for item in uids]
    
    def getCptUid(self) -> list:
        uids = self.getUid("concepts")
        return [item[0] for item in uids]

    def getLrnUntQusCount(self) -> list:
        lrn_untqus_uids = self.getUidUid("interaction", "lrn_uid, unt_uid")
        counter = Counter(lrn_untqus_uids)
        return [(lrn_uid, untqus_uid, count) for (lrn_uid, untqus_uid), count in counter.items()]
    
    def getLrnCrsCount(self) -> list:
        lrn_crs_uid = self.getUidUid("learner_course", "lrn_uid, crs_uid")
        counter = Counter(lrn_crs_uid)
        return [(lrn_uid, crs_uid, count) for (lrn_uid, crs_uid), count in counter.items()]
    
    def getLrnTpcCount(self) -> list:
        lrn_tpc_uid = self.getUidUid("learner_topic", "lrn_uid, tpc_uid")
        counter = Counter(lrn_tpc_uid)
        return [(lrn_uid, tpc_uid, count) for (lrn_uid, tpc_uid), count in counter.items()]
    
    # def getCptTpc(self) -> list:

    def getUntQusCpt(self):
        unt_cpt_uids = self.getUidUid("unit_concept", "unt_uid, cpt_uid")
        qus_cpt_uids = self.getUidUid("question_concept", "qus_uid, cpt_uid")
        return [(unt_uid, cpt_uid) for (unt_uid, cpt_uid) in unt_cpt_uids] + [(qus_uid, cpt_uid) for (qus_uid, cpt_uid) in qus_cpt_uids]

    def getUntCrs(self):
        unt_crs_uid = self.getUidUid("course_unit", "unt_uid, crs_uid")
        return [(unt_uid, crs_uid) for (unt_uid, crs_uid) in unt_crs_uid]
    
    def getUntUnt(self):
        unt_unt_uid = self.getUidUid("unit_unit", "uid1, uid2")
        return [(uid1, uid2) for (uid1, uid2) in unt_unt_uid]

    def getCptUidName(self):
        cpt_uid_name = mysqlop.get_all_records(
            "concepts",
            "uid, name",
            order_by="id"
        )
        return [(uid, name) for (uid, name) in cpt_uid_name]
    
    def getCptTpc(self):
        cpt_tpc_uid = self.getUidUid("topic_concept", "cpt_uid, tpc_uid")
        return [(cpt_uid, tpc_uid) for (cpt_uid, tpc_uid) in cpt_tpc_uid]
    
    def getCptCpt(self):
        cpt_cpt_uid = self.getUidUid("concept_concept", "pre_uid, aft_uid")
        return [(pre_uid, aft_uid) for (pre_uid, aft_uid) in cpt_cpt_uid]

hgcrepo = HGCRepository()

