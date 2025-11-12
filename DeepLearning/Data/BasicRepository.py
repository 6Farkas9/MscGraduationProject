import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter

from Data.MySQLOperator import mysqlop

class BasicRepository():

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
    
basicrepo = BasicRepository()