import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sentence_transformers import SentenceTransformer

from Data.BasicRepository import basicrepo

class BasicDataReader():

    def __init__(self):
        self.getLrnUid2idx()
        self.getUntQusUid2idx()
        self.getTpcUid2ids()
        self.getCrsUid2idx()
        self.getCptUid2idx()

    def getLrnUid2idx(self):
        uids = basicrepo.getLrnUid()
        self.lrn_uid = {uid : idx for idx, uid in enumerate(uids)}
        self.lrn_num = len(self.lrn_uid)

    def getUntQusUid2idx(self):
        unt_uids = basicrepo.getUntUid()
        qus_uids = basicrepo.getQusUid()

        self.unt_uid = {uid : idx for idx, uid in enumerate(unt_uids)}
        self.unt_num = len(self.unt_uid)
        self.qus_uid = {uid : idx + self.unt_num for idx, uid in enumerate(qus_uids)}
        self.qus_num = len(self.qus_uid)

        uids = unt_uids + qus_uids
        self.untqus_uid = {uid : idx for idx, uid in enumerate(uids)}
        self.untqus_num = len(self.untqus_uid)

    def getTpcUid2ids(self):
        uids = basicrepo.getTpcUid()
        self.tpc_uid = {uid : idx for idx, uid in enumerate(uids)}
        self.tpc_num = len(self.tpc_uid)

    def getCrsUid2idx(self):
        uids = basicrepo.getCrsUid()
        self.crs_uid = {uid : idx for idx, uid in enumerate(uids)}
        self.crs_num = len(self.crs_uid)

    def getCptUid2idx(self):
        uids = basicrepo.getCptUid()
        self.cpt_uid = {uid : idx for idx, uid in enumerate(uids)}
        self.cpt_num = len(self.cpt_uid)

basicdr = BasicDataReader()