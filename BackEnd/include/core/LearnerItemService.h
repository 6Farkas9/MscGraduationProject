#ifndef LEARNER_SERVICE_H
#define LEARNER_SERVICE_H

#include "MongoDBOperator.h"
#include "MySQLOperator.h"
#include "MLS_config.h"
#include "MLSTimer.h"
#include "UidCreator.h"

#include "KT.h"
#include "CD.h"
#include "RR.h"

#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <filesystem>
#include <windows.h>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <queue>


class LearnerItemService{

public:
    LearnerItemService();
    ~LearnerItemService();

    std::string addOneLearner(bool has_result, std::unordered_map<std::string, float> &cpt_uid2diff);
    bool deleteOneLearner(std::string scn_uid);

    std::unordered_map<std::string, float> predict_lrn_kt_in_are(const std::string &lrn_uid, const std::string &are_uid);

    std::unordered_map<std::string, float> predict_lrn_cd_in_are(const std::string &lrn_uid, const std::string &are_uid);

    std::unordered_map<std::string, float> predict_lrn_rr(const std::string &lrn_uid);

    std::deque<std::string> predict_topK_cpt(const std::string &lrn_uid, int K);

private:
    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;
};

#endif //ifndef LEARNER_SERVICE_H