#ifndef LEARNER_INFO_SERVICE_H
#define LEARNER_INFO_SERVICE_H

#include "MongoDBOperator.h"
#include "MySQLOperator.h"
#include "MLS_config.h"
#include "LearnerItemService.h"

#include <string>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
#include <map>

class LearnerInfoService {

public:
    LearnerInfoService();
    ~LearnerInfoService();

    // 根据lrn_uid获取该学习者的基础信息以及交互过的领域
    // are_info:
    // are_uid : <
    //     are_name :[
    //         cpt_uid : cpt_name - pred
    //         cpt_uid : cpt_name - pred
    //     ]
    // >
    bool get_lrn_info(
        std::string &lrn_uid,
        std::unordered_map<std::string, std::string> &lrn_info,
        std::unordered_map<
            std::string, std::pair<
                std::string, std::unordered_map<std::string, std::pair<
                    std::string, float>>>> &are_info
    );

    // 获取推荐的知识点，学习伙伴，学习伴侣的uid
    bool get_recommend_info(
        std::string &lrn_uid,
        std::unordered_map<std::string, std::string> &cpt_uid2name,
        std::vector<std::string> &lrn_partners,
        std::vector<std::string> &lrn_models
    );


private:
    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop;
};

#endif