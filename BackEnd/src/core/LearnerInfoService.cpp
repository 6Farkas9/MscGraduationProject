#include "LearnerInfoService.h"

LearnerInfoService::LearnerInfoService() :
    mysqlop(MySQLOperator::getInstance()),
    mongodbop(MongoDBOperator::getInstance())
{

}

LearnerInfoService::~LearnerInfoService() {

}

bool LearnerInfoService::get_lrn_info(
    std::string &lrn_uid,
    std::unordered_map<std::string, std::string> &lrn_info,
    std::unordered_map<
        std::string, std::pair<
            std::string, std::unordered_map<std::string, std::pair<
                std::string, float>>>> &are_info
) {
    // 判断lrn是否存在
    if (!mysqlop.judge_lrn_uid_exist(lrn_uid)){
        return false;
    }
    // 根据lrn_uid获取基础信息
    auto lrn_basic_info = mysqlop.get_email_phone_by_lrn_uids(std::unordered_set<std::string>{lrn_uid});
    // 获取lrn交互过得are_uid和are_name
    auto are_uid2name = mysqlop.get_are_uid_name_by_lrn_uid(lrn_uid);
    if (!are_uid2name.size()) {
        return false;
    }
    std::unordered_set<std::string> are_uids;
    for (auto &uid2name : are_uid2name) {
        are_uids.insert(uid2name.first);
    }
    // 获取指定are_uid下属的cpt_uid和name
    auto cpt_uid2name = mysqlop.get_cpt_uid_name_of_multi_area(are_uids);
    // 获取lrn在指定cpt上的成绩
    std::unordered_set<std::string> cpt_uids;
    for (auto &uid2name : cpt_uid2name) {
        cpt_uids.insert(uid2name.first);
    }
    auto cpt_pred = mongodbop.get_lrn_kt_cd_by_cpt_uids(lrn_uid, cpt_uids);
    // 获取cpt对应的are_uid
    auto cpt_uid2are_uid = mysqlop.get_are_uid_by_multi_cpt_uids(cpt_uids);
    // 构建返回结果
    // 构建基础信息
    lrn_info["email"] = std::move(lrn_basic_info[lrn_uid]["email"]);
    lrn_info["phone"] = std::move(lrn_basic_info[lrn_uid]["phone"]);
    // 构建领域信息
    // are_info:
    // are_uid : <
    //     are_name :[
    //         cpt_uid : cpt_name - pred
    //         cpt_uid : cpt_name - pred
    //     ]
    // >
    for (auto & are_uid : are_uids) {
        // are_info[are_uid] = std::pair<std::string, std::unordered_map<std::string, std::pair<std::string, float>>>();
        are_info[are_uid] = std::move(std::make_pair(
            are_uid2name[are_uid], 
            std::unordered_map<std::string, std::pair<std::string, float>>()
        ));
    }
    for (auto &uid2pred : cpt_pred) {
        are_info[cpt_uid2are_uid[uid2pred.first]].second[uid2pred.first] = std::move(std::make_pair(
            cpt_uid2name[uid2pred.first],
            (uid2pred.second["KT"] + uid2pred.second["CD"]) / 2
        ));
    }
    return true;
}

bool LearnerInfoService::get_recommend_info(
    std::string &lrn_uid,
    std::unordered_map<std::string, std::string> &cpt_uid2name,
    std::vector<std::string> &lrn_partners,
    std::vector<std::string> &lrn_models
) {
    if (!mysqlop.judge_lrn_uid_exist(lrn_uid)){
        return false;
    }
    // 获取RR的结果
    LearnerService lrn_ser;
    auto pred_rr_res = lrn_ser.predict_topK_cpt(
        lrn_uid,
        20
    );
    // 构建cpt的set
    std::unordered_set<std::string> cpt_uids_set;
    for (auto & cpt_uid : pred_rr_res) {
        cpt_uids_set.insert(cpt_uid);
    }
    // 获取对应cpt的name
    cpt_uid2name = std::move(mysqlop.get_cpt_name_by_cpt_uid(
        cpt_uids_set
    ));
    // 获取学习伙伴
    auto partners = mongodbop.get_lrn_partners_by_lrn_cpt_uid(
        lrn_uid,
        cpt_uids_set,
        0.1,
        5
    );
    if (partners != std::nullopt) {
        for (auto & lrn_p_uid : *partners) {
            lrn_partners.emplace_back(std::move(lrn_p_uid));
        }
    }
    // 构建学习榜样
    auto models = mongodbop.get_lrn_models_by_lrn_cpt_uid(
        lrn_uid,
        cpt_uids_set,
        5
    );
    if (models != std::nullopt) {
        for (auto & lrn_m_uid : *models) {
            lrn_models.emplace_back(std::move(lrn_m_uid));
        }
    }
    return true;
}