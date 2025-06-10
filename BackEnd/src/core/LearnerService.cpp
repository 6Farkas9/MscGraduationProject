#include "LearnerService.h"

LearnerService::LearnerService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {

}

LearnerService::~LearnerService() {

}

std::unordered_map<std::string, float> LearnerService::predict_lrn_kt_in_are(const std::string &lrn_uid, const std::string &are_uid) {
    /*
        1.根据当前时间获取一个月前的时间得到两个时间界限
        2.根据are_uid获取对应的pt文件
        3.加载pt文件
        4.根据are_uid、lrn_uid、时间界限，获取在该领域内的时间界限内的该学生的学习记录
        5.输入到模型中获得结果
        6.获取最后一个的预测结果作为返回
    */
    // 获取时间界限
    auto twotime = MLSTimer::getCurrentand30daysTime();
    auto end_time = twotime[0];
    auto start_time = twotime[1];
    // 获取当前领域的所有知识点
    auto cpt_uids = mysqlop.get_cpt_uid_id_of_area(are_uid);
    int cpt_num = cpt_uids.size();
    // 获取当前领域的知识点的该学生的一个月内的交互数据
    auto interacts = mysqlop.get_Are_lrn_Interacts_Time(
        are_uid,
        lrn_uid,
        start_time,
        end_time
    );
    // 获取交互过的所有场景uid
    std::unordered_set<std::string> scn_uids;
    for(auto & interact : interacts) {
        scn_uids.insert(interact[0]);
    }
    // 获取每个场景所涉及的知识点
    auto scn_cpt = mysqlop.get_Cpt_of_Scn(scn_uids);
    // 构造输入interact
    std::vector<std::vector<int>> interacts_input;
    for (auto & interact : interacts) {
        int skip_num = interact[1] == "1" ? 0 : cpt_num;
        std::vector<int> cpt_idx;
        for (auto & cpt_uid : scn_cpt[interact[0]]) {
            cpt_idx.emplace_back(cpt_uids[cpt_uid] + skip_num);
        }
        interacts_input.emplace_back(std::move(cpt_idx));
    }
    // 获取预测结果
    KT kt(mysqlop, mongodbop);
    auto r_pred = kt.forward(
        are_uid,
        interacts_input,
        cpt_num
    );
    // 构建输出结果
    std::unordered_map<int, std::string> cpt_idx2uid;
    for (auto &uid_idx : cpt_uids) {
        cpt_idx2uid[uid_idx.second] = uid_idx.first;
    }
    std::unordered_map<std::string, float> ans;
    for (int i = 0; i < cpt_num; ++i) {
        ans[cpt_idx2uid[i]] = r_pred[i];
    }
    return ans;
}

std::unordered_map<std::string, float> LearnerService::predicr_lrn_cd_in_are(const std::string &lrn_uid, const std::string &are_uid) {

}

std::unordered_map<std::string, float> LearnerService::predict_lrn_rr(const std::string &lrn_uid) {

}