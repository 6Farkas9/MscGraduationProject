#include "LearnerItemService.h"

LearnerItemService::LearnerItemService() :
    mysqlop(MySQLOperator::getInstance()),
    mongodbop(MongoDBOperator::getInstance())
{

}

LearnerItemService::~LearnerItemService() {

}

std::unordered_map<std::string, float> LearnerItemService::predict_lrn_kt_in_are(const std::string &lrn_uid, const std::string &are_uid) {
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
    auto interacts = mysqlop.get_interacts_in_area_of_lrn_with_time(
        are_uid,
        lrn_uid,
        start_time,
        end_time
    );
    // 获取交互过的所有场景uid
    std::unordered_set<std::string> unt_uids;
    for(auto & interact : interacts) {
        unt_uids.insert(interact[0]);
    }
    // 获取每个场景所涉及的知识点
    auto unt_cpt = mysqlop.get_cpt_of_unt(unt_uids);
    // 构造输入interact
    std::vector<std::vector<int>> interacts_input;
    for (auto & interact : interacts) {
        int skip_num = interact[1] == "1" ? 0 : cpt_num;
        std::vector<int> cpt_idx;
        for (auto & cpt_uid : unt_cpt[interact[0]]) {
            cpt_idx.emplace_back(cpt_uids[cpt_uid] + skip_num);
        }
        interacts_input.emplace_back(std::move(cpt_idx));
    }
    // 获取预测结果
    KT kt(mysqlop, mongodbop);
    auto r_pred = kt.forward(
        lrn_uid,
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

std::unordered_map<std::string, float> LearnerItemService::predict_lrn_cd_in_are(const std::string &lrn_uid, const std::string &are_uid) {
    // 获取时间界限
    auto twotime = MLSTimer::getCurrentand30daysTime();
    auto end_time = twotime[0];
    auto start_time = twotime[1];
    // 获取近30天内关于are_uid的交互记录
    auto interacts = mysqlop.get_interacts_in_area_of_lrn_with_time(
        are_uid, lrn_uid, 
        start_time, 
        end_time
    );
    // 从交互记录中获取交互的unt_uid
    std::unordered_set<std::string> unt_uids, cpt_uids;
    for(auto & interact : interacts) {
        unt_uids.insert(interact[0]);
    }
    // 获取对应unt_uid的KCGE_Emb
    auto interact_unt_emb_map = mongodbop.get_unt_kcge_by_unt_uid(unt_uids);
    unt_uids.clear();
    std::vector<std::vector<float>> interact_unt_emb;
    for (auto & kv : interact_unt_emb_map) {
        interact_unt_emb.emplace_back(std::move(kv.second));
    }
    // 获取are_uid相关的所有special_unt及其对应的cpt
    std::unordered_map<std::string, std::string> special_unt_cpt = mysqlop.get_special_unt_cpt_uid_of_are(are_uid);
    // 获取special_unt和cpt的KCGE_Emb - h_unt和h_cpt
    for (auto &unt_cpt : special_unt_cpt) {
        unt_uids.insert(std::move(unt_cpt.first));
        cpt_uids.insert(std::move(unt_cpt.second));
    }
    std::unordered_map<std::string, std::vector<float>> unt_emb_map = mongodbop.get_unt_kcge_by_unt_uid(unt_uids);
    std::unordered_map<std::string, std::vector<float>> cpt_emb_map = mongodbop.get_cpt_kcge_by_cpt_uid(cpt_uids);
    std::vector<std::vector<float>> unt_emb, cpt_emb;
    std::vector<std::string> ordered_cpt_uid;
    for (auto &unt_e : unt_emb_map) {
        unt_emb.emplace_back(std::move(unt_e.second));
    }
    for (auto &cpt_e : cpt_emb_map) {
        ordered_cpt_uid.emplace_back(std::move(cpt_e.first));
        cpt_emb.emplace_back(std::move(cpt_e.second));
    }
    // 调用模型
    CD cd = CD(mysqlop, mongodbop);
    auto r_pred = cd.forward(
        lrn_uid,
        are_uid,
        interact_unt_emb,
        unt_emb,
        cpt_emb
    );
    std::unordered_map<std::string, float> ans;
    int cpt_num = cpt_emb.size();
    for (int i = 0; i < cpt_num; ++i) {
        ans[ordered_cpt_uid[i]] = r_pred[i];
    }
    return ans;
}

std::unordered_map<std::string, float> LearnerItemService::predict_lrn_rr(const std::string &lrn_uid) {
    // 获取时间界限
    auto twotime = MLSTimer::getCurrentand30daysTime();
    auto end_time = twotime[0];
    auto start_time = twotime[1];
    // 获取指定lrn的HGC_Emb
    std::unordered_set<std::string> lrn_uids, unt_uids, cpt_uids;
    lrn_uids.insert(lrn_uid);
    std::unordered_map<std::string, std::vector<float>> lrn_emb_map = mongodbop.get_lrn_hgc_by_lrn_uid(lrn_uids);
    std::vector<float> lrn_emb = std::move(lrn_emb_map[lrn_uid]);
    // 获取近30天内lrn_uid的交互记录
    auto interacts = mysqlop.get_lrn_interacts_time(
        lrn_uid, 
        start_time, 
        end_time
    );
    // 获取交互记录中的unt_uids
    for(auto & interact : interacts) {
        unt_uids.insert(interact[0]);
    }
    // 获取unt_uids对应的HGC_Emb
    std::unordered_map<std::string, std::vector<float>> unt_emb_map = mongodbop.get_unt_hgc_by_unt_uid(unt_uids);
    std::vector<std::vector<float>> unt_emb;
    std::unordered_map<std::string, int> unt_uid2idx;
    int idx = -1;
    for (auto &unt_e : unt_emb_map) {
        unt_uid2idx[unt_e.first] = ++idx;
        unt_emb.emplace_back(std::move(unt_e.second));
    }
    // 获取所有知识点（涉及推荐范围）的HGC_Emb
    std::unordered_map<std::string, std::vector<float>> cpt_emb_map = mongodbop.get_all_cpt_hgc();
    std::vector<std::vector<float>> cpt_emb;
    std::vector<std::string> ordered_cpt_uid;
    for (auto &cpt_e : cpt_emb_map) {
        ordered_cpt_uid.emplace_back(std::move(cpt_e.first));
        cpt_emb.emplace_back(std::move(cpt_e.second));
    }
    // 构造unt_index_vec
    std::vector<int> unt_index;
    for (auto & interact : interacts) {
        unt_index.emplace_back(unt_uid2idx[interact[0]]);
    }
    // 调用模型
    RR rr = RR(mysqlop, mongodbop);
    auto r_pred = rr.forward(
        lrn_uid,
        lrn_emb, 
        unt_emb, 
        cpt_emb, 
        unt_index
    );
    // 构建结果
    std::unordered_map<std::string, float> ans;
    int cpt_num = cpt_emb.size();
    for (int i = 0; i < cpt_num; ++i) {
        ans[ordered_cpt_uid[i]] = r_pred[i];
    }
    return ans;
}

std::deque<std::string> LearnerItemService::predict_topK_cpt(
    const std::string &lrn_uid,
    int K
) {
    auto pred_rr_res = predict_lrn_rr(lrn_uid);

    auto cmp = [&](const std::string& a, const std::string& b) {
        return pred_rr_res[a] > pred_rr_res[b];
    };

    std::priority_queue<std::string, std::deque<std::string>, decltype(cmp)> heap(cmp);
    for (auto & cpt_uid2score : pred_rr_res) {
        heap.push(cpt_uid2score.first);
        if (heap.size() > K) {
            heap.pop();
        }
    }
    std::deque<std::string> ans;
    while(!heap.empty()) {
        ans.push_front(heap.top());
        heap.pop();
    }
    return ans;
}