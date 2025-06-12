#include "ConceptService.h"

ConceptService::ConceptService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {

}

ConceptService::~ConceptService(){

}

std::string ConceptService::addOneConcept(
    std::string &are_uid, 
    std::unordered_set<std::string> &pre_cpt_uids, 
    std::unordered_set<std::string> &aft_cpt_uids,
    std::string &name
) {
    // 向mysql-concepts中添加新的cpt
    // 新建uid
    std::string cpt_uid = UidCreator::generate_uuid_winapi();
    while (mysqlop.judge_cpt_uid_exist(cpt_uid)) {
        cpt_uid = UidCreator::generate_uuid_winapi();
    }
    cpt_uid = std::string("cpt_") + cpt_uid;
    // 向concepts中插入数据
    mysqlop.insert_one_cpt_to_concepts(are_uid, cpt_uid, name);
    // 向graph_belong中插入数据
    mysqlop.insert_one_are_cpt_to_graph_belong(are_uid, cpt_uid);
    // 向graph_precondition中插入数据
    std::vector<std::pair<std::string, std::string>> pre_con;
    for (auto &pre_cpt : pre_cpt_uids) {
        pre_con.emplace_back(std::make_pair(pre_cpt, cpt_uid));
    }
    for (auto &aft_cpt : aft_cpt_uids) {
        pre_con.emplace_back(std::make_pair(cpt_uid, aft_cpt));
    }
    mysqlop.insert_multi_cpt_cpt_to_graph_precondition(pre_con);

    // 重新计算相关emb
    recalculate_kcge_cpt_after_add(
        are_uid,
        cpt_uid,
        pre_cpt_uids,
        aft_cpt_uids
    );

    return cpt_uid;
}

bool ConceptService::deleteOneConcept(std::string &cpt_uid) {
    // 从graph_belong中删除记录
    mysqlop.delete_one_cpt_from_graph_belong(cpt_uid);
    // 从graph_precondition中删除记录
    mysqlop.delete_one_cpt_from_graph_precondition(cpt_uid);
    // 从graph_involve中删除记录
    mysqlop.delete_one_cpt_from_graph_involve(cpt_uid);
    // 从concepts中删除
    mysqlop.delete_one_cpt_from_concepts(cpt_uid);
    // 从MongoDB中删除数据
    mongodbop.delete_cpt_from_concepts(std::vector<std::string>{cpt_uid});
    return true;
}

bool ConceptService::recalculate_kcge_cpt_after_add(
    std::string &are_uid,
    std::string &cpt_uid,
    std::unordered_set<std::string> &pre_cpt_uids, 
    std::unordered_set<std::string> &aft_cpt_uids
) {
    /*
        1. 关系类型0 ： 自连接 权重1
        2. 关系类型1 ： 知识点 - 领域 权重1
        3. 关系类型2 ： 知识点 - 知识点 权重1
        4. 关系类型3 ： 知识点 - 场景 权重 难度
            新添加的知识点没有关联的场景，关系类型3实际上不需要计算
    */
    // 添加新知识点的重点在于关系0和关系2（关系1有但是不重要，关系3没有）
    // 构建idx2uid
    std::unordered_map<int, std::string> cpt_idx2uid;
    std::unordered_map<std::string, int> cpt_uid2idx;
    // 构建基础容器
    std::vector<std::vector<float>> x;
    std::vector<std::vector<int64_t>> edge_index(2, std::vector<int64_t>());
    std::vector<int64_t> edge_type;
    std::vector<float> edge_attr;
    // 获取are_uid对应的emb
    std::unordered_set<std::string> are_uid_set{are_uid};
    auto are_emb_dict = mongodbop.get_are_kcge_by_are_uid(are_uid_set);
    int dim = are_emb_dict[are_uid].size();
    // 将新的cpt_emb置为全1 - x中idx为0
    x.emplace_back(std::move(std::vector<float>(dim, 1.0f)));
    cpt_idx2uid[0] = cpt_uid;
    // 添加are_emb
    x.emplace_back(std::move(are_emb_dict[are_uid]));
    edge_index[0].emplace_back(0);
    edge_index[1].emplace_back(1);
    edge_type.emplace_back(1);
    edge_attr.emplace_back(1.0f);
    edge_index[0].emplace_back(1);
    edge_index[1].emplace_back(0);
    edge_type.emplace_back(1);
    edge_attr.emplace_back(1.0f);
    // 根据pre_cpt_uids获取前置条件的emb
    // 根据aft_cpt_uids获取后置条件的emb
    std::unordered_set<std::string> condition_cpt_uids;
    condition_cpt_uids.insert(pre_cpt_uids.begin(), pre_cpt_uids.end());
    condition_cpt_uids.insert(aft_cpt_uids.begin(), aft_cpt_uids.end());
    // condition_cpt之间也可能有precondition关系
    auto conditon_cpt_emb_dict = mongodbop.get_cpt_kcge_by_cpt_uid(condition_cpt_uids);
    int64_t condition_cpt_start_idx = 2;
    for (auto & cpt_uid_pre : pre_cpt_uids) {
        cpt_uid2idx[cpt_uid_pre] = condition_cpt_start_idx;
        cpt_idx2uid[condition_cpt_start_idx] = cpt_uid_pre;
        x.emplace_back(std::move(conditon_cpt_emb_dict[cpt_uid_pre]));
        edge_index[0].emplace_back(condition_cpt_start_idx++);
        edge_index[1].emplace_back(0);
        edge_type.emplace_back(2);
        edge_attr.emplace_back(1.0f);
    }
    for (auto & cpt_uid_aft : aft_cpt_uids) {
        cpt_uid2idx[cpt_uid_aft] = condition_cpt_start_idx;
        cpt_idx2uid[condition_cpt_start_idx] = cpt_uid_aft;
        x.emplace_back(std::move(conditon_cpt_emb_dict[cpt_uid_aft]));
        edge_index[0].emplace_back(0);
        edge_index[1].emplace_back(condition_cpt_start_idx++);
        edge_type.emplace_back(2);
        edge_attr.emplace_back(1.0f);
    }
    // 构造condition_cpt之间的关系
    auto condition_cpt_cpt = mysqlop.get_cpt_cpt_from_graph_precondition_with_both_in(
        condition_cpt_uids
    );
    for (auto &cpt_cpt : condition_cpt_cpt) {
        edge_index[0].emplace_back(cpt_uid2idx[cpt_cpt[0]]);
        edge_index[1].emplace_back(cpt_uid2idx[cpt_cpt[1]]);
        edge_type.emplace_back(2);
        edge_attr.emplace_back(1.0f);
    }
    // 构造自连接变
    int item_num = x.size();
    for (int64_t i = 0; i < item_num; ++i) {
        edge_index[0].emplace_back(i);
        edge_index[1].emplace_back(i);
        edge_type.emplace_back(3);
        edge_attr.emplace_back(1.0f);
    }
    // 输入模型
    KCGE model_kcge = KCGE(mysqlop, mongodbop);
    try{
    auto x_emb = model_kcge.forward(
        x, 
        edge_index, 
        edge_type, 
        edge_attr
    );
    // 获取结果
    std::unordered_map<std::string, std::vector<float>> cpt_emb_write;
    std::unordered_map<std::string, std::vector<float>> are_emb_write;
    cpt_emb_write[cpt_uid] = std::vector<float>(x_emb[0]);
    are_emb_write[are_uid] = std::vector<float>(x_emb[1]);
    int condition_cpt_num = pre_cpt_uids.size() + aft_cpt_uids.size();
    for (int i = 2; i < condition_cpt_num + 2; ++i) {
        cpt_emb_write[cpt_idx2uid[i]] = std::vector<float>(x_emb[i]);
    }
    // 保存新的结果
    mongodbop.update_cpt_kcge_emb(cpt_emb_write);
    mongodbop.update_are_kcge_emb(are_emb_write);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }
    
    return true;
}