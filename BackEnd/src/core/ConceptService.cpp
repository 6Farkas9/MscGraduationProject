#include "ConceptService.h"

ConceptService::ConceptService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {

}

ConceptService::~ConceptService(){

}

std::string ConceptService::addOneConcept(
    std::string &are_uid, 
    std::vector<std::string> &pre_cpt_uids, 
    std::vector<std::string> &aft_cpt_uids,
    std::string &name
) {
    // 向mysql-concepts中添加新的cpt
    // 新建uid
    std::string cpt_uid = UidCreator::generate_uuid_winapi();
    while (mysqlop.judgeConceptsHadUid(cpt_uid)) {
        cpt_uid = UidCreator::generate_uuid_winapi();
    }
    cpt_uid = std::string("cpt_") + cpt_uid;
    // 向concepts中插入数据
    mysqlop.insertNewCpt_one(are_uid, cpt_uid, name);
    // 向graph_belong中插入数据
    mysqlop.insert_are_cpt_one(are_uid, cpt_uid);
    // 向graph_precondition中插入数据
    std::vector<std::pair<std::string, std::string>> pre_con;
    for (auto &pre_cpt : pre_cpt_uids) {
        pre_con.emplace_back(std::make_pair(pre_cpt, cpt_uid));
    }
    for (auto &aft_cpt : aft_cpt_uids) {
        pre_con.emplace_back(std::make_pair(cpt_uid, aft_cpt));
    }
    mysqlop.insert_cpt_cpt_many(pre_con);

    return cpt_uid;
}

bool ConceptService::deleteOneConcept(std::string &cpt_uid) {
    // 从graph_belong中删除记录
    mysqlop.delete_cpt_from_graph_belong_one(cpt_uid);
    // 从graph_precondition中删除记录
    mysqlop.delete_cpt_cpt_by_cpt_uid_one(cpt_uid);
    // 从graph_involve中删除记录
    mysqlop.delete_scn_cpt_by_cpt_uid_one(cpt_uid);
    // 从concepts中删除
    mysqlop.delete_cpt_from_concepts_one(cpt_uid);

    // 从MongoDB中删除数据
    mongodbop.delete_cpt_from_concepts(std::vector<std::string>{cpt_uid});
}

bool ConceptService::recalculate_kcge_cpt_after_add(std::string &cpt_uid) {
    /*
        1. 关系类型0 ： 自连接 权重1
        2. 关系类型1 ： 知识点 - 领域 权重1
        3. 关系类型2 ： 知识点 - 知识点 权重1
        4. 关系类型3 ： 知识点 - 场景 权重 难度
            新添加的知识点没有关联的场景，关系类型3实际上不需要计算
    */

    
}