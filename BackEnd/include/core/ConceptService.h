#ifndef CONCEPT_SERVICE_H
#define CONCEPT_SERVICE_H

#include "MongoDBOperator.h"

#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <filesystem>
#include <windows.h>
#include <unordered_set>
#include <unordered_map>
#include <sstream>

#include "MySQLOperator.h"
#include "MLS_config.h"
#include "MLSTimer.h"
#include "UidCreator.h"

class ConceptService{

public:
    ConceptService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~ConceptService();

    // 添加一个新的知识点
    std::string addOneConcept(
        std::string &are_uid, 
        std::vector<std::string> &pre_cpt_uids, 
        std::vector<std::string> &aft_cpt_uids,
        std::string &name
    );

    // 根据uid删除该知识点
    bool deleteOneConcept(std::string &cpt_uid);

    // 添加新知识点后的KCGE计算
    bool recalculate_kcge_cpt_after_add(std::string &cpt_uid);

    // 删除知识点前的KCGE计算 ？ - 不确定是不是应该重新计算，虽然被删除了，但是其本身的信息对预测还是有用的 - 写出来，用不用后说
    bool recalculate_kcge_cpt_before_delete(std::string &cpt_uid);

private:
    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;
};

#endif //ifndef CONCEPT_SERVICE_H