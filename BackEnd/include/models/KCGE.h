#ifndef KCGE_H
#define KCGE_H

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

class KCGE{

public:
    KCGE(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~KCGE();

    std::unordered_map<std::string, float> forward(const std::string are_uid, const std::string lrn_uid);

    // 新增scn后的重新计算
    // 删除scn后的重新计算
    // 新增cpt后的重新计算
    // 删除cpt后的重新计算


private:

    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop; 
};

#endif //ifndef KCGE_H