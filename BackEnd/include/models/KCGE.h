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

class {

public:
    KCGE(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~KCGE();

    // 本质上的功能就是增删改对应的嵌入
    // 其中不应该存在单纯的修改的情况
    // 先不管删除的情况
    // 就是新增场景和知识点
    // 对于新增场景和知识点 -- 不应该在这里完成所有的新增工作，这里的工作是为新增的场景/知识点计算对应的KCGE嵌入表达
    // 在SceneService中完成场景的初步新增，然后在这里完成KCGE的计算

    std::unordered_map<std::string, float> forward(const std::string are_uid, const std::string lrn_uid);

private:
    std::string now_time;
    std::string thirty_days_ago_time;

    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop; 
};

#endif //ifndef KCGE_H