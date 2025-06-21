#ifndef PLATFORM_STATS_SERVICE_H
#define PLATFORM_STATS_SERVICE_H

#include "MongoDBOperator.h"
#include "MySQLOperator.h"
#include "MLS_config.h"

#include <string>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
#include <map>

class PlatformStatsService {

public:
    PlatformStatsService();
    ~PlatformStatsService();

    // 获取平台状态页面的各种数量数据
    std::unordered_map<std::string, int> get_count_data();

    // 获取深度学习模型的元数据
    std::unordered_map<std::string, std::string> get_deeplearning_data();

private:
    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop;
};

#endif