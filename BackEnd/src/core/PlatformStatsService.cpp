#include "PlatformStatsService.h"

PlatformStatsService::PlatformStatsService() :
    mysqlop(MySQLOperator::getInstance()),
    mongodbop(MongoDBOperator::getInstance())
{

}

PlatformStatsService::~PlatformStatsService() {

}

std::unordered_map<std::string, int> PlatformStatsService::get_count_data() {
    std::unordered_map<std::string, int> ans;
    ans["are_num"] = mysqlop.get_are_num();
    ans["lrn_num"] = mysqlop.get_lrn_num();
    ans["scn_num"] = mysqlop.get_scn_num();
    ans["cpt_num"] = mysqlop.get_cpt_num();
    ans["ict_num"] = mysqlop.get_ict_num();
    return ans;
}

std::unordered_map<std::string, std::string> PlatformStatsService::get_deeplearning_data() {
    std::string meta_data_path = R"(\training_metadata.txt)";
    meta_data_path = DEEPLEARNING_ROOT + meta_data_path;

    std::unordered_map<std::string, std::string> ans;

    std::ifstream file(meta_data_path);
    if (!file.is_open()) {
        std::cerr << "无法打开 training_metadata.txt" << std::endl;
        return ans;
    }

    std::unordered_map<std::string, std::string> config;
    std::string line;

    while (std::getline(file, line)) {
        size_t eqPos = line.find('=');
        if (eqPos != std::string::npos) {
            std::string key = line.substr(0, eqPos);
            std::string value = line.substr(eqPos + 1);
            config[key] = value;
        }
    }

    if (config.count("lastTrainingTime") && config.count("modelVersion")) {
        ans["lastTrainingTime"] = config["lastTrainingTime"];
        ans["modelVersion"] = config["modelVersion"];
        return ans;
    }

    std::cerr << "文件格式错误，缺少必要的键值" << std::endl;
    return ans;
}

