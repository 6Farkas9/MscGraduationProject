#ifndef KT_H
#define KT_H

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

class KT{

public:
    KT(MySQLOperator &db);
    ~KT();

    std::vector<float> forward(const std::string &are_uid, const std::string &lrn_uid);

private:
    std::string now_time;
    std::string thirty_days_ago_time;

    MySQLOperator &mysqlop; 
};

#endif //ifndef KT_H