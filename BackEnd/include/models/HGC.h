#ifndef HGC_H
#define HGC_H

#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <filesystem>
#include <windows.h>
#include <unordered_set>
#include <unordered_map>
#include <sstream>

#include "DBOperator.h"
#include "MLS_config.h"
#include "MLSTimer.h"

class HGC{

public:
    HGC(DBOperator &db);
    ~HGC();

    std::vector<float> forward();

private:
    std::string now_time;
    std::string thirty_days_ago_time;

    DBOperator &db; 
};

#endif //ifndef HGC_H