#ifndef CD_H
#define CD_H

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

class CD{

public:
    CD(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~CD();

    std::unordered_map<std::string, float> forward(const std::string are_uid, const std::string lrn_uid);

private:
    std::string now_time;
    std::string thirty_days_ago_time;

    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop; 
};

#endif //ifndef CD_H