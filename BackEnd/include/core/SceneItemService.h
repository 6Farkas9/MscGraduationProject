#ifndef UNIT_SERVICE_H
#define UNIT_SERVICE_H

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

class UnitItemService{

public:
    UnitItemService();
    ~UnitItemService();

    std::string addOneUnit(bool has_result, std::unordered_map<std::string, float> &cpt_uid2diff);
    bool deleteOneUnit(std::string unt_uid);

private:
    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;

    bool update_after_add_unt(std::string unt_uid);
};

#endif //ifndef UNIT_SERVICE_H