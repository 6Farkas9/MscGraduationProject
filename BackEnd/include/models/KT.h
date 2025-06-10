#ifndef KT_H
#define KT_H

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

class KT{

public:
    KT(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~KT();

    std::vector<float> forward(
        const std::string &are_uid, 
        const std::vector<std::vector<int>> &interacts, 
        const int &cpt_num  
    );

private:
    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;

    torch::jit::Module IPDKT;
};

#endif //ifndef KT_H