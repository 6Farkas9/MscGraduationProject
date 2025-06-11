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

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>> &x_vec,
        const std::vector<std::vector<int64_t>> &edge_index_vec,
        const std::vector<int64_t> &edge_type_vec,
        const std::vector<float> &edge_attr_vec
    );

private:
    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop; 

    torch::jit::Module model_kcge;
};

#endif //ifndef KCGE_H