#ifndef RR_H
#define RR_H

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

class RR{

public:
    RR(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~RR();

    std::vector<float> forward(
        const std::vector<float> &lrn_emb_in,
        const std::vector<std::vector<float>> &scn_emb_in,
        const std::vector<std::vector<float>> &cpt_emb_in,
        const std::vector<int> &scn_index_vec
    );

private:
    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop; 

    torch::jit::Module model_rr;
};

#endif //ifndef RR_H