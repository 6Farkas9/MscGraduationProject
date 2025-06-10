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

    std::vector<float> forward(
        const std::string &are_uid, 
        const std::vector<std::vector<float>> &interact_scn_emb,
        const std::vector<std::vector<float>> &scn_emb,
        const std::vector<std::vector<float>> &cpt_emb
    );

private:
    MySQLOperator &mysqlop;
    MongoDBOperator &mongodbop; 

    torch::jit::Module model_cd;
};

#endif //ifndef CD_H