#ifndef SCENE_SERVICE_H
#define SCENE_SERVICE_H

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

class SceneService{

public:
    SceneService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~SceneService();

    std::string addNewScene(bool has_result, std::unordered_map<std::string, float> &cpt_uid2diff);
    bool deleteOneScene(std::string scn_uid);

private:
    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;
};

#endif //ifndef SCENE_SERVICE_H