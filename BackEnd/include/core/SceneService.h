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

class SceneService{

public:
    SceneService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~SceneService();

    std::string addNewScene();

private:
    std::string now_time;
    std::string thirty_days_ago_time;

    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;
};

#endif //ifndef SCENE_SERVICE_H