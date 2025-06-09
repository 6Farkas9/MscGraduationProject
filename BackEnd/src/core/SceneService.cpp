#include "SceneService.h"

SceneService::SceneService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop)
{

}

SceneService::~SceneService(){

}

std::string SceneService::addNewScene(std::unordered_map<std::string, float> &cpt_uid2diff) {
    /*
        1. scene的基本信息
        2. scene的cpt难度信息
    */

    // 向mysql-scenes中添加新的scn

    // 根据传入的数据向graph_involve中添加对应的记录

    // 
}