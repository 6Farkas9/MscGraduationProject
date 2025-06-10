#include "SceneService.h"

SceneService::SceneService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop)
{

}

SceneService::~SceneService(){

}

std::string SceneService::addNewScene(bool has_result, std::unordered_map<std::string, float> &cpt_uid2diff) {
    /*
        1. scene的基本信息
        2. scene的cpt难度信息
    */

    // 向mysql-scenes中添加新的scn
    // 新建uid
    std::string scn_uid = UidCreator::generate_uuid_winapi();
    while (mysqlop.judgeScenesHadUid(scn_uid)) {
        scn_uid = UidCreator::generate_uuid_winapi();
    }
    scn_uid = std::string("scn_") + scn_uid;
    std::cout << scn_uid << std::endl;
    // 向scenes中插入数据
    mysqlop.insertNewScn(scn_uid, has_result);
    // 根据传入的数据向graph_involve中添加对应的记录
    mysqlop.insert_scn_cpt_record(scn_uid, cpt_uid2diff);

    return scn_uid;
}

bool SceneService::deleteOneScene(std::string scn_uid) {
    // 从graph_involve中删除scn_uid
    mysqlop.delete_scn_cpt_by_scn_uid(scn_uid);
    // 既然上面的都删除了，那么从interacts中删除scn_uid
    mysqlop.delete_scn_from_interacts(scn_uid);
    // 从graph_interact中删除scn_uid
    mysqlop.delete_scn_from_graph_interact(scn_uid);
    // 从scenes中删除scn_uid
    mysqlop.delete_scn_from_scenes(scn_uid);

    // 从mongodb中删除对应的记录
    mongodbop.delete_scn_from_scenes(std::vector<std::string>{scn_uid});
}