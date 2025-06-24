#include "UnitItemService.h"

UnitItemService::UnitItemService() :
    mysqlop(MySQLOperator::getInstance()),
    mongodbop(MongoDBOperator::getInstance())
{

}

UnitItemService::~UnitItemService(){

}

std::string UnitItemService::addOneUnit(bool has_result, std::unordered_map<std::string, float> &cpt_uid2diff) {
    /*
        1. unit的基本信息
        2. unit的cpt难度信息
    */

    // 向mysql-units中添加新的unt
    // 新建uid
    std::string unt_uid = UidCreator::generate_uuid_winapi();
    while (mysqlop.judge_unt_uid_exist(unt_uid)) {
        unt_uid = UidCreator::generate_uuid_winapi();
    }
    unt_uid = std::string("unt_") + unt_uid;
    std::cout << unt_uid << std::endl;
    // 向units中插入数据
    mysqlop.insert_one_unt_to_units(unt_uid, has_result);
    // 根据传入的数据向graph_involve中添加对应的记录
    mysqlop.insert_one_unt_to_graph_involve(unt_uid, cpt_uid2diff);

    return unt_uid;
}

bool UnitItemService::deleteOneUnit(std::string unt_uid) {
    // 从graph_involve中删除unt_uid
    mysqlop.delete_one_unt_from_graph_involve(unt_uid);
    // 既然上面的都删除了，那么从interacts中删除unt_uid
    mysqlop.delete_one_unt_from_interacts(unt_uid);
    // 从graph_interact中删除unt_uid
    mysqlop.delete_one_unt_from_graph_interact(unt_uid);
    // 从units中删除unt_uid
    mysqlop.delete_one_unt_from_units(unt_uid);

    // 从mongodb中删除对应的记录
    mongodbop.delete_unt_from_units(std::vector<std::string>{unt_uid});
}