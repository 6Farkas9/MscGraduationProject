#include <iostream>
#include "MongoDBOperator.h"
#include "MySQLOperator.h"
// #include "UidCreator.h"
// #include "SceneService.h"
// #include "LearnerService.h"
#include "ConceptService.h"

int main() {
    MySQLOperator& mysqlop = MySQLOperator::getInstance();
    mysqlop.initialize();
    MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
    mongodbop.initialize();

    ConceptService cpt_ser = ConceptService(mysqlop, mongodbop);

    std::string are_uid = "are_3fee9e47d0f3428382f4afbcb1004117";
    std::string name = "test_name";

    std::vector<std::string> pre_cpt, aft_cpt;

    pre_cpt.emplace_back("cpt_f4e10b32f85746d7900fdbff3b27276e");
    pre_cpt.emplace_back("cpt_5a315add91b0469f8537cb37feb0dc0c");
    pre_cpt.emplace_back("cpt_a86e9f3aff6245979bf1c8a9454b5dde");

    aft_cpt.emplace_back("cpt_f361c531a42048c18d55769b782c3fd5");
    aft_cpt.emplace_back("cpt_c7d6af53b25944e6bb3b990c3076df05");
    aft_cpt.emplace_back("cpt_92b35b614dbc4dcfaa4f3592f8e6d0cd");

    std::string cpt_uid = cpt_ser.addOneConcept(
        are_uid,
        pre_cpt,
        aft_cpt,
        name
    );

    std::cout << cpt_uid << std::endl;

    cpt_ser.deleteOneConcept(cpt_uid);
    // cpt_ser.deleteOneConcept("cpt_a0a30edc635b477d88fc531d79f97f4e");

    return 0;

    // LearnerService lrn_ser = LearnerService(mysqlop, mongodbop);

    // auto res = lrn_ser.predict_lrn_rr(
    //     "lrn_aee0624932cf4affa00626e8f038c4e8"
    // );

    // std::cout << res.size() << std::endl;

    // for (auto & kv : res) {
    //     std::cout << kv.first << " , " << kv.second << std::endl;
    // }

}

// "are_3fee9e47d0f3428382f4afbcb1004117"

