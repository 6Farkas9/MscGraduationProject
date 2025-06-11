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

    std::unordered_set<std::string> pre_cpt, aft_cpt;

    pre_cpt.insert("cpt_f4e10b32f85746d7900fdbff3b27276e");
    pre_cpt.insert("cpt_5a315add91b0469f8537cb37feb0dc0c");
    pre_cpt.insert("cpt_a86e9f3aff6245979bf1c8a9454b5dde");

    aft_cpt.insert("cpt_f361c531a42048c18d55769b782c3fd5");
    aft_cpt.insert("cpt_c7d6af53b25944e6bb3b990c3076df05");
    aft_cpt.insert("cpt_92b35b614dbc4dcfaa4f3592f8e6d0cd");

    std::string cpt_uid = cpt_ser.addOneConcept(
        are_uid,
        pre_cpt,
        aft_cpt,
        name
    );

    std::cout << cpt_uid << std::endl;

    cpt_ser.deleteOneConcept(cpt_uid);

    // std::string to_del = "cpt_78d344e6001b44f692771130c154d3bc";
    // cpt_ser.deleteOneConcept(to_del);

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

// KCGE_Emb
// Array (32)
// 0
// 0.00025798901333473623
// 1
// 0.00023150006018113345
// 2
// -0.000001519316015219374
// 3
// 0.000139836804009974
// 4
// 0.0002999796997755766
// 5
// 0.000015198971595964395
// 6
// 0.000018709421055973507
// 7
// 0.000009768238669494167
// 8
// -0.000002336791567358887
// 9
// 0.00028062882483936846
// 10
// -0.0000029016505322942976
// 11
// 0.00031215211492963135
// 12
// 0.00032200056011788547
// 13
// 0.000011528577488206793
// 14
// -0.0000015977234397723805
// 15
// 0.0000914266420295462
// 16
// 0.00024686381220817566
// 17
// 0.000015845416783122346
// 18
// 0.0003171757562085986
// 19
// 0.00001947951204783749
// 20
// 0.0002763731754384935
// 21
// -0.0000011185833272975287
// 22
// -0.0000019406998035265133
// 23
// -0.000002422797024337342
// 24
// -0.000001737853608574369
// 25
// -0.0000021338360056688543
// 26
// 0.00023211594088934362
// 27
// 0.00032360493787564337
// 28
// -0.0000012492748737713555
// 29
// 0.0003617081674747169
// 30
// 0.0003610206476878375
// 31
// -0.0000015676852171964129

