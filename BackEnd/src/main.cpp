#include <iostream>
#include "MongoDBOperator.h"
#include "MySQLOperator.h"
// #include "UidCreator.h"
// #include "SceneService.h"
#include "LearnerService.h"

int main() {
    MySQLOperator& mysqlop = MySQLOperator::getInstance();
    mysqlop.initialize();
    MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
    mongodbop.initialize();

    LearnerService lrn_ser = LearnerService(mysqlop, mongodbop);

    auto res = lrn_ser.predict_lrn_kt_in_are(
        "lrn_aee0624932cf4affa00626e8f038c4e8",
        "are_3fee9e47d0f3428382f4afbcb1004117"
    );

    std::cout << res.size() << std::endl;

    for (auto & kv : res) {
        std::cout << kv.first << " , " << kv.second << std::endl;
    }

    // SceneService scn_ser = SceneService(mysqldb, mongodbop);

    // for (auto &kv : cpt_uid2diff) {
    //     std::cout << kv.first << " , " << kv.second << std::endl;
    // }

    // std::string scn_uid = scn_ser.addOneScene(false, cpt_uid2diff);

    // scn_ser.deleteOneScene(scn_uid);

    // return 0;
}

    // std::unordered_map<std::string, float> cpt_uid2diff;
    // // cpt_f4e10b32f85746d7900fdbff3b27276e
    // // cpt_5a315add91b0469f8537cb37feb0dc0c
    // // cpt_a86e9f3aff6245979bf1c8a9454b5dde
    // cpt_uid2diff["cpt_f4e10b32f85746d7900fdbff3b27276e"] = 0.1;
    // cpt_uid2diff["cpt_5a315add91b0469f8537cb37feb0dc0c"] = 0.2;
    // cpt_uid2diff["cpt_a86e9f3aff6245979bf1c8a9454b5dde"] = 0.3;

// int main() {
//     MySQLOperator& mysqldb = MySQLOperator::getInstance();
//     mysqldb.initialize();
//     MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
//     mongodbop.initialize();

//     RR rr(mysqldb, mongodbop);

//     std::unordered_map<std::string, float> rr_pred =  rr.forward("lrn_aee0624932cf4affa00626e8f038c4e8");

//     for (auto & kv : rr_pred){
//         std::cout << kv.first << " - " << kv.second << std::endl;
//     }

//     return 0;
// }

// int main(){
//     MySQLOperator& mysqldb = MySQLOperator::getInstance();
//     mysqldb.initialize();
//     MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
//     mongodbop.initialize();

//     CD cd(mysqldb, mongodbop);

//     std::unordered_map<std::string, float> cd_pred =  cd.forward(
//         "are_3fee9e47d0f3428382f4afbcb1004117",
//         "lrn_aee0624932cf4affa00626e8f038c4e8"
//     );

//     for (auto & kv : cd_pred){
//         std::cout << kv.first << " - " << kv.second << std::endl;
//     }

//     return 0;
// }

// int main() {
//     try{
//         MySQLOperator& db = MySQLOperator::getInstance();

//         if(!db.initialize()){
//             std::cerr << "Failed to initialize database!" << std::endl;
//             return 1;
//         }

//         // db.get_cpt_uid_id_of_area("are_3fee9e47d0f3428382f4afbcb1004117");

//         KT kt(db);
//         // lrn_aee0624932cf4affa00626e8f038c4e8
//         // are_3fee9e47d0f3428382f4afbcb1004117
//         auto ans = kt.forward(
//             "are_3fee9e47d0f3428382f4afbcb1004117",
//             "lrn_aee0624932cf4affa00626e8f038c4e8"
//         );

//         for(auto & item : ans){
//             std::cout << item << " ";
//         }
//         std::cout << std::endl;
//     }
//     catch(const std::exception& e){
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }

//     return 0;
// }

