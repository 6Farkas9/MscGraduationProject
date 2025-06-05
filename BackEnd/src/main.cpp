#include <iostream>
#include "MongoDBOperator.h"
#include "MySQLOperator.h"
#include "KT.h"
#include "CD.h"

int main(){
    MySQLOperator& mysqldb = MySQLOperator::getInstance();
    mysqldb.initialize();
    MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
    mongodbop.initialize();

    CD cd(mysqldb, mongodbop);

    std::unordered_map<std::string, float> cd_pred =  cd.forward(
        "are_3fee9e47d0f3428382f4afbcb1004117",
        "lrn_aee0624932cf4affa00626e8f038c4e8"
    );

    for (auto & kv : cd_pred){
        std::cout << kv.first << " - " << kv.second << std::endl;
    }

    return 0;
}

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

