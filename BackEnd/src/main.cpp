#include <iostream>
#include "MySQLOperator.h"
#include "KT.h"

// #include "bsoncxx/builder/stream/document.hpp"
// #include "mongocxx/instance.hpp"
// #include "mongocxx/uri.hpp"
// #include "mongocxx/client.hpp"

// int main(){
//     try {
//         // 初始化驱动（完全限定类型）
//         mongocxx::instance inst{};

//         // 连接MongoDB（默认本地27017端口）
//         mongocxx::client client{ mongocxx::uri{} };

//         // 获取集合引用（替换your_db和your_collection）
//         mongocxx::collection coll = client["MLS_db"]["learners"];

//         // 1. 构建查询条件（完全限定builder命名空间）
//         auto query = bsoncxx::builder::stream::document{}
//             << "_id" << "lrn_002c6050d7f24038a1b7d63b4e8fc116"
//             << bsoncxx::builder::stream::finalize;

//         // 2. 执行查询
//         mongocxx::cursor cursor = coll.find(query.view());

//         // 3. 处理结果
//         for (const bsoncxx::document::view& doc : cursor) {
//             // 3.1 获取_id字段（完全限定类型检查）
//             if (doc["_id"] && doc["_id"].type() == bsoncxx::type::k_string) {
//                 std::string id(doc["_id"].get_string().value.data()); // 显式构造string
//                 std::cout << "Found document _id: " << id << std::endl;
//             }

//             // 3.2 获取HGC_Emb数组字段（完全限定类型检查）
//             if (doc["HGC_Emb"] && doc["HGC_Emb"].type() == bsoncxx::type::k_array) {
//                 bsoncxx::array::view emb_array = doc["HGC_Emb"].get_array().value;

//                 // 3.3 提取数组中的浮点数
//                 std::vector<float> emb_values;
//                 for (const bsoncxx::array::element& elem : emb_array) {
//                     if (elem.type() == bsoncxx::type::k_double) {
//                         emb_values.push_back(static_cast<float>(elem.get_double().value));
//                     }
//                     else if (elem.type() == bsoncxx::type::k_int32) {
//                         emb_values.push_back(static_cast<float>(elem.get_int32().value));
//                     }
//                 }

//                 // 3.4 打印结果
//                 std::cout << "HGC_Emb values: ";
//                 for (float val : emb_values) {
//                     std::cout << val << " ";
//                 }
//                 std::cout << std::endl;
//             }
//         }
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return 1;
//     }
//     return 0;
// }

int main() {
    try{
        MySQLOperator& db = MySQLOperator::getInstance();

        if(!db.initialize()){
            std::cerr << "Failed to initialize database!" << std::endl;
            return 1;
        }

        // db.get_cpt_uid_id_of_area("are_3fee9e47d0f3428382f4afbcb1004117");

        KT kt(db);
        // lrn_aee0624932cf4affa00626e8f038c4e8
        // are_3fee9e47d0f3428382f4afbcb1004117
        auto ans = kt.forward(
            "are_3fee9e47d0f3428382f4afbcb1004117",
            "lrn_aee0624932cf4affa00626e8f038c4e8"
        );

        for(auto & item : ans){
            std::cout << item << " ";
        }
        std::cout << std::endl;
    }
    catch(const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

