#include <iostream>
#include <fstream>
#include <string>
#include "MLS_config.h"
// #include "MongoDBOperator.h"
// #include "MySQLOperator.h"
// #include "UidCreator.h"
// #include "SceneService.h"
// #include "LearnerService.h"
// #include "ConceptService.h"

#include "crow.h"
//#include "crow/middlewares/cors.h"

// int main() {
//     MySQLOperator& mysqlop = MySQLOperator::getInstance();
//     mysqlop.initialize();
//     MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
//     mongodbop.initialize();

//     LearnerService lrn_ser = LearnerService(mysqlop, mongodbop);

//     auto res_kt = lrn_ser.predict_lrn_kt_in_are(
//         "lrn_aee0624932cf4affa00626e8f038c4e8",
//         "are_3fee9e47d0f3428382f4afbcb1004117"
//     );

//     for (auto & kv : res_kt) {
//         std::cout << kv.first << " - " << kv.second << std::endl;
//     }
//     std::cout << "KT finished" << std::endl;

//     auto res_cd = lrn_ser.predict_lrn_cd_in_are(
//         "lrn_aee0624932cf4affa00626e8f038c4e8",
//         "are_3fee9e47d0f3428382f4afbcb1004117"
//     );

//     for (auto & kv : res_cd) {
//         std::cout << kv.first << " - " << kv.second << std::endl;
//     }
//     std::cout << "CD finished" << std::endl;

//     auto res_rr = lrn_ser.predict_lrn_rr(
//         "lrn_aee0624932cf4affa00626e8f038c4e8"
//     );

//     for (auto & kv : res_rr) {
//         std::cout << kv.first << " - " << kv.second << std::endl;
//     }
//     std::cout << "RR finished" << std::endl;
// }

std::string read_file_content(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        return "";
    }
    return std::string(std::istreambuf_iterator<char>(file), 
                      std::istreambuf_iterator<char>());
}

int main() {
    crow::SimpleApp app;

    // 1. 处理静态资源
    CROW_ROUTE(app, "/assets/<string>")
    ([](const std::string& filename) {
        std::string file_path = std::string(FRONTEND_ROOT) + "/dist/assets/" + filename;
        std::string content = read_file_content(file_path);
        
        if (content.empty()) {
            return crow::response(404);
        }
        
        crow::response res(content);
        if (filename.find(".js") != std::string::npos) {
            res.set_header("Content-Type", "application/javascript");
        } 
        else if (filename.find(".css") != std::string::npos) {
            res.set_header("Content-Type", "text/css");
        }
        return res;
    });

    // 2. 处理根路径
    CROW_ROUTE(app, "/")
    ([]() {
        std::string file_path = std::string(FRONTEND_ROOT) + R"(\dist\index.html)";
        std::cout << file_path << std::endl;
        std::string content = read_file_content(file_path);
        
        if (content.empty()) {
            return crow::response(404);
        }
        
        crow::response res(content);
        res.set_header("Content-Type", "text/html");
        return res;
    });

    // 3. 处理前端路由（Vue Router history模式）
    CROW_ROUTE(app, "/<string>")
    ([](const std::string&) {  // 参数忽略，统一返回index.html
        std::string file_path = std::string(FRONTEND_ROOT) + R"(\dist\index.html)";
        std::string content = read_file_content(file_path);
        
        if (content.empty()) {
            return crow::response(404);
        }
        
        crow::response res(content);
        res.set_header("Content-Type", "text/html");
        return res;
    });

    // 4. 处理favicon
    CROW_ROUTE(app, "/favicon.ico")
    ([]() {
        std::string file_path = std::string(FRONTEND_ROOT) + R"(\dist\favicon.ico)";
        std::string content = read_file_content(file_path);
        
        if (content.empty()) {
            return crow::response(404);
        }
        
        crow::response res(content);
        res.set_header("Content-Type", "image/x-icon");
        return res;
    });

    // 启动服务器
    std::cout << "Server running on http://localhost:8080\n";
    app.port(8080).multithreaded().run();
}

// int main() {
//     MySQLOperator& mysqlop = MySQLOperator::getInstance();
//     mysqlop.initialize();
//     MongoDBOperator &mongodbop = MongoDBOperator::getInstance();
//     mongodbop.initialize();

//     ConceptService cpt_ser = ConceptService(mysqlop, mongodbop);

//     std::string are_uid = "are_3fee9e47d0f3428382f4afbcb1004117";
//     std::string name = "test_name";

//     std::unordered_set<std::string> pre_cpt, aft_cpt;

//     pre_cpt.insert("cpt_f4e10b32f85746d7900fdbff3b27276e");
//     pre_cpt.insert("cpt_5a315add91b0469f8537cb37feb0dc0c");
//     pre_cpt.insert("cpt_a86e9f3aff6245979bf1c8a9454b5dde");

//     aft_cpt.insert("cpt_f361c531a42048c18d55769b782c3fd5");
//     aft_cpt.insert("cpt_c7d6af53b25944e6bb3b990c3076df05");
//     aft_cpt.insert("cpt_92b35b614dbc4dcfaa4f3592f8e6d0cd");

//     std::string cpt_uid = cpt_ser.addOneConcept(
//         are_uid,
//         pre_cpt,
//         aft_cpt,
//         name
//     );

//     std::cout << cpt_uid << std::endl;

//     std::string scn_uid = "scn_6653ca78e6f74b8088769c4a08dc6784";
//     mysqlop.insert_one_scn_cpt_to_graph_involve(
//         scn_uid,
//         cpt_uid,
//         0.3
//     );

//     cpt_ser.deleteOneConcept(cpt_uid);

//     // std::string to_del = "cpt_3ef8d6ab595b49e0b26512e57f8ec9e8";
//     // cpt_ser.deleteOneConcept(to_del);

//     return 0;

//     // LearnerService lrn_ser = LearnerService(mysqlop, mongodbop);

//     // auto res = lrn_ser.predict_lrn_rr(
//     //     "lrn_aee0624932cf4affa00626e8f038c4e8"
//     // );

//     // std::cout << res.size() << std::endl;

//     // for (auto & kv : res) {
//     //     std::cout << kv.first << " , " << kv.second << std::endl;
//     // }

// }

// "are_3fee9e47d0f3428382f4afbcb1004117"

// _id
// "are_3fee9e47d0f3428382f4afbcb1004117"

// KCGE_Emb
// Array (32)
// 0
// 0.000778811052441597
// 1
// 0.017386481165885925
// 2
// 0.018265951424837112
// 3
// 0.005330365616828203
// 4
// -0.00029044775874353945
// 5
// 0.006320205982774496
// 6
// 0.0017165003810077906
// 7
// 0.013655570335686207
// 8
// 0.0011236343998461962
// 9
// 0.0021221523638814688
// 10
// -0.00008515048830304295
// 11
// 0.006304697133600712
// 12
// -0.0000921031751204282
// 13
// 0.00004258104308973998
// 14
// 0.017691681161522865
// 15
// 0.0026335851289331913
// 16
// -0.0002605313202366233
// 17
// 0.0119218984618783
// 18
// -0.00001300366420764476
// 19
// 0.0008939251420088112
// 20
// 0.0085999621078372
// 21
// -0.000030758059438085184
// 22
// -0.00010373058466939256
// 23
// -0.00003692611790029332
// 24
// 0.022111844271421432
// 25
// 0.025210900232195854
// 26
// 0.005639111623167992
// 27
// 0.0011709787650033832
// 28
// -0.00010438965546200052
// 29
// 0.0007192904013209045
// 30
// 0.00008912644989322871
// 31
// -0.00012165604857727885
