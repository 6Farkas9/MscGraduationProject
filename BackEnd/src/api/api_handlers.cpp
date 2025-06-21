// src/api_handlers.cpp
#include "api_handlers.h"
#include "data_simulator.h"

namespace api{

void setupPlatformStatsRoutes(crow::SimpleApp& app) {
    // 获取平台统计数据
    CROW_ROUTE(app, "/api/platform-stats")
    .methods("GET"_method)
    ([](){
        PlatformStatsService plat_ser;
        auto plat_data = plat_ser.get_count_data();

        crow::json::wvalue stats;
        stats["are_num"] = std::move(plat_data["are_num"]);
        stats["lrn_num"] = std::move(plat_data["lrn_num"]);
        stats["scn_num"] = std::move(plat_data["scn_num"]);
        stats["cpt_num"] = std::move(plat_data["cpt_num"]);
        stats["ict_num"] = std::move(plat_data["ict_num"]);

        // auto sim_data = DataSimulator::getPlatformStats();
        auto dl_meta_data = plat_ser.get_deeplearning_data();

        stats["lastTrainingTime"] = std::move(dl_meta_data["lastTrainingTime"]);
        stats["modelVersion"] = std::move(dl_meta_data["modelVersion"]);

        return crow::response(stats);
    });
    
    // 触发模型训练
    CROW_ROUTE(app, "/api/trigger-training")
    .methods("POST"_method)
    ([](){
        // 模拟训练触发
        crow::json::wvalue response;
        response["message"] = "训练任务已启动";
        response["success"] = true;
        return crow::response(response);
    });
}

void setupLearnerInfoRoutes(crow::SimpleApp& app) {
    // 获取学习者信息
    CROW_ROUTE(app, "/api/learner-info/<string>")
    .methods("GET"_method)
    ([](const crow::request& req, std::string uid){
        return crow::response(DataSimulator::getLearnerInfo(uid));
    });
    
    // 获取推荐内容
    CROW_ROUTE(app, "/api/recommendations/<string>")
    .methods("GET"_method)
    ([](const crow::request& req, std::string uid){
        return crow::response(DataSimulator::getRecommendations(uid));
    });
}

}
