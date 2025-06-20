// src/api_handlers.cpp
#include "api_handlers.h"
#include "data_simulator.h"

namespace api{

void setupPlatformStatsRoutes(crow::SimpleApp& app) {
    // 获取平台统计数据
    CROW_ROUTE(app, "/api/platform-stats")
    .methods("GET"_method)
    ([](){
        return crow::response(DataSimulator::getPlatformStats());
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
