#include "api_PlatformStats.h"

void api_PlatformStats::setup_stats_routes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/api/platform-stats")
    ([]() {
        // auto stats = core::get_platform_stats();
        crow::json::wvalue stats;
        stats["domainCount"] = 12;
        stats["learnerCount"] = 8562;
        stats["scenarioCount"] = 34;
        stats["knowledgePointCount"] = 1258;
        stats["lastTrainingTime"] = "2023-11-15 14:30:22";
        stats["modelVersion"] = "v2.3.1";
        stats["accuracy"] = 87.5;
        crow::response res(stats);
        res.set_header("Content-Type", "application/json");
        return res;
    });
}

void api_PlatformStats::setup_triggertrain_routes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/api/trigger-training")
    .methods("POST"_method)
    ([](const crow::request& req) {
        // 模拟训练触发逻辑
        crow::json::wvalue response;
        response["message"] = "success";
        response["success"] = true;

        crow::response res(response);
        res.set_header("Content-Type", "application/json");
        return res;
    });
}
