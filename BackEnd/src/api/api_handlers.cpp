// src/api_handlers.cpp
#include "api_handlers.h"
#include "data_simulator.h"

namespace api{

void setupPlatformStatsRoutes(crow::SimpleApp& app) {
    // 获取平台统计数据
    CROW_ROUTE(app, "/api/platform-stats")
    .methods("GET"_method)
    ([](){
        PlatformStatsApplication plat_ser;
        auto plat_data = plat_ser.get_count_data();

        crow::json::wvalue stats;
        stats["are_num"] = std::move(plat_data["are_num"]);
        stats["lrn_num"] = std::move(plat_data["lrn_num"]);
        stats["unt_num"] = std::move(plat_data["unt_num"]);
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
    ([](const crow::request& req, std::string lrn_uid){
        // 获取该学习者在各领域下的各知识点上的表现
        // std::string lrn_uid = "lrn_aee0624932cf4affa00626e8f038c4e8";
        std::unordered_map<std::string, std::string> lrn_info;
        std::unordered_map<
            std::string, std::pair<
                std::string, std::unordered_map<std::string, std::pair<
                    std::string, float>>>> are_info;
        
        crow::json::wvalue response;

        LearnerInfoApplication lrninfo_ser;
        if (!lrninfo_ser.get_lrn_info(
            lrn_uid,
            lrn_info,
            are_info
        )) {
            response["code"] = 404;
            response["message"] = "未找到该学习者的数据";
            response["data"] = nullptr;
            return crow::response(response);
        }
        
        // 基础数据
        crow::json::wvalue learner;
        learner["lrn_uid"] = lrn_uid;
        learner["email"] = std::move(lrn_info["email"]);
        learner["phone"] = std::move(lrn_info["phone"]);

        // 领域数据
        crow::json::wvalue areas = crow::json::wvalue::list();
        int are_idx = 0;
        for (auto &are_uid2are_data : are_info) {
            crow::json::wvalue single_area_data;
            single_area_data["are_uid"] = std::move(are_uid2are_data.first);
            single_area_data["are_name"] = std::move(are_uid2are_data.second.first);

            crow::json::wvalue single_are_cpt_data = crow::json::wvalue::list();
            int cpt_idx = 0;
            for (auto & cpt_uid2cpt_data : are_uid2are_data.second.second) {
                single_are_cpt_data[cpt_idx++] = crow::json::wvalue{
                    {"cpt_uid", std::move(cpt_uid2cpt_data.first)},
                    {"cpt_name", std::move(cpt_uid2cpt_data.second.first)}, 
                    {"score", std::move(cpt_uid2cpt_data.second.second)}
                };
            }
            single_area_data["cpt_data"] = std::move(single_are_cpt_data);
            areas[are_idx++] = std::move(single_area_data);
        }

        learner["are_data"] = std::move(areas);

        response["data"] = std::move(learner);
        response["code"] = 200;
        response["message"] = "获取成功";

        return crow::response(response);
    });
    
    // 获取推荐内容
    CROW_ROUTE(app, "/api/recommendations/<string>")
    .methods("GET"_method)
    ([](const crow::request& req, std::string lrn_uid) {
        std::unordered_map<std::string, std::string> cpt_uids;
        std::vector<std::string> lrn_partners;
        std::vector<std::string> lrn_models;

        crow::json::wvalue response;
        crow::json::wvalue data;

        LearnerInfoApplication lrn_info_ser;
        if (!lrn_info_ser.get_recommend_info(
            lrn_uid,
            cpt_uids,
            lrn_partners,
            lrn_models
        )) {
            response["code"] = 404;
            response["message"] = "获取失败";
            response["data"] = std::move(data);
        }
        crow::json::wvalue cpts = crow::json::wvalue::list();
        int idx = 0;
        for (auto &cpt_uid2name : cpt_uids) {
            cpts[idx++] = std::move(cpt_uid2name.second);
        }
        
        crow::json::wvalue studyPartners = crow::json::wvalue::list();
        idx = 0;
        for (auto &lrn_p_uid : lrn_partners) {
            studyPartners[idx++] = std::move(lrn_p_uid);
        }
        
        crow::json::wvalue studyModels = crow::json::wvalue::list();
        idx = 0;
        for (auto &lrn_m_uid : lrn_models) {
            studyModels[idx++] = std::move(lrn_m_uid);
        }
        
        data["cpts"] = std::move(cpts);
        data["studyPartners"] = std::move(studyPartners);
        data["studyModels"] = std::move(studyModels);

        response["code"] = 200;
        response["message"] = "获取成功";
        response["data"] = std::move(data);


        return crow::response(response);
    });
}

}
