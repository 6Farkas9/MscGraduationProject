// src/data_simulator.cpp
#pragma execution_character_set("utf-8")
#include "data_simulator.h"
#include <random>
#include <map>

#include "MLSTimer.h"

namespace DataSimulator {

std::string getCurrentTime() {
    // time_t now = time(0);
    // tm* localtm = localtime(&now);
    // char buffer[80];
    // strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtm);
    return MLSTimer::getCurrentand30daysTime()[0];
}

crow::json::wvalue getPlatformStats() {
    crow::json::wvalue stats;
    
    // 模拟平台统计数据
    stats["domainCount"] = 12;
    stats["learnerCount"] = 856;
    stats["scenarioCount"] = 24;
    stats["knowledgePointCount"] = 342;
    stats["lastTrainingTime"] = getCurrentTime();
    stats["modelVersion"] = "v2.1.5";
    
    // 随机生成准确率 (80-95%)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(80, 95);
    stats["accuracy"] = dis(gen);
    
    return stats;
}

crow::json::wvalue getLearnerInfo(const std::string& uid) {
    crow::json::wvalue response;
    
    // 模拟学习者数据
    if (uid == "10001") {
        crow::json::wvalue learner;
        learner["uid"] = "10001";
        learner["email"] = "student10001@example.com";
        learner["phone"] = "13800010001";
        
        // 领域数据
        crow::json::wvalue domains = crow::json::wvalue::list();
        
        // 数学领域
        crow::json::wvalue math;
        math["id"] = "math";
        math["name"] = "数学";
        
        crow::json::wvalue mathPoints = crow::json::wvalue::list();
        mathPoints[0] = crow::json::wvalue{{"name", "代数基础"}, {"score", 0.85}};
        mathPoints[1] = crow::json::wvalue{{"name", "几何基础"}, {"score", 0.88}};
        mathPoints[2] = crow::json::wvalue{{"name", "概率统计"}, {"score", 0.92}};
        math["knowledgePoints"] = std::move(mathPoints);
        
        domains[0] = std::move(math);
        
        // 物理领域
        crow::json::wvalue physics;
        physics["id"] = "physics";
        physics["name"] = "物理";
        
        crow::json::wvalue physicsPoints = crow::json::wvalue::list();
        physicsPoints[0] = crow::json::wvalue{{"name", "力学"}, {"score", 0.72}};
        physicsPoints[1] = crow::json::wvalue{{"name", "电磁学"}, {"score", 0.75}};
        physicsPoints[2] = crow::json::wvalue{{"name", "热学"}, {"score", 0.78}};
        physics["knowledgePoints"] = std::move(physicsPoints);
        
        domains[1] = std::move(physics);
        
        learner["domains"] = std::move(domains);
        response["data"] = std::move(learner);
        response["code"] = 200;
        response["message"] = "获取成功";
    } 
    else if (uid == "10086") {
        crow::json::wvalue learner;
        learner["uid"] = "10002";
        learner["email"] = "student10002@example.com";
        learner["phone"] = "13800010002";
        
        // 领域数据
        crow::json::wvalue domains = crow::json::wvalue::list();
        
        // 化学领域
        crow::json::wvalue chemistry;
        chemistry["id"] = "chemistry";
        chemistry["name"] = "化学";
        
        crow::json::wvalue chemPoints = crow::json::wvalue::list();
        chemPoints[0] = crow::json::wvalue{{"name", "无机化学"}, {"score", 0.55}};
        chemPoints[1] = crow::json::wvalue{{"name", "有机化学"}, {"score", 0.65}};
        chemPoints[2] = crow::json::wvalue{{"name", "物理化学"}, {"score", 0.58}};
        chemistry["knowledgePoints"] = std::move(chemPoints);
        
        domains[0] = std::move(chemistry);
        
        // 生物领域
        crow::json::wvalue biology;
        biology["id"] = "biology";
        biology["name"] = "生物";
        
        crow::json::wvalue bioPoints = crow::json::wvalue::list();
        bioPoints[0] = crow::json::wvalue{{"name", "细胞生物学"}, {"score", 0.35}};
        bioPoints[1] = crow::json::wvalue{{"name", "遗传学"}, {"score", 0.45}};
        bioPoints[2] = crow::json::wvalue{{"name", "生态学"}, {"score", 0.38}};
        biology["knowledgePoints"] = std::move(bioPoints);
        
        domains[1] = std::move(biology);
        
        learner["domains"] = std::move(domains);
        response["data"] = std::move(learner);
        response["code"] = 200;
        response["message"] = "获取成功";
    }
    else {
        response["code"] = 404;
        response["message"] = "未找到该学习者的数据";
        response["data"] = nullptr;
    }
    
    return response;
}

crow::json::wvalue getRecommendations(const std::string& uid) {
    crow::json::wvalue response;
    crow::json::wvalue data;
    
    if (uid == "10001") {
        crow::json::wvalue knowledgePoints = crow::json::wvalue::list();
        knowledgePoints[0] = "高级算法";
        knowledgePoints[1] = "机器学习";
        knowledgePoints[2] = "深度学习基础";
        knowledgePoints[3] = "神经网络";
        knowledgePoints[4] = "计算机视觉";
        
        crow::json::wvalue studyPartners = crow::json::wvalue::list();
        studyPartners[0] = "10002";
        studyPartners[1] = "10003";
        studyPartners[2] = "10005";
        
        crow::json::wvalue studyModels = crow::json::wvalue::list();
        studyModels[0] = "10010";
        studyModels[1] = "10015";
        studyModels[2] = "10020";
        
        data["knowledgePoints"] = std::move(knowledgePoints);
        data["studyPartners"] = std::move(studyPartners);
        data["studyModels"] = std::move(studyModels);
    }
    else if (uid == "10002") {
        crow::json::wvalue knowledgePoints = crow::json::wvalue::list();
        knowledgePoints[0] = "生物化学";
        knowledgePoints[1] = "分子生物学";
        knowledgePoints[2] = "细胞工程";
        knowledgePoints[3] = "基因工程";
        knowledgePoints[4] = "生物信息学";
        
        crow::json::wvalue studyPartners = crow::json::wvalue::list();
        studyPartners[0] = "10001";
        studyPartners[1] = "10004";
        studyPartners[2] = "10006";
        
        crow::json::wvalue studyModels = crow::json::wvalue::list();
        studyModels[0] = "10011";
        studyModels[1] = "10016";
        studyModels[2] = "10021";
        
        data["knowledgePoints"] = std::move(knowledgePoints);
        data["studyPartners"] = std::move(studyPartners);
        data["studyModels"] = std::move(studyModels);
    }
    else {
        // 默认推荐
        crow::json::wvalue knowledgePoints = crow::json::wvalue::list();
        knowledgePoints[0] = "默认知识点1";
        knowledgePoints[1] = "默认知识点2";
        knowledgePoints[2] = "默认知识点3";
        
        crow::json::wvalue studyPartners = crow::json::wvalue::list();
        studyPartners[0] = "默认伙伴1";
        studyPartners[1] = "默认伙伴2";
        
        crow::json::wvalue studyModels = crow::json::wvalue::list();
        studyModels[0] = "默认榜样1";
        studyModels[1] = "默认榜样2";
        
        data["knowledgePoints"] = std::move(knowledgePoints);
        data["studyPartners"] = std::move(studyPartners);
        data["studyModels"] = std::move(studyModels);
    }
    
    response["code"] = 200;
    response["message"] = "获取成功";
    response["data"] = std::move(data);
    
    return response;
}

} // namespace DataSimulator