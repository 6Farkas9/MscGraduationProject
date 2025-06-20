// include/data_simulator.h
#ifndef DATA_SIMULATOR_H
#define DATA_SIMULATOR_H

#include <string>
#include <ctime>
#include <crow/json.h>

namespace DataSimulator {
    crow::json::wvalue getPlatformStats();
    crow::json::wvalue getLearnerInfo(const std::string& uid);
    crow::json::wvalue getRecommendations(const std::string& uid);
    std::string getCurrentTime();
}

#endif // DATA_SIMULATOR_H