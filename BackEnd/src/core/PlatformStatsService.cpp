#include "PlatformStatsService.h"

crow::json::wvalue PlatformStatsService::get_platform_stats() {
    crow::json::wvalue stats;
    stats["domainCount"] = 12;
    stats["learnerCount"] = 8562;
    stats["scenarioCount"] = 34;
    stats["knowledgePointCount"] = 1258;
    stats["lastTrainingTime"] = "2023-11-15 14:30:22";
    stats["modelVersion"] = "v2.3.1";
    stats["accuracy"] = 87.5;
    return stats;
}