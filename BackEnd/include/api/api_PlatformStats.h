#ifndef API_PLATFORM_STATS_H
#define API_PLATFORM_STATS_H

#include "crow.h"
#include "crow/json.h"
#include "PlatformStatsService.h"

class api_PlatformStats{
public:
    static void setup_stats_routes(crow::SimpleApp& app); 
    static void setup_triggertrain_routes(crow::SimpleApp& app);
};

#endif