#ifndef PLATFORM_STATS_SERVICE_H
#define PLATFORM_STATS_SERVICE_H

#include <crow.h>

class PlatformStatsService {
public:
    static crow::json::wvalue get_platform_stats();

};

#endif