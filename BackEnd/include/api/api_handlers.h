// include/api_handlers.h
#ifndef API_HANDLERS_H
#define API_HANDLERS_H

#include "crow.h"
#include "PlatformStatsService.h"

namespace api {

    void setupPlatformStatsRoutes(crow::SimpleApp& app);

    void setupLearnerInfoRoutes(crow::SimpleApp& app);
}

#endif // API_HANDLERS_H