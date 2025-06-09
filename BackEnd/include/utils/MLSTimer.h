#ifndef MLSTIMER_H
#define MLSTIMER_H

#include <iostream>
#include <chrono>
#include <iomanip>  // 用于 std::put_time
#include <ctime>    // 用于 std::localtime 和 std::strftime
#include <vector>
#include <string>

class MLSTimer{
public:
    MLSTimer();
    ~MLSTimer();

    static std::vector<std::string> getCurrentand30daysTime();

private:
    static std::string format_time(const std::chrono::system_clock::time_point& time_point);
};

#endif // #ifndef MLSTIMER_H