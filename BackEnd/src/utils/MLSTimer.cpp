#include "MLSTimer.h"

MLSTimer::MLSTimer(){

}
MLSTimer::~MLSTimer(){

}

std::string MLSTimer::format_time(const std::chrono::system_clock::time_point& time_point) {
    std::time_t time = std::chrono::system_clock::to_time_t(time_point); // 将时间转换为 std::time_t
    std::tm tm;
    localtime_s(&tm, &time);
    char buffer[20];  // 用来存储格式化后的字符串
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm); // 格式化时间
    return std::string(buffer);  // 返回格式化后的时间字符串
}

std::vector<std::string> MLSTimer::getCurrentand30daysTime(){
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    // 获取30天前的时间
    auto thirty_days_ago = now - std::chrono::hours(24 * 30);

    std::string now_str = format_time(now);
    std::string thirty_days_ago_str = format_time(thirty_days_ago);
    
    // 格式化输出
    // std::cout << now_str << std::endl;
    // std::cout << thirty_days_ago_str << std::endl;

    return std::vector<std::string>{now_str, thirty_days_ago_str};
}