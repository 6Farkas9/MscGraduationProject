#include "UidCreator.h"

UidCreator::UidCreator() {

}

UidCreator::~UidCreator() {

}

std::string UidCreator::generate_uuid_winapi() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint16_t> dis(0, 15);
    
    const char hex_chars[] = "0123456789abcdef";
    std::string uuid(32, '0');
    
    // 生成随机十六进制字符
    for (char& c : uuid) {
        c = hex_chars[dis(gen)];
    }
    
    // 设置版本号和变体位
    // 版本4（第13个字符）
    uuid[12] = '4';
    // 变体1（第17个字符的高位设为1）
    uuid[16] = hex_chars[(uuid[16] & 0x3) | 0x8];
    
    return uuid;
}