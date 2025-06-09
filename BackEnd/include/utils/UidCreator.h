#ifndef UID_CREATOR_H
#define UID_CREATOR_H

#include <random>
#include <string>
#include <sstream>
#include <iomanip>

class UidCreator {

public:
    UidCreator();
    ~UidCreator();

    static std::string generate_uuid_winapi();
};

#endif //UID_CREATOR_H