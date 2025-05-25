#include "CD.h"

CD::CD(DBOperator &db) :
    db(db)
{
    auto twotime = MLSTimer().getCurrentand30daysTime();
    now_time = twotime[0];
    thirty_days_ago_time = twotime[1];

    // std::cout << now_time << std::endl;
    // std::cout << thirty_days_ago_time << std::endl;
}

CD::~CD(){

}

