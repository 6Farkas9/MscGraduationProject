#ifndef CONCEPT_SERVICE_H
#define CONCEPT_SERVICE_H

#include "MongoDBOperator.h"

#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <filesystem>
#include <windows.h>
#include <unordered_set>
#include <unordered_map>
#include <sstream>

#include "MySQLOperator.h"
#include "MLS_config.h"
#include "MLSTimer.h"
#include "UidCreator.h"

class ConceptService{

public:
    ConceptService(MySQLOperator &mysqlop, MongoDBOperator &mongodbop);
    ~ConceptService();

    std::string addOneConcept(
        std::string &are_uid, 
        std::vector<std::string> &pre_cpt_uids, 
        std::vector<std::string> &aft_cpt_uids,
        std::string &name
    );

    bool deleteOneConcept(std::string cpt_uid);

private:
    MySQLOperator &mysqlop; 
    MongoDBOperator &mongodbop;
};

#endif //ifndef CONCEPT_SERVICE_H