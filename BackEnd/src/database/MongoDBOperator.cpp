#include "MongoDBOperator.h"
#include <iostream>

using namespace bsoncxx::builder::stream;

// 初始化静态成员
std::mutex MongoDBOperator::instanceMutex_;

// PIMPL实现结构体
struct MongoDBOperator::Impl {
    std::unique_ptr<mongocxx::instance> instance;  // MongoDB驱动实例
    std::unique_ptr<mongocxx::pool> pool;          // 连接池
    std::string databaseName;                      // 数据库名
};

// 单例实例获取
MongoDBOperator& MongoDBOperator::getInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    static MongoDBOperator instance;
    return instance;
}

// 构造函数
MongoDBOperator::MongoDBOperator() : pImpl_(std::make_unique<Impl>()) {
    // 驱动实例在initialize中创建
}

// 析构函数
MongoDBOperator::~MongoDBOperator() {
    close();
}

// 初始化数据库连接
bool MongoDBOperator::initialize() {
    try {
        pImpl_->instance = std::make_unique<mongocxx::instance>();
        pImpl_->pool = std::make_unique<mongocxx::pool>(mongocxx::uri{"mongodb://localhost:27017"});
        pImpl_->databaseName = "MLS_db";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Connection Error: " << e.what() << std::endl;
        return false;
    }
}

// 检查连接状态
bool MongoDBOperator::isConnected() const {
    return pImpl_->pool != nullptr;
}

// 关闭连接池
void MongoDBOperator::close() {
    pImpl_->pool.reset();
    pImpl_->instance.reset();
}

// ========== 通用操作方法 ==========

std::optional<bsoncxx::document::value> MongoDBOperator::findOne(
    const std::string& collection, 
    bsoncxx::document::view_or_value filter,
    bsoncxx::document::view_or_value projection) {
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        
        auto opts = mongocxx::options::find{};
        if (!projection.view().empty()) {
            opts.projection(projection.view());
        }
        
        auto result = coll.find_one(filter.view(), opts);
        if (result) {
            return std::make_optional<bsoncxx::document::value>(std::move(*result));
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Query Error: " << e.what() << std::endl;
        throw;
    }
}

std::vector<bsoncxx::document::value> MongoDBOperator::findMany(
    const std::string& collection, 
    bsoncxx::document::view_or_value filter,
    bsoncxx::document::view_or_value projection,
    std::optional<int64_t> limit) {
    
    std::vector<bsoncxx::document::value> results;
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        
        auto opts = mongocxx::options::find{};
        if (!projection.view().empty()) {
            opts.projection(projection.view());
        }
        if (limit) {
            opts.limit(*limit);
        }
        
        auto cursor = coll.find(filter.view(), opts);
        
        for (auto&& doc : cursor) {
            results.emplace_back(bsoncxx::document::value{doc});
        }
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Query Error: " << e.what() << std::endl;
        throw;
    }
    
    return results;
}

std::optional<bsoncxx::document::value> MongoDBOperator::insertOne(
    const std::string& collection, 
    bsoncxx::document::view_or_value document) {
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        
        auto result = coll.insert_one(document.view());
        if (result) {  // 检查 optional 是否有值
            auto id = result->inserted_id();
            if (id.type() == bsoncxx::type::k_oid) {  // 检查类型是否正确
                auto doc = bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("_id", id.get_oid().value));
                return doc;
            }
        }
        return std::nullopt;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Insert Error: " << e.what() << std::endl;
        return std::nullopt;
    }
}

bool MongoDBOperator::updateOne(
    const std::string& collection, 
    bsoncxx::document::view_or_value filter,
    bsoncxx::document::view_or_value update,
    bool upsert) {
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        
        auto opts = mongocxx::options::update{};
        opts.upsert(upsert);
        
        auto result = coll.update_one(filter.view(), update.view(), opts);
        return result && result->modified_count() > 0;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Update Error: " << e.what() << std::endl;
        return false;
    }
}

bool MongoDBOperator::deleteOne(
    const std::string& collection, 
    bsoncxx::document::view_or_value filter) {
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        
        auto result = coll.delete_one(filter.view());
        return result && result->deleted_count() > 0;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Delete Error: " << e.what() << std::endl;
        return false;
    }
}

// ========== 业务方法示例 ==========

std::optional<std::unordered_map<std::string, float>> MongoDBOperator::testGetLearnerInfo(const std::string& lrn_uid) {
    auto filter = document{} 
        << "_id" << lrn_uid 
        << finalize;
    
    auto result = findOne("learners", filter.view());
    if (result == std::nullopt){
        return std::nullopt;
    }
    auto view = result->view();
    auto kt = view["KT"];

    if (kt.type() != bsoncxx::type::k_document) {
        throw std::runtime_error("KT field is not a document");
    }

    auto kt_doc = kt.get_document().view();
    std::unordered_map<std::string, float> kt_map;

    for (const auto& elem : kt_doc) {
        kt_map[elem.key().data()] = elem.get_double().value;
    }

    return std::make_optional<std::unordered_map<std::string, float>>(std::move(kt_map));
}

std::optional<std::string> MongoDBOperator::logUserActivity(
    const std::string& userId, 
    const std::string& activityType,
    const std::string& details) {
    
    auto doc = document{} 
        << "user_id" << userId
        << "activity_type" << activityType
        << "details" << details
        << "timestamp" << bsoncxx::types::b_date{std::chrono::system_clock::now()}
        << finalize;
    
    auto result = insertOne("user_activities", doc.view());
    if (result) {
        if (auto id = result->view()["_id"]) {
            return id.get_oid().value.to_string();
        }
    }
    return std::nullopt;
}

std::vector<std::string> MongoDBOperator::getUserFavorites(const std::string& userId) {
    std::vector<std::string> favorites;
    
    auto filter = document{} 
        << "user_id" << userId 
        << finalize;
    
    auto projection = document{} 
        << "favorite_items" << 1 
        << "_id" << 0 
        << finalize;
    
    auto result = findOne("user_preferences", filter.view(), projection.view());
    
}