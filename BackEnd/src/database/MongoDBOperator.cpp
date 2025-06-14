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

std::optional<mongocxx::cursor> MongoDBOperator::findMany(
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

        return std::make_optional<mongocxx::cursor>(std::move(cursor));

    } catch (const std::exception& e) {
        std::cerr << "MongoDB Query Error: " << e.what() << std::endl;
        throw;
    }
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

std::optional<std::vector<bsoncxx::document::value>> MongoDBOperator::insertMany(
    const std::string& collection,
    const std::vector<bsoncxx::document::view_or_value>& documents) {
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }

    if (documents.empty()) {
        return std::vector<bsoncxx::document::value>{}; // 空输入返回空数组
    }

    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];

        // 转换为view数组
        std::vector<bsoncxx::document::view> views;
        views.reserve(documents.size());
        for (const auto& doc : documents) {
            views.push_back(doc.view());
        }

        auto result = coll.insert_many(views);
        if (!result) {
            return std::nullopt;
        }

        // 构建返回结果（包含所有生成的_id）
        std::vector<bsoncxx::document::value> insertedIds;
        for (const auto& idPair : result->inserted_ids()) {
            auto doc = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("_id", idPair.second.get_oid().value));
            insertedIds.emplace_back(std::move(doc));
        }

        return insertedIds;
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

int MongoDBOperator::updateMany(
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

        auto result = coll.update_many(filter.view(), update.view(), opts);
        if (!result) {
            return -1;
        }

        // 计算受影响总数 = 修改数 + (upsert是否发生?1:0)
        int affectedCount = result->modified_count();
        if (upsert && result->upserted_id()) {
            affectedCount += 1;
        }

        return affectedCount;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Update Error: " << e.what() << std::endl;
        return -1;
    }
}

int MongoDBOperator::bulkUpdateMany(
    const std::string& collection,
    const std::vector<std::pair<
        bsoncxx::document::view_or_value, // filter
        bsoncxx::document::view_or_value   // update
    >>& filter_updates,
    bool upsert
) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }

    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        mongocxx::options::bulk_write bulk_opts;
        bulk_opts.ordered(false);
        std::vector<mongocxx::model::update_many> bulk_ops;

        for (const auto& [filter, update] : filter_updates) {
            mongocxx::model::update_many op(filter, update);
            op.upsert(upsert); // 设置是否启用 upsert
            bulk_ops.emplace_back(std::move(op));
        }
        auto result = coll.bulk_write(bulk_ops, bulk_opts);
        return result->modified_count() + result->upserted_count();
    } catch (const bsoncxx::exception& e) {  // 正确：bsoncxx 确实有 exception 类
        std::cerr << "bulk update BSON Error: " << e.what() << std::endl;
        return -2;
    } catch (const mongocxx::exception& e) {  // 特定操作异常
        std::cerr << "bulk update MongoDB Operation Failed: " << e.what() << std::endl;
        return -3;
    }catch (const std::exception& e) {  // 兜底
        std::cerr << "System Error: " << e.what() << std::endl;
        return -1;
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

int MongoDBOperator::deleteMany(
    const std::string& collection, 
    bsoncxx::document::view_or_value filter) {
    
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        auto client = pImpl_->pool->acquire();
        auto db = (*client)[pImpl_->databaseName];
        auto coll = db[collection];
        
        auto result = coll.delete_many(filter.view());
        return result ? result->deleted_count() : 0;
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Delete Error: " << e.what() << std::endl;
        return -1;  // 用-1表示错误
    }
}

// ========== 业务方法示例 ==========

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_are_kcge_by_are_uid(const std::unordered_set<std::string> &are_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        // 1. 构建查询条件：_id 在 scn_uids 集合中
        bsoncxx::builder::basic::array in_array;
        for (const auto& uid : are_uids) {
            in_array.append(uid);
        }
        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("KCGE_Emb", 1)
        ); // 返回KCGE字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("areas", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["KCGE_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;
    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_scn_kcge_by_scn_uid(const std::unordered_set<std::string> &scn_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        // 1. 构建查询条件：_id 在 scn_uids 集合中
        bsoncxx::builder::basic::array in_array;
        for (const auto& uid : scn_uids) {
            in_array.append(uid);
        }
        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("KCGE_Emb", 1)
        ); // 返回KCGE字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("scenes", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["KCGE_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;
    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_cpt_kcge_by_cpt_uid(const std::unordered_set<std::string> &cpt_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        // 1. 构建查询条件：_id 在 scn_uids 集合中
        bsoncxx::builder::basic::array in_array;
        for (const auto& uid : cpt_uids) {
            in_array.append(uid);
        }
        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("KCGE_Emb", 1)
        ); // 返回KCGE字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("concepts", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["KCGE_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;

    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_lrn_hgc_by_lrn_uid(const std::unordered_set<std::string> &lrn_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        // 1. 构建查询条件：_id 在 lrn_uids 集合中
        bsoncxx::builder::basic::array in_array;
        for (const auto& uid : lrn_uids) {
            in_array.append(uid);
        }
        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("HGC_Emb", 1)
        ); // 返回HGC字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("learners", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["HGC_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;

    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_scn_hgc_by_scn_uid(const std::unordered_set<std::string> &scn_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        // 1. 构建查询条件：_id 在 lrn_uids 集合中
        bsoncxx::builder::basic::array in_array;
        for (const auto& uid : scn_uids) {
            in_array.append(uid);
        }
        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("HGC_Emb", 1)
        ); // 返回HGC字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("scenes", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["HGC_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;

    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_cpt_hgc_by_cpt_uid(const std::unordered_set<std::string> &cpt_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        // 1. 构建查询条件：_id 在 lrn_uids 集合中
        bsoncxx::builder::basic::array in_array;
        for (const auto& uid : cpt_uids) {
            in_array.append(uid);
        }
        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("HGC_Emb", 1)
        ); // 返回HGC字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("concepts", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["HGC_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;

    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

std::unordered_map<std::string, std::vector<float>> MongoDBOperator::get_all_cpt_hgc() {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    try {
        auto filter = bsoncxx::builder::basic::make_document();
        // 2. 内部构造其他参数
        auto projection = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 1),
            bsoncxx::builder::basic::kvp("HGC_Emb", 1)
        ); // 返回HGC字段
        std::optional<int64_t> limit = std::nullopt; // 不限制结果数量
        std::optional<mongocxx::cursor> res = findMany("concepts", filter.view(), projection.view(), limit);
        std::unordered_map<std::string, std::vector<float>> ans;
        if (res == std::nullopt || res->begin() == res->end()) {
            return ans;
        }
        for (auto &&doc : *res){
            std::string scn_uid = doc["_id"].get_string().value.data();
            ans[scn_uid] = std::vector<float>();
            for (auto & ele : doc["HGC_Emb"].get_array().value){
                ans[scn_uid].emplace_back(ele.get_double().value);
            }
        }
        return ans;

    } catch (const std::exception& e) {
        std::cerr << "System error: " << e.what() << std::endl;
        throw;
    }
}

int MongoDBOperator::delete_scn_from_scenes(const std::vector<std::string> &scn_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        // 构建过滤条件：_id等于输入字符串
        auto in_array = bsoncxx::builder::basic::array{};
        for (const auto& id : scn_uids) {
            in_array.append(id);
        }

        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        
        return deleteMany("scenes", filter.view());
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Delete Scene Error: " << e.what() << std::endl;
        return -1;
    }
}

int MongoDBOperator::delete_cpt_from_concepts(const std::vector<std::string> &cpt_uids) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }
    
    try {
        // 构建过滤条件：_id等于输入字符串
        auto in_array = bsoncxx::builder::basic::array{};
        for (const auto& id : cpt_uids) {
            in_array.append(id);
        }

        auto filter = bsoncxx::builder::basic::make_document(
            bsoncxx::builder::basic::kvp("_id", 
                bsoncxx::builder::basic::make_document(
                    bsoncxx::builder::basic::kvp("$in", in_array)
                )
            )
        );
        
        return deleteMany("concepts", filter.view());
    } catch (const std::exception& e) {
        std::cerr << "MongoDB Delete Scene Error: " << e.what() << std::endl;
        return -1;
    }
}

int MongoDBOperator::update_cpt_kcge_emb(const std::unordered_map<std::string, std::vector<float>> &cpt_emb) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }

    try {
        std::vector<std::pair<
            bsoncxx::document::view_or_value,
            bsoncxx::document::view_or_value
        >> filter_updates;

        for (const auto& [key, vec] : cpt_emb) {
            // 构造 filter: { _id: key }
            auto filter = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("_id", key)
            );

            bsoncxx::builder::basic::array array_builder;
            for (const auto& v : vec) {
                array_builder.append(static_cast<double>(v));  // float -> double
            }
            auto update = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("$set", 
                    bsoncxx::builder::basic::make_document(
                        bsoncxx::builder::basic::kvp("KCGE_Emb", array_builder)
                    )
                )
            );

            filter_updates.emplace_back(std::move(filter), std::move(update));
        }

        // 2. 调用批量更新
        return bulkUpdateMany("concepts", filter_updates, true);
    } catch (const bsoncxx::exception& e) {  // 正确：bsoncxx 确实有 exception 类
        std::cerr << "cpt update BSON Error: " << e.what() << std::endl;
        return -2;
    } catch (const mongocxx::exception& e) {  // 特定操作异常
        std::cerr << "cpt update MongoDB Operation Failed: " << e.what() << std::endl;
        return -3;
    }catch (const std::exception& e) {  // 兜底
        std::cerr << "System Error: " << e.what() << std::endl;
        return -1;
    }
}

int MongoDBOperator::update_are_kcge_emb(const std::unordered_map<std::string, std::vector<float>> &are_emb) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }

    try {
        std::vector<std::pair<
            bsoncxx::document::view_or_value,
            bsoncxx::document::view_or_value
        >> filter_updates;

        for (const auto& [key, vec] : are_emb) {
            auto filter = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("_id", key)
            );

            bsoncxx::builder::basic::array array_builder;
            for (const auto& v : vec) {
                array_builder.append(static_cast<double>(v));  // float -> double
            }
            auto update = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("$set", 
                    bsoncxx::builder::basic::make_document(
                        bsoncxx::builder::basic::kvp("KCGE_Emb", array_builder)
                    )
                )
            );

            filter_updates.emplace_back(std::move(filter), std::move(update));
        }

        // 2. 调用批量更新
        return bulkUpdateMany("areas", filter_updates, true);
    } catch (const bsoncxx::exception& e) {  // 正确：bsoncxx 确实有 exception 类
        std::cerr << "are update BSON Error: " << e.what() << std::endl;
        return -2;
    } catch (const mongocxx::exception& e) {  // 特定操作异常
        std::cerr << "are update MongoDB Operation Failed: " << e.what() << std::endl;
        return -3;
    }catch (const std::exception& e) {  // 兜底
        std::cerr << "System Error: " << e.what() << std::endl;
        return -1;
    }
}

int MongoDBOperator::update_scn_kcge_emb(const std::unordered_map<std::string, std::vector<float>> &scn_emb) {
    if (!isConnected()) {
        throw std::runtime_error("MongoDB connection is not initialized");
    }

    try {
        std::vector<std::pair<
            bsoncxx::document::view_or_value,
            bsoncxx::document::view_or_value
        >> filter_updates;

        for (const auto& [key, vec] : scn_emb) {
            auto filter = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("_id", key)
            );

            bsoncxx::builder::basic::array array_builder;
            for (const auto& v : vec) {
                array_builder.append(static_cast<double>(v));  // float -> double
            }
            auto update = bsoncxx::builder::basic::make_document(
                bsoncxx::builder::basic::kvp("$set", 
                    bsoncxx::builder::basic::make_document(
                        bsoncxx::builder::basic::kvp("KCGE_Emb", array_builder)
                    )
                )
            );

            filter_updates.emplace_back(std::move(filter), std::move(update));
        }

        // 2. 调用批量更新
        return bulkUpdateMany("scenes", filter_updates, true);
    } catch (const bsoncxx::exception& e) {  // 正确：bsoncxx 确实有 exception 类
        std::cerr << "scn update BSON Error: " << e.what() << std::endl;
        return -2;
    } catch (const mongocxx::exception& e) {  // 特定操作异常
        std::cerr << "scn update MongoDB Operation Failed: " << e.what() << std::endl;
        return -3;
    }catch (const std::exception& e) {  // 兜底
        std::cerr << "System Error: " << e.what() << std::endl;
        return -1;
    }
}





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