#ifndef MONGODBOPERATOR_H
#define MONGODBOPERATOR_H

#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/pool.hpp>
#include <mongocxx/uri.hpp>
#include <mongocxx/cursor.hpp>

#include <bsoncxx/document/element.hpp>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/document/view.hpp>
#include <bsoncxx/json.hpp>
#include <bsoncxx/document/view_or_value.hpp>
#include <bsoncxx/builder/basic/document.hpp>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <optional>

class MongoDBOperator {
private:
    // 私有构造/析构
    MongoDBOperator();
    ~MongoDBOperator();

    // PIMPL模式隐藏实现细节
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    static std::mutex instanceMutex_;

public:
    MongoDBOperator(const MongoDBOperator&) = delete;
    MongoDBOperator& operator=(const MongoDBOperator&) = delete;
    
    // 获取单例实例
    static MongoDBOperator& getInstance();
    
    // 初始化数据库连接
    bool initialize();
    
    // 检查连接状态
    bool isConnected() const;
    
    // 关闭连接池
    void close();

    // ========== 通用操作方法 ==========
    
    // 查询文档（返回可选值）
    std::optional<bsoncxx::document::value> findOne(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter,
        bsoncxx::document::view_or_value projection = {});
    
    // 查询多个文档
    std::optional<mongocxx::cursor> findMany(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter,
        bsoncxx::document::view_or_value projection = {},
        std::optional<int64_t> limit = std::nullopt);
    
    // 插入单个文档
    std::optional<bsoncxx::document::value> insertOne(
        const std::string& collection, 
        bsoncxx::document::view_or_value document);

    // 插入多个文档
    std::optional<std::vector<bsoncxx::document::value>> insertMany(
        const std::string& collection,
        const std::vector<bsoncxx::document::view_or_value>& documents);
    
    // 更新文档
    bool updateOne(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter,
        bsoncxx::document::view_or_value update,
        bool upsert = false);
    
    // 更新多个文档
    int updateMany(
        const std::string& collection,
        bsoncxx::document::view_or_value filter,
        bsoncxx::document::view_or_value update,
        bool upsert);
    
    // 删除文档
    bool deleteOne(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter);

    // 删除多个文档
    int deleteMany(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter);

    // ========== 业务方法示例 ==========

    // 获取指定scn的kcge嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_scn_kcge_by_scn_uid(const std::unordered_set<std::string> &scn_uids);
    
    // 获取指定cpt的kcge嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_cpt_kcge_by_cpt_uid(const std::unordered_set<std::string> &cpt_uids);

    // 获取指定lrn的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_lrn_hgc_by_lrn_uid(const std::unordered_set<std::string> &lrn_uids);

    // 获取指定scn的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_scn_hgc_by_scn_uid(const std::unordered_set<std::string> &scn_uids);

    // 获取指定cpt的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_cpt_hgc_by_cpt_uid(const std::unordered_set<std::string> &cpt_uids);

    // 获取所有cpt的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_all_cpt_hgc();

    // 从scenes中删除指定scn_uid的文档
    int delete_scn_from_scenes(const std::vector<std::string> &scn_uids);

    // 从concepts中删除指定cpt_uid的文档
    int delete_cpt_from_concepts(const std::vector<std::string> &cpt_uids);


    
    // 示例1: 获取用户信息（返回可选文档）
    std::optional<std::unordered_map<std::string, float>> testGetLearnerInfo(const std::string& lrn_uid);
    
    // 示例2: 记录用户活动（返回插入的文档ID）
    std::optional<std::string> logUserActivity(
        const std::string& userId, 
        const std::string& activityType,
        const std::string& details);
    
    // 示例3: 获取用户收藏项
    std::vector<std::string> getUserFavorites(const std::string& userId);
};

#endif