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
#include <mongocxx/options/bulk_write.hpp>
#include <mongocxx/exception/exception.hpp>
#include <bsoncxx/exception/exception.hpp>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <optional>
#include <utility>

class MongoDBOperator {
private:
    // 私有构造/析构
    MongoDBOperator();
    ~MongoDBOperator();

    // PIMPL模式隐藏实现细节
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    static std::mutex instanceMutex_;

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

    int bulkUpdateMany(
        const std::string& collection,
        const std::vector<std::pair<
            bsoncxx::document::view_or_value, // filter
            bsoncxx::document::view_or_value   // update
        >>& filter_updates,
        bool upsert
    );
    
    // 删除文档
    bool deleteOne(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter);

    // 删除多个文档
    int deleteMany(
        const std::string& collection, 
        bsoncxx::document::view_or_value filter);

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

    // ========== 业务方法示例 ==========

    // 获取指定unt的kcge嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_are_kcge_by_are_uid(const std::unordered_set<std::string> &are_uids);

    // 获取指定unt的kcge嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_unt_kcge_by_unt_uid(const std::unordered_set<std::string> &unt_uids);
    
    // 获取指定cpt的kcge嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_cpt_kcge_by_cpt_uid(const std::unordered_set<std::string> &cpt_uids);

    // 获取指定lrn的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_lrn_hgc_by_lrn_uid(const std::unordered_set<std::string> &lrn_uids);

    // 获取指定unt的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_unt_hgc_by_unt_uid(const std::unordered_set<std::string> &unt_uids);

    // 获取指定cpt的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_cpt_hgc_by_cpt_uid(const std::unordered_set<std::string> &cpt_uids);

    // 获取所有cpt的HGC嵌入表达
    std::unordered_map<std::string, std::vector<float>> get_all_cpt_hgc();

    // 从units中删除指定unt_uid的文档
    int delete_unt_from_units(const std::vector<std::string> &unt_uids);

    // 从concepts中删除指定cpt_uid的文档
    int delete_cpt_from_concepts(const std::vector<std::string> &cpt_uids);

    // 更新concept的kcge嵌入
    int update_cpt_kcge_emb(const std::unordered_map<std::string, std::vector<float>> &cpt_emb);

    // 更新area的kcge嵌入
    int update_are_kcge_emb(const std::unordered_map<std::string, std::vector<float>> &are_emb);

    // 更新unit的kcge嵌入
    int update_unt_kcge_emb(const std::unordered_map<std::string, std::vector<float>> &unt_emb);

    // 获取单一学习者在指定cpt上的预测成绩
    std::unordered_map<std::string, std::unordered_map<std::string, float>> get_lrn_kt_cd_by_cpt_uids(
        std::string &lrn_uid,
        std::unordered_set<std::string> &cpt_uids
    );

    // 根据学习者uid和知识点查找学习榜样
    std::optional<std::vector<std::string>> get_lrn_models_by_lrn_cpt_uid(
        const std::string& lrn_uid,
        const std::unordered_set<std::string>& cpt_uids,
        size_t max_results /* = 5 */
    );

    // 根据学习者uid和知识点查找学习伙伴
    std::optional<std::vector<std::string>> get_lrn_partners_by_lrn_cpt_uid(
        const std::string& lrn_uid,
        const std::unordered_set<std::string>& cpt_uids,
        double similarity_threshold /* = 0.1 */,
        size_t max_results /* = 5 */
    );

    
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