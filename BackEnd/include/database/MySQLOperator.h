#ifndef DBOPERATOR_H
#define DBOPERATOR_H

#include <jdbc/cppconn/prepared_statement.h>
#include <jdbc/cppconn/resultset.h>
#include <jdbc/cppconn/statement.h>
#include <jdbc/mysql_driver.h>
#include <jdbc/mysql_connection.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <sstream>

class MySQLOperator {
private:
    // 私有构造/析构
    MySQLOperator();
    ~MySQLOperator();

    // 事务支持
    bool beginTransaction();
    bool commit();
    bool rollback();

    // 私有通用执行方法
    std::vector<std::vector<std::string>> executeQuery(const std::string& query);

    int executeUpdate(const std::string& sql);

    // PIMPL模式隐藏实现细节
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    static std::mutex instanceMutex_;

public:
    MySQLOperator(const MySQLOperator&) = delete;
    MySQLOperator& operator=(const MySQLOperator&) = delete;
    // 获取单例实例
    static MySQLOperator& getInstance();
    // 初始化数据库连接
    bool initialize();
    // 检查连接状态
    bool isConnected() const;
    // 关闭连接
    void close();

    void testSelect(std::string table, int limit);

    // 获取指定学习者在指定领域下的时间区间内的交互记录
    std::vector<std::vector<std::string>> get_Are_lrn_Interacts_Time(const std::string &are_uid, const std::string &lrn_uid, const std::string &time_start, const std::string &time_end);

    // 获取指定scn所涉及的cpt
    std::unordered_map<std::string, std::unordered_set<std::string>> get_Cpt_of_Scn(const std::unordered_set<std::string> &scn_uids);

    // 获取指定领域的所有cpt
    std::unordered_map<std::string, int> get_cpt_uid_id_of_area(const std::string &are_uid);

    // 获取指定领域的所有特殊scn和其对应的cpt
    std::unordered_map<std::string, std::string> get_special_scn_cpt_uid_of_are(const std::string &are_uid);

    // 获取指定lrn的近一个月内的交互记录
    std::vector<std::vector<std::string>> get_lrn_interacts_time(const std::string &lrn_uid, const std::string &time_start, const std::string &time_end);
};

#endif