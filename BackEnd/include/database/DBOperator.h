#ifndef DBOPERATOR_H
#define DBOPERATOR_H


#include <jdbc/cppconn/prepared_statement.h>  // 必须包含
#include <jdbc/cppconn/resultset.h>
#include <jdbc/cppconn/statement.h>
#include <jdbc/mysql_driver.h>
#include <jdbc/mysql_connection.h>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

class DBOperator {
private:
    // 私有构造/析构
    DBOperator();
    ~DBOperator();

    // 事务支持
    bool beginTransaction();
    bool commit();
    bool rollback();

    // 私有通用执行方法
    std::vector<std::unordered_map<std::string, std::string>> 
    executeQuery(const std::string& query);

    int executeUpdate(const std::string& sql);

    // PIMPL模式隐藏实现细节
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    static std::mutex instanceMutex_;

public:
    DBOperator(const DBOperator&) = delete;
    DBOperator& operator=(const DBOperator&) = delete;
    // 获取单例实例
    static DBOperator& getInstance();
    // 初始化数据库连接
    bool initialize();
    // 检查连接状态
    bool isConnected() const;
    // 关闭连接
    void close();

    void testSelect(std::string table, int limit);
};

#endif