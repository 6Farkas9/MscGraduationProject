#include "DBOperator.h"

#include <iostream>

// 初始化静态成员
std::mutex DBOperator::instanceMutex_;

// PIMPL实现结构体
struct DBOperator::Impl {
    sql::mysql::MySQL_Driver* driver;
    std::unique_ptr<sql::Connection> conn;
    std::mutex dbMutex;
    bool isTransactionActive;

    Impl() : driver(nullptr), isTransactionActive(false) {}
};

// 单例实例获取
DBOperator& DBOperator::getInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    static DBOperator instance;
    return instance;
}

// 构造函数
DBOperator::DBOperator() : pImpl_(std::make_unique<Impl>()) {
    pImpl_->driver = sql::mysql::get_mysql_driver_instance();
}

// 析构函数
DBOperator::~DBOperator() {
    close();
}

// 初始化数据库连接
bool DBOperator::initialize() {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    try {
        // 构建连接字符串
        std::string connectionStr = "tcp://127.0.0.1:3306?authMethod=mysql_native_password&characterEncoding=utf8mb4";
        
        // 建立连接
        pImpl_->conn.reset(pImpl_->driver->connect(connectionStr, "root", "123456"));
        
        // 选择数据库
        pImpl_->conn->setSchema("MLS_db");
        
        // 设置字符集
        pImpl_->conn->setClientOption("charset", "utf8mb4");
        
        return true;
    } catch (const sql::SQLException& e) {
        std::cerr << "MySQL Connection Error: " << e.what() 
                  << " (MySQL error code: " << e.getErrorCode() 
                  << ", SQLState: " << e.getSQLState() << ")" << std::endl;
        return false;
    }
}

// 事务支持
bool DBOperator::beginTransaction() {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    try {
        if (pImpl_->conn && !pImpl_->conn->isClosed()) {
            pImpl_->conn->setAutoCommit(false);
            pImpl_->isTransactionActive = true;
            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

bool DBOperator::commit() {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    try {
        if (pImpl_->conn && !pImpl_->conn->isClosed() && pImpl_->isTransactionActive) {
            pImpl_->conn->commit();
            pImpl_->conn->setAutoCommit(true);
            pImpl_->isTransactionActive = false;
            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

bool DBOperator::rollback() {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    try {
        if (pImpl_->conn && !pImpl_->conn->isClosed() && pImpl_->isTransactionActive) {
            pImpl_->conn->rollback();
            pImpl_->conn->setAutoCommit(true);
            pImpl_->isTransactionActive = false;
            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

// 检查连接状态
bool DBOperator::isConnected() const {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    return pImpl_->conn && !pImpl_->conn->isClosed();
}

// 关闭连接
void DBOperator::close() {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    try {
        if (pImpl_->isTransactionActive) {
            pImpl_->conn->rollback();
        }
        if (pImpl_->conn && !pImpl_->conn->isClosed()) {
            pImpl_->conn->close();
        }
    } catch (...) {
        // 确保析构不抛出异常
    }
}

// ========== 私有通用执行方法 ==========

std::vector<std::unordered_map<std::string, std::string>> 
DBOperator::executeQuery(const std::string& query) {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    std::vector<std::unordered_map<std::string, std::string>> results;
    
    try {
        if (!pImpl_->conn || pImpl_->conn->isClosed()) {
            throw sql::SQLException("Connection is not initialized or closed");
        }

        std::unique_ptr<sql::PreparedStatement> stmt(pImpl_->conn->prepareStatement(query));
        
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery());
        sql::ResultSetMetaData* meta = res->getMetaData();
        unsigned int columns = meta->getColumnCount();
        
        while (res->next()) {
            std::unordered_map<std::string, std::string> row;
            for (unsigned int i = 1; i <= columns; ++i) {
                row[meta->getColumnLabel(i)] = res->getString(i);
            }
            results.push_back(std::move(row));
        }
    } catch (const sql::SQLException& e) {
        std::cerr << "MySQL Query Error: " << e.what() 
                  << " (MySQL error code: " << e.getErrorCode() 
                  << ", SQLState: " << e.getSQLState() << ")" << std::endl;
        throw;
    }
    
    return results;
}

int DBOperator::executeUpdate(const std::string& sql) {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    
    try {
        if (!pImpl_->conn || pImpl_->conn->isClosed()) {
            throw sql::SQLException("Connection is not initialized or closed");
        }

        std::unique_ptr<sql::PreparedStatement> stmt(pImpl_->conn->prepareStatement(sql));
        
        return stmt->executeUpdate();
    } catch (const sql::SQLException& e) {
        std::cerr << "MySQL Update Error: " << e.what() 
                  << " (MySQL error code: " << e.getErrorCode() 
                  << ", SQLState: " << e.getSQLState() << ")" << std::endl;
        throw;
    }
}

void DBOperator::testSelect(std::string table, int limit) {
    std::string sql = "select * from " + table + " limit " + std::to_string(limit);
    std::cout << sql << std::endl;
    auto res = executeQuery(sql);
    for(auto row : res){
        for(auto kv : row){
            std::cout << kv.second << "\t";
        }
        std::cout << std::endl;
    }
}