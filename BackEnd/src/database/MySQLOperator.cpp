#include "MySQLOperator.h"

#include <iostream>

// 初始化静态成员
std::mutex MySQLOperator::instanceMutex_;

// PIMPL实现结构体
struct MySQLOperator::Impl {
    sql::mysql::MySQL_Driver* driver;
    std::unique_ptr<sql::Connection> conn;
    std::mutex dbMutex;
    bool isTransactionActive;

    Impl() : driver(nullptr), isTransactionActive(false) {}
};

// 单例实例获取
MySQLOperator& MySQLOperator::getInstance() {
    std::lock_guard<std::mutex> lock(instanceMutex_);
    static MySQLOperator instance;
    return instance;
}

// 构造函数
MySQLOperator::MySQLOperator() : pImpl_(std::make_unique<Impl>()) {
    pImpl_->driver = sql::mysql::get_mysql_driver_instance();
}

// 析构函数
MySQLOperator::~MySQLOperator() {
    close();
}

// 初始化数据库连接
bool MySQLOperator::initialize() {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    try {
        // 构建连接字符串
        std::string connectionStr = "tcp://127.0.0.1:3306?authMethod=mysql_native_password&characterEncoding=utf8mb4";
        
        // 建立连接
        try{
            auto x = pImpl_->driver->connect(connectionStr, "root", "123456");
            pImpl_->conn.reset(x);
        }
        catch(const std::exception& e){
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
        
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
bool MySQLOperator::beginTransaction() {
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

bool MySQLOperator::commit() {
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

bool MySQLOperator::rollback() {
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
bool MySQLOperator::isConnected() const {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    return pImpl_->conn && !pImpl_->conn->isClosed();
}

// 关闭连接
void MySQLOperator::close() {
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

std::vector<std::vector<std::string>> MySQLOperator::executeQuery(const std::string& query) {
    std::lock_guard<std::mutex> lock(pImpl_->dbMutex);
    std::vector<std::vector<std::string>> results;
    try {
        if (!pImpl_->conn || pImpl_->conn->isClosed()) {
            throw sql::SQLException("Connection is not initialized or closed");
        }
        std::unique_ptr<sql::PreparedStatement> stmt(pImpl_->conn->prepareStatement(query));
        
        std::unique_ptr<sql::ResultSet> res(stmt->executeQuery());
        sql::ResultSetMetaData* meta = res->getMetaData();
        unsigned int columns = meta->getColumnCount();
        while (res->next()) {
            std::vector<std::string> row;
            for (unsigned int i = 1; i <= columns; ++i) {
                // row[meta->getColumnLabel(i)] = res->getString(i);
                row.emplace_back(res->getString(i));
                // std::cout << res->getString(i) << std::endl;
            }
            results.emplace_back(std::move(row));
        }
    } catch (const sql::SQLException& e) {
        std::cerr << "MySQL Query Error: " << e.what() 
                  << " (MySQL error code: " << e.getErrorCode() 
                  << ", SQLState: " << e.getSQLState() << ")" << std::endl;
        throw;
    }
    
    return results;
}

int MySQLOperator::executeUpdate(const std::string& sql) {
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

void MySQLOperator::testSelect(std::string table, int limit) {
    std::string sql = "select * from " + table + " limit " + std::to_string(limit);
    std::cout << sql << std::endl;
    auto res = executeQuery(sql);
    for(auto row : res){
        for(auto item : row){
            std::cout << item << "\t";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<std::string>> MySQLOperator::get_Are_lrn_Interacts_Time(const std::string &are_uid, const std::string &lrn_uid, const std::string &time_start, const std::string &time_end){
    std::string sql = R"(
        SELECT 
            i.scn_uid, 
            i.result
        FROM 
            interacts i
        INNER JOIN 
            graph_involve gi ON i.scn_uid = gi.scn_uid
        INNER JOIN 
            graph_belong gb ON gi.cpt_uid = gb.cpt_uid
        WHERE 
            i.lrn_uid = ")";
    sql += lrn_uid + R"(" 
        AND gb.are_uid = ")";
    sql += are_uid + R"(" 
        AND i.created_at >= ")";
    sql += time_start + R"(" 
        AND i.created_at <= ")";
    sql += time_end + R"(" 
        ORDER BY 
        i.created_at ASC;)";

    // std::cout << sql << std::endl;

    auto result = executeQuery(sql);
    
    std::vector<std::vector<std::string>> ans;

    for(auto &row : result){
        ans.emplace_back(std::vector<std::string>());
        for(auto &item : row){
            ans.back().emplace_back(item);
            // std::cout << item << " ";
        }
        // std::cout << std::endl;
    }

    return ans;
}

std::unordered_map<std::string, int> MySQLOperator::get_cpt_uid_id_of_area(const std::string &are_uid){
    std::string sql = 
        R"(select cpt.cpt_uid, cpt.id_in_area
        from concepts cpt
        join graph_belong bg
        on cpt.cpt_uid = bg.cpt_uid 
        where bg.are_uid = ")";
    sql += are_uid + R"(";)";

    // std::cout << sql << std::endl;
    auto result = executeQuery(sql);
    // std::cout << cpt_num << std::endl;

    std::unordered_map<std::string, int> ans;
    for(auto &row : result){
        int id_in_area;
        std::istringstream(row[1]) >> id_in_area;
        // std::cout << id_in_area << std::endl;
        ans[row[0]] = id_in_area;
    }
    return ans;
}

std::unordered_map<std::string, std::unordered_set<std::string>> MySQLOperator::get_Cpt_of_Scn(const std::unordered_set<std::string> &scn_uids){
    std::string sql = R"(
        select scn_uid, cpt_uid
        from graph_involve
        where scn_uid in ()";

    int i = 0;
    int length = scn_uids.size() - 1;
    for (auto &scn_uid : scn_uids){
        sql += R"(")" + scn_uid + R"(")";
        if(i++ < length) 
            sql += R"(,)";
    }
    sql += R"())";
    // std::cout << sql << std::endl;
    auto result = executeQuery(sql);
    std::unordered_map<std::string, std::unordered_set<std::string>> ans;
    for (auto & row : result){
        if(ans.find(row[0]) == ans.end()){
            ans[row[0]] = std::unordered_set<std::string>();
        }
        ans[row[0]].insert(row[1]);
    }
    return ans;
}

std::unordered_map<std::string, std::string> MySQLOperator::get_special_scn_cpt_uid_of_are(const std::string &are_uid) {
    std::string sql = R"(
        SELECT ss.scn_uid, ss.cpt_uid
        FROM special_scenes ss
        JOIN graph_belong gb ON ss.cpt_uid = gb.cpt_uid
        WHERE gb.are_uid = ")";
    sql = sql + are_uid + R"(";)";
    auto result = executeQuery(sql);
    std::unordered_map<std::string, std::string> ans;
    for(auto &row : result){
        ans[row[0]] = row[1];
    }
    return ans;
}

std::vector<std::vector<std::string>> MySQLOperator::get_lrn_interacts_time(const std::string &lrn_uid, const std::string &time_start, const std::string &time_end) {
        std::string sql = R"(
        SELECT 
            scn_uid, 
            result
        FROM 
            interacts
        WHERE 
            lrn_uid = ")";
    sql += lrn_uid  + R"(" 
        AND created_at >= ")";
    sql += time_start + R"(" 
        AND created_at <= ")";
    sql += time_end + R"(" 
        ORDER BY 
        created_at ASC;)";

    auto result = executeQuery(sql);
    
    std::vector<std::vector<std::string>> ans;

    for(auto &row : result){
        ans.emplace_back(std::vector<std::string>());
        for(auto &item : row){
            ans.back().emplace_back(item);
            // std::cout << item << " ";
        }
        // std::cout << std::endl;
    }

    return ans;
}
