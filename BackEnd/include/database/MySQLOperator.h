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

    bool judge_uid_exist(std::string &table, std::string &pre, std::string &uid);

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
    std::vector<std::vector<std::string>> get_interacts_in_area_of_lrn_with_time(
        const std::string &are_uid, 
        const std::string &lrn_uid, 
        const std::string &time_start, 
        const std::string &time_end
    );

    // 获取指定scn所涉及的cpt
    std::unordered_map<std::string, std::unordered_set<std::string>> get_cpt_of_scn(
        const std::unordered_set<std::string> &scn_uids
    );

    // 获取指定领域的所有cpt
    std::unordered_map<std::string, int> get_cpt_uid_id_of_area(const std::string &are_uid);

    // 获取指定领域的cpt数量
    int get_cpt_num_of_area(const std::string &are_uid);

    // 获取指定领域的所有特殊scn和其对应的cpt
    std::unordered_map<std::string, std::string> get_special_scn_cpt_uid_of_are(const std::string &are_uid);

    // 获取指定lrn的近一个月内的交互记录
    std::vector<std::vector<std::string>> get_lrn_interacts_time(
        const std::string &lrn_uid, 
        const std::string &time_start, 
        const std::string &time_end
    );

    // 判断learners中是否有重复uid
    bool judge_lrn_uid_exist(std::string &uid);

    // 判断scenes中是否有重复uid
    bool judge_scn_uid_exist(std::string &uid);

    // 判断concepts中是否有重复uid
    bool judge_cpt_uid_exist(std::string &uid);

    // 判断areas中是否有重复uid
    bool judge_are_uid_exist(std::string &uid);

    // 向scenes中插入新的scn
    int insert_one_scn_to_scenes(std::string &scn_uid, bool has_result);

    // 从scenes中删除scn
    int delete_one_scn_from_scenes(std::string &scn_uid);

    // 从interacts中删除scn
    int delete_one_scn_from_interacts(std::string &scn_uid);

    // 从graph_interact中删除scn
    int delete_one_scn_from_graph_interact(std::string &scn_uid);

    // 向graph_involve中添加数据
    int insert_one_scn_to_graph_involve(
        std::string &scn_uid, 
        std::unordered_map<std::string, float> &cpt_uid2diff
    );

    // 根据scn_uid从graph_involve中删除数据
    int delete_one_scn_from_graph_involve(std::string &scn_uid);

    // 向concepts中插入新的cpt
    int insert_one_cpt_to_concepts(std::string &are_uid, std::string &cpt_uid, std::string &name);

    // 向graph_precondition中插入多条新的记录
    int insert_multi_cpt_cpt_to_graph_precondition(
        std::vector<std::pair<std::string, std::string>> cpt_cpt
    );

    // 向graph_belong中插入新的记录
    int insert_one_are_cpt_to_graph_belong(std::string &are_uid, std::string &cpt_uid);

    // 从concept中删除cpt
    int delete_one_cpt_from_concepts(std::string &cpt_uid);

    // 从graph_belong中根据cpt删除记录
    int delete_one_cpt_from_graph_belong(std::string &cpt_uid);

    // 根据cpt_uid从graph_involve中删除数据
    int delete_one_cpt_from_graph_involve(std::string &cpt_uid);

    // 从graph_precondition中删除数据
    int delete_one_cpt_from_graph_precondition(std::string &cpt_uid);

    // 从graph_precondition中获取pre和aft均在参数内的记录
    std::vector<std::vector<std::string>> get_cpt_cpt_from_graph_precondition_with_both_in(
        std::unordered_set<std::string> &cpt_uids
    );

    // 从graph_belong中根据cpt_uid获取对应的are_uid
    std::string get_are_uid_by_cpt_uid(std::string &cpt_uid);

    // 从graph_belong中根据cpt_uid获取对应的are_uid
    std::unordered_map<std::string, std::string> get_are_uid_by_multi_cpt_uid(
        std::unordered_set<std::string> &cpt_uids
    );

    // 获取指定cpt相关的scn_uid
    std::unordered_set<std::string> get_scn_uid_from_graph_involve_by_cpt_uid(std::string &cpt_uid);

    // 获取指定cpt相关的前后置条件的cpt
    std::unordered_set<std::string> get_cpt_uid_from_graph_precondition_by_cpt_uid(std::string &cpt_uid);

    // 根据指定的scn和cpt获取involve关系
    std::unordered_map<std::string, std::unordered_map<std::string, float>> get_scn_cpt_from_graph_involve_by_scns_cpts(
        std::unordered_set<std::string> &scn_uids,
        std::unordered_set<std::string> &cpt_uids
    );

    // 向graph_involve中插入一条数据
    int insert_one_scn_cpt_to_graph_involve(std::string &scn_uid, std::string &cpt_uid, float difficulty);
};

#endif