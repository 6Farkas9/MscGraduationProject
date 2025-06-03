#ifndef MONGO_H
#define MONGO_H
 
#include <QObject>
#include <QDebug>
#include <bsoncxx/builder/stream/document.hpp>
#include <bsoncxx/types.hpp>
#include <mongocxx/client.hpp>
#include <mongocxx/instance.hpp>
#include <mongocxx/uri.hpp>
#include <bsoncxx/json.hpp>
 
 
// #pragma execution_character_set("utf-8");
 
class Mongo : public QObject
{
    Q_OBJECT
public:
    explicit Mongo(QObject *parent = nullptr);
    ~Mongo();
 
    static void connectToMongoDB();
    bool connectToHost(QString m_hostName, QString m_port);
 
private:
    mongocxx::instance* m_dbInstance = nullptr;
    mongocxx::client* m_client = nullptr;
 
signals:
};
 
#endif // MONGO_H