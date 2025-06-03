#include "MongoDBOperator.h"
 
Mongo::Mongo(QObject *parent)
    : QObject{parent}
{
    bool ret = connectToHost("localhost", "27017");
    if(!ret) {
        qDebug() << "mongodb连接失败！";
    }
}
 
Mongo::~Mongo()
{
    delete m_dbInstance;
    m_dbInstance = NULL;
    delete m_client;
    m_client = NULL;
}
 
 
void Mongo::connectToMongoDB(){
 
}
 
bool Mongo::connectToHost(QString m_hostName, QString m_port)
{
    if(m_hostName=="" || m_port=="")
        return false;
 
    mongocxx::uri uri{ QString("mongodb://%1:%2").arg(m_hostName).arg(m_port).toLatin1().data()};
    m_dbInstance = new(std::nothrow) mongocxx::instance();
    m_client = new(std::nothrow) mongocxx::client(uri);
    if (!m_client)
        return false;
    qDebug() << "连接成功!";
 
    return true;
}