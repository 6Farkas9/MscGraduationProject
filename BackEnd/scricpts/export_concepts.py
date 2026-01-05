import mysql.connector

# 1. 连接数据库
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="mls_sample",
    charset="utf8mb4"
)

cursor = conn.cursor()

# 2. 只查询 name 字段
cursor.execute("SELECT name FROM Concepts WHERE name IS NOT NULL")

# 3. 取出所有 name
names = [row[0] for row in cursor.fetchall()]

# 4. 写入 concept.txt，用英文逗号连接
with open("concept.txt", "w", encoding="utf-8") as f:
    f.write(",".join(names))

# 5. 关闭连接
cursor.close()
conn.close()

print("导出完成，已生成 concept.txt")
