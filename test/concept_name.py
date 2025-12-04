import mysql.connector
import os

def save_concepts_to_txt():
    try:
        # 连接到MySQL数据库
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="mls_sample"
        )
        
        cursor = conn.cursor()
        
        # 查询所有name字段
        query = "SELECT name FROM concepts"
        cursor.execute(query)
        
        # 获取所有结果
        results = cursor.fetchall()
        
        # 写入到txt文件
        output_file = "concepts_names.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for row in results:
                if row[0]:  # 确保name不为None或空
                    f.write(str(row[0]) + '\n')
        
        print(f"成功保存 {len(results)} 个name到 {output_file}")
        print(f"文件路径: {os.path.abspath(output_file)}")
        
    except mysql.connector.Error as err:
        print(f"数据库连接错误: {err}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 确保关闭连接
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    save_concepts_to_txt()