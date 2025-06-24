import sys
from pathlib import Path
deeplearning_root = str(Path(__file__).parent.parent)
if deeplearning_root not in sys.path:
    sys.path.insert(0, deeplearning_root)

import mysql
import mysql.connector

class MySQLDB():

    def __init__(self):
        self.con = mysql.connector.connect(
            host="localhost",  # MySQL服务器地址
            user="root",   # 用户名
            password="123456",  # 密码
            database="MLS_db"  # 数据库名称
        )
    
    # 获取从time_start开始的所有交互数据
    def get_all_interacts(self):
        sql = f"""
        select lrn_uid, unt_uid, result
        from interacts 
        order by created_at desc
        """
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    # 获取从time_start开始的所有交互数据
    def get_interacts_from(self, time_start, limit = -1):
        sql = f"""
        select lrn_uid, unt_uid, result from interacts 
        where created_at >= %s 
        order by created_at desc
        """
        if limit > 0:
            sql += f" limit {limit}"
        cursor = self.con.cursor()
        cursor.execute(sql, [time_start])
        result = cursor.fetchall
        cursor.close()
        return result

    # 获取are_uid下的知识点相关的所有从time_start开始的交互数据
    def get_interacts_of_are(self, are_uid, time_start, limit = -1):
        sql = f"""
        WITH relevant_units AS (
            SELECT DISTINCT gi.unt_uid
            FROM graph_belong gb
            JOIN graph_involve gi ON gb.cpt_uid = gi.cpt_uid
            WHERE gb.are_uid = %s
        ),
        qualified_learners AS (
            SELECT i.lrn_uid
            FROM interacts i
            JOIN relevant_units rs ON i.unt_uid = rs.unt_uid
            WHERE i.created_at > %s
            GROUP BY i.lrn_uid
            HAVING COUNT(*) >= 4
        )
        SELECT i.lrn_uid, i.unt_uid, i.result
        FROM interacts i
        JOIN relevant_units rs ON i.unt_uid = rs.unt_uid
        JOIN qualified_learners ql ON i.lrn_uid = ql.lrn_uid
        WHERE i.created_at > %s
        ORDER BY i.created_at;
        """
        if limit > 0:
            sql += f" limit {limit}"
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid, time_start, time_start])
        result = cursor.fetchall()
        cursor.close()
        return result

    # 获取are_uid下的知识点相关的所有从time_start开始的有result的交互数据
    def get_interacts_with_cpt_in_are_with_result(self, are_uid, time_start, limit = -1):
        sql = f"""
        WITH cpt_in_are AS (
            SELECT cpt_uid 
            FROM graph_belong 
            WHERE are_uid = %s
        ),
        unt_has_result AS (
            SELECT gi.unt_uid
            FROM graph_involve gi
            LEFT JOIN cpt_in_are cia ON gi.cpt_uid = cia.cpt_uid
            GROUP BY gi.unt_uid
            HAVING COUNT(*) = COUNT(cia.cpt_uid)
        )
        SELECT i.lrn_uid, i.unt_uid, i.result
        FROM interacts i
        JOIN units s ON i.unt_uid = s.unt_uid AND s.has_result = 1
        JOIN unt_has_result shr ON i.unt_uid = shr.unt_uid
        WHERE i.created_at >= %s and i.lrn_uid IN (
            SELECT i2.lrn_uid
            FROM interacts i2
            JOIN units s2 ON i2.unt_uid = s2.unt_uid AND s2.has_result = 1
            JOIN unt_has_result shr2 ON i2.unt_uid = shr2.unt_uid
            WHERE i2.created_at >= %s
            GROUP BY i2.lrn_uid
            HAVING COUNT(*) >= 4
        )
        ORDER BY i.created_at ASC;
        """
        if limit > 0:
            sql += f" limit {limit}"
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid, time_start, time_start])
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取are_uid下的所有知识点uid
    def get_all_concepts_of_area(self, are_uid):
        sql = f"""
        select cpt.cpt_uid
        from concepts cpt
        join graph_belong bg
        on cpt.cpt_uid = bg.cpt_uid 
        where bg.are_uid = %s
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
    # 获取are_uid下的所有知识点uid和id_in_area
    def get_all_concepts_uid_and_id_of_area(self, are_uid):
        sql = f"""
        select cpt.cpt_uid, cpt.id_in_area
        from concepts cpt
        join graph_belong bg
        on cpt.cpt_uid = bg.cpt_uid 
        where bg.are_uid = %s
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = {}
        for item in cursor.fetchall():
            result[item[0]] = item[1]
        cursor.close()
        return result
    
    # 获取unt_uids中所有场景所涉及的知识点 - unt_uid cpt_uid
    def get_concepts_of_units(self, unt_uids):
        sql = f"""
        select unt_uid, cpt_uid
        from graph_involve
        where unt_uid in (%s)
        """
        place_holders = ','.join(['%s'] * len(unt_uids))
        cursor = self.con.cursor()
        cursor.execute(sql % place_holders, unt_uids)
        result = {}
        for unt_uid, cpt_uid in cursor.fetchall():
            if unt_uid not in result:
                result[unt_uid] = set()
            result[unt_uid].add(cpt_uid)
        cursor.close()
        return result
    
    # 获取are_uid下的知识点数量
    def get_concept_num_of_area(self, are_uid):
        sql = f"""
        select count(*)
        from graph_belong
        where are_uid = %s
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = cursor.fetchone()
        cursor.close()
        return result
    
    # 获取场景涉及的知识点的内部id
    def get_concepts_uid_of_units(self, unt_uids):
        sql = f"""
        select gi.unt_uid, cpt.cpt_uid
        from graph_involve gi
        join concepts cpt
        on gi.cpt_uid = cpt.cpt_uid
        where gi.unt_uid in (%s)
        """
        place_holders = ','.join(['%s'] * len(unt_uids))
        cursor = self.con.cursor()
        cursor.execute(sql % place_holders, unt_uids)
        result = {}
        for unt_uid, cpt_uid in cursor.fetchall():
            if unt_uid not in result:
                result[unt_uid] = []
            result[unt_uid].append(cpt_uid)
        cursor.close()
        return result
    
    # 获取学习者数量
    def get_learner_num(self):
        sql = '''
        select count(*)
        from learners
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()[0]
        cursor.close()
        return result
    
    # 获取场景数量
    def get_unit_num(self):
        sql = '''
        select count(*)
        from units
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()[0]
        cursor.close()
        return result
    
    # 获取知识点数量
    def get_concept_num(self):
        sql = '''
        select count(*)
        from concepts
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()[0]
        cursor.close()
        return result

    # 获取所有学习者的uid
    def get_learners_uid(self):
        sql = '''
        select lrn_uid
        from learners
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
    # 获取所有场景的uid
    def get_units_uid(self):
        sql = '''
        select unt_uid
        from units
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
    # 获取所有知识点的uid
    def get_concepts_uid(self):
        sql = '''
        select cpt_uid
        from concepts
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
    # 从graph_interact中获取所有的交互记录以及交互总次数
    def get_lrn_unt_num(self):
        sql = '''
        select lrn_uid, unt_uid, all_times
        from graph_interact
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 从graph_involve中获取所有场景和知识点的难度信息
    def get_unt_cpt_dif(self):
        sql = '''
        select unt_uid, cpt_uid, difficulty
        from graph_involve
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取所有cpt_uid和name
    def get_cpt_uid_name(self):
        sql = '''
        select cpt_uid, cpt_name
        from concepts
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取area数量
    def get_area_num(self):
        sql = '''
        select count(*)
        from areas
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()[0]
        cursor.close()
        return result
    
    # 从graph_precondition中获取所有前置关系
    def get_cpt_cpt(self):
        sql = '''
        select cpt_uid_pre, cpt_uid_aft
        from graph_precondition
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 从graph_belong中获取所有属于关系
    def get_cpt_are(self):
        sql = '''
        select cpt_uid, are_uid
        from graph_belong
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取所有area信息
    def get_areas_uid(self):
        sql = '''
        select are_uid
        from areas
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
    # 获取所有至少交互过两个场景的学习者
    def get_learners_uid_with_unt_greater_4(self):
        sql = '''
        select lrn_uid
        from interacts
        group by lrn_uid
        having count(*) >= 4
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
     # 获取所有至少交互过两个场景的学习者的交互图信息
    def get_lrn_unt_num_with_unt_greater_4(self):
        sql = '''
        select lrn_uid, unt_uid, all_times
        from graph_interact
        where lrn_uid in (
            select lrn_uid
            from interacts
            group by lrn_uid
            having count(*) >= 4
        )
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取所有至少交互过4个场景的学习者的所有交互信息
    def get_interacts_with_unt_greater_4(self):
        sql = f"""
        select ict1.lrn_uid, ict1.unt_uid, ict1.result
        from interacts ict1
        where ict1.lrn_uid in (
            select ict2.lrn_uid
            from interacts ict2
            group by ict2.lrn_uid
            having count(*) >= 4
        )
        order by ict1.created_at asc
        """
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取指定are下的special_units中的所有unt_uid和cpt_uid
    def get_special_unt_cpt_uid_of_are(self, are_uid):
        sql = f"""
        SELECT ss.unt_uid, ss.cpt_uid
        FROM special_units ss
        JOIN graph_belong gb ON ss.cpt_uid = gb.cpt_uid
        WHERE gb.are_uid = %s;
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # KT中将参与训练的cpt均置为trained，方便之后使用时辨别哪些可用KT预测
    def make_cpt_trained(self, cpt_uids):
        sql = f"""
        update concepts
        set trained = 1
        where cpt_uid in (%s)
        """
        place_holders = ','.join(['%s'] * len(cpt_uids))
        cursor = self.con.cursor()
        cursor.execute(sql % place_holders, cpt_uids)
        self.con.commit()
        cursor.close()

    # 从graph_belong中获取are_uid所有cpt_uid
    def get_cpt_uid_of_are(self, are_uid):
        sql = f"""
        select cpt_uid
        from graph_belong
        where are_uid = %s
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
        cursor.close()
        return result
    
    # 从graph_involve中获取are相关的所有记录
    def get_unt_cpt_uid_of_are(self, are_uid):
        sql = f"""
        SELECT gi.unt_uid, gi.cpt_uid, gi.difficulty
        FROM graph_involve AS gi
        JOIN graph_belong AS gb ON gb.cpt_uid = gi.cpt_uid
        WHERE gb.are_uid = %s
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 从graph_precondition中获取are相关的所有记录
    def get_cpt_cpt_of_are(self, are_uid):
        sql = f"""
        with cpt_in_are as (
            select cpt_uid
            from graph_belong
            where are_uid = %s 
        )
        select gp.cpt_uid_pre, gp.cpt_uid_aft
        from graph_precondition as gp
        join cpt_in_are as cia1 on gp.cpt_uid_pre = cia1.cpt_uid
        join cpt_in_are as cia2 on gp.cpt_uid_aft = cia2.cpt_uid
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取are下的所有有result的unt的uid
    def get_unt_of_are_with_result(self, are_uid):
        sql = f"""
        SELECT DISTINCT s.unt_uid
        FROM graph_belong gb
        JOIN graph_involve gi ON gb.cpt_uid = gi.cpt_uid
        JOIN units s ON gi.unt_uid = s.unt_uid
        WHERE gb.are_uid = %s AND s.has_result = 1;
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = [item[0] for item in cursor.fetchall()]
        result = set(result)
        cursor.close()
        return result

mysqldb = MySQLDB()