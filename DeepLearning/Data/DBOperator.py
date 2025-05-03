import mysql
import mysql.connector

class DB():

    def __init__(self):
        self.con = mysql.connector.connect(
            host="localhost",  # MySQL服务器地址
            user="root",   # 用户名
            password="123456",  # 密码
            database="MLS_db"  # 数据库名称
        )
    
    # 获取从time_start开始的所有交互数据
    def get_interacts_from(self, time_start, limit = -1):
        sql = f"""
        select lrn_uid, scn_uid, result from interacts 
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
    def get_interacts_with_cpt_in_are_from(self, are_uid, time_start, limit = -1):
        sql = f"""
        select itc.lrn_uid, itc.scn_uid, itc.result 
        from interacts itc 
        join (
            select gi.scn_uid 
            from graph_involve gi 
            left join (
                select gb.cpt_uid 
                from graph_belong gb 
                join areas a on gb.are_uid = a.are_uid
                where a.are_uid != %s
            ) non_are_name_cpts on gi.cpt_uid = non_are_name_cpts.cpt_uid 
            group by gi.scn_uid
            having count(non_are_name_cpts.cpt_uid) = 0 
            and count(gi.cpt_uid) > 0
        ) valid_scn on itc.scn_uid = valid_scn.scn_uid
        where itc.created_at >= %s
        order by itc.lrn_uid, itc.created_at asc 
        """
        if limit > 0:
            sql += f" limit {limit}"
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid, time_start])
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取are_uid下的所有知识点uid和id_in_area
    def get_all_concepts_of_area(self, are_uid):
        sql = f"""
        select cpt.cpt_uid, cpt.id_in_area
        from concepts cpt
        join graph_belong bg
        on cpt.cpt_uid = bg.cpt_uid 
        where bg.are_uid = %s
        """
        cursor = self.con.cursor()
        cursor.execute(sql, [are_uid])
        result = cursor.fetchall()
        cursor.close()
        return result
    
    # 获取scn_uids中所有场景所涉及的知识点 - scn_uid cpt_uid
    def get_concepts_of_scenes(self, scn_uids):
        sql = f"""
        select scn_uid, cpt_uid
        from graph_involve
        where scn_uid in (%s)
        """
        place_holders = ','.join(['%s'] * len(scn_uids))
        cursor = self.con.cursor()
        cursor.execute(sql % place_holders, scn_uids)
        result = {}
        for scn_uid, cpt_uid in cursor.fetchall():
            if scn_uid not in result:
                result[scn_uid] = []
            result[scn_uid].append(cpt_uid)
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
        return result
    
    # 获取场景涉及的知识点的内部id
    def get_concepts_id_in_area_of_scenes(self, scn_uids):
        sql = f"""
        select gi.scn_uid, cpt.id_in_area
        from graph_involve gi
        join concepts cpt
        on gi.cpt_uid = cpt.cpt_uid
        where gi.scn_uid in (%s)
        """
        place_holders = ','.join(['%s'] * len(scn_uids))
        cursor = self.con.cursor()
        cursor.execute(sql % place_holders, scn_uids)
        result = {}
        for scn_uid, cpt_uid in cursor.fetchall():
            if scn_uid not in result:
                result[scn_uid] = []
            result[scn_uid].append(cpt_uid)
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
        return result
    
    # 获取场景数量
    def get_scene_num(self):
        sql = '''
        select count(*)
        from scenes
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()[0]
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
        return result
    
    # 获取所有场景的uid
    def get_scenes_uid(self):
        sql = '''
        select scn_uid
        from scenes
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = []
        for item in cursor.fetchall():
            result.append(item[0])
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
        return result
    
    # 从graph_interact中获取所有的交互记录以及交互总次数
    def get_lrn_scn_num(self):
        sql = '''
        select lrn_uid, scn_uid, all_times
        from graph_interact
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    
    # 从graph_involve中获取所有场景和知识点的难度信息
    def get_scn_cpt_dif(self):
        sql = '''
        select scn_uid, cpt_uid, difficulty
        from graph_involve
        '''
        cursor = self.con.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
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
        return result
db = DB()