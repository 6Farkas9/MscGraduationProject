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
    
db = DB()