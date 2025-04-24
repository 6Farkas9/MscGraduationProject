# import sys

# sys.path.append('../..')

from Data.DBOperator import db
from datetime import datetime, timedelta

class DataReader():
    def __init__(self, are_uid):
        self.are_uid = are_uid
        print(self.are_uid)

    # 这里是要准备KT所需要的交互数据
    # 学习者uid - 数字id
    # 场景uid   - 数字id
    # 学习者数字id    场景数字id  知识点数字id，数字id...     正确与否

    # 获得只包含当前领域相关知识点的场景的交互信息
    def get_all_recordings(self, limit = -1):
        time_start = self.get_30days_before()
        result = db.get_interacts_with_cpt_in_are_from(
                self.are_uid,
                time_start,
                limit
            )
        return result
    
    # 获得当前领域的所有知识点，cpt_uid和id_in_area
    def get_all_concepts_of_area(self):
        result = db.get_all_concepts_of_area(self.are_uid)
        cpt_uid2id_in = {}
        for line in result:
            cpt_uid2id_in[line[0]] = line[1]
        return cpt_uid2id_in

    def get_30days_before(self):
        return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    
    def get_concepts_of_scene(self, scn_uids):
        return db.get_concepts_of_scenes(scn_uids)
    
    def get_concept_num_of_area(self):
        return db.get_concept_num_of_area(self.are_uid)[0]
    
    def get_cpt_id_in_area_of_scene(self, scn_uids):
        return db.get_concepts_id_in_area_of_scenes(scn_uids)

    # 从数据库中获取所有数据
    def load_data_from_db(self):
        # 获取当前领域所有知识点相关的场景的交互信息
        # result = [(lrn_uid, scn_uid, correct),...]
        interacts = self.get_all_recordings()
        cpt_num = self.get_concept_num_of_area()
        # 构建lrn和scn的数字id -- 貌似，转化就行
        lrn_uids = {}
        scn_uids = set()
        for interact in interacts:
            if interact[0] not in lrn_uids:
                temp_id = len(lrn_uids)
                lrn_uids[interact[0]] = temp_id
            scn_uids.add(interact[1])
            # if interact[1] not in scn_uids:
            #     temp_id = len(scn_uids)
            #     scn_uids[interact[1]] = temp_id
        scn_cpts = self.get_cpt_id_in_area_of_scene(list(scn_uids))
        data = []
        current_pos = -1
        current_lrn = -1
        for interact in interacts:
            lrn_id = lrn_uids[interact[0]]
            # scn_id = scn_uids[interact[1]]
            correct = float(interact[2])
            if lrn_id != current_lrn:
                current_lrn = lrn_id
                data.append([])
                current_pos += 1
                data[current_pos].append(lrn_id)
                # data[current_pos].append([]) # scn_uids
                data[current_pos].append([]) # cpt_ids
                data[current_pos].append([]) # corrects
            # data[current_pos][1].append(scn_id)
            data[current_pos][1].append(scn_cpts[interact[1]])
            data[current_pos][2].append(correct)
        train_data = []
        master_data = []
        pos = -1
        for onelrndata in data:
            scn_num = len(onelrndata[2])
            if scn_num == 1:
                continue
            scn_train_num = int(scn_num * 0.8)
            # scn_master_num = scn_num - scn_train_num
            pos += 1
            train_data.append([0, [] ,[]])
            master_data.append([0, [] ,[]])
            train_data[pos][0] = onelrndata[0]
            master_data[pos][0] = onelrndata[0]
            
            for i in range(scn_train_num):
                train_data[pos][1].append(onelrndata[1][i])
                train_data[pos][2].append(onelrndata[2][i])
                # train_data[pos][3].append(onelrndata[3][i])

            for i in range(scn_train_num, scn_num):
                master_data[pos][1].append(onelrndata[1][i])
                master_data[pos][2].append(onelrndata[2][i])
                # master_data[pos][3].append(onelrndata[3][i])
            
        return train_data, master_data, cpt_num

    # 获取当前时间前推30天内的所有interacts数据
    
    # def load_data(self):
    #     data = []
    #     data_f = open(self.dir, 'r')
    #     line = data_f.readline()
    #     current_stu_id = ''
    #     temp = []
    #     while line:
    #         line = line.strip('\n')
    #         line = line.split('\t')
    #         if line[0] != current_stu_id:
    #             if current_stu_id:
    #                 data.append(temp)
    #             current_stu_id = line[0]
    #             temp = []
    #             temp.append(int(current_stu_id))
    #             for i in range(3):
    #                 temp.append([])
            
    #         temp[1].append(int(line[1]))

    #         temp_item = line[2].split(',')
    #         for i in range(len(temp_item)):
    #             temp_item[i] = int(temp_item[i]) - 1
    #         temp[2].append(temp_item)

    #         temp[3].append(float(line[3]))
    #         temp_item = line[4].split(',')
    #         for i in range(len(temp_item)):
    #             temp_item[i] = float(temp_item[i])
    #         temp[4].append(temp_item)
    #         line = data_f.readline()
    #     data_f.close()
    #     return data,max_pro,kc_num

if __name__ == '__main__':
    dr = DataReader('are_3fee9e47d0f3428382f4afbcb1004117')
    # are_3fee9e47d0f3428382f4afbcb1004117
    data, num = dr.load_data_from_db()
    print(num)
    print(data)