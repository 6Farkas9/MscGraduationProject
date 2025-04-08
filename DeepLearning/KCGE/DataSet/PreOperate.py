import pandas as pd
from tqdm import tqdm
import random

# 1.把试题名称换成数字id
# 2.把学生id如果需要处理则处理为数字id
# 3.构建当前的知识图谱
#     3.1.可能需要系数矩阵存储
#     3.2.知识点名称替换为id-目前看一个试题一个知识点，很怪
#     3.3.是否应该构建非完整的三元组？可能以三元组存储更合适一些
# 4.最终的文件应该是：
#     ①做题记录文件：train+test
#     ②试题间关系文件
#     ③试题topic关系文件
#     ④topic-area关系文件
#     将三种关系文件合成一个文件，内记录所有的关系内容，每组关系用一个四元组表示(实体1，实体2，关系类型，关系数值)
#     其中关系类型：0：相似度，1：难度，2：先决条件，3：对应（试题和topic），4：包含/被包含（tpoic和area）
# 5.train和test文件以7:3划分，由于数据量足够了吧，所以不拆分多次尝试，元组结构为：(stuid,exerid,correct)

def get_First(elem):
    return elem[0]

print('processing exercise info file...')
reader_exer_info = pd.read_csv('../原数据/JunYi/junyi_Exercise_table.csv',usecols=['name','topic','area'], header=0, on_bad_lines='skip')

len_row,len_col = reader_exer_info.shape

entity_all = {}
id_all = 0

relation_all = []

for i in tqdm(range(len_row)):
    exer_id = 0
    topic_id = 0
    area_id = 0

    exer_name = reader_exer_info.iloc[i,0]
    topic_name = reader_exer_info.iloc[i,1]
    area_name = reader_exer_info.iloc[i,2]

    if exer_name not in entity_all.keys():
        entity_all[exer_name] = id_all
        id_all += 1
    exer_id = entity_all[exer_name]
    if topic_name not in entity_all.keys():
        entity_all[topic_name] = id_all
        id_all += 1
    topic_id = entity_all[topic_name]
    if area_name not in entity_all.keys():
        entity_all[area_name] = id_all
        id_all += 1
    area_id = entity_all[area_name]

    corr_relat = [exer_id, topic_id, 3, 1]
    contain_relat = [topic_id, area_id, 4, 1]

    if corr_relat not in relation_all:
        relation_all.append(corr_relat)
    if contain_relat not in relation_all:
        relation_all.append(contain_relat)
    
del reader_exer_info

print('processing relation file...')
reader_exer_rela = pd.read_csv(
    '../原数据/JunYi/relationship_annotation.csv',usecols=['Exercise_A','Exercise_B','Similarity_avg','Difficulty_avg','Prerequisite_avg'], header=0, on_bad_lines='skip'
)

len_row,len_col = reader_exer_rela.shape

for i in tqdm(range(len_row)):
    exer_name_1 = reader_exer_rela.iloc[i,0]
    exer_name_2 = reader_exer_rela.iloc[i,1]
    simi = reader_exer_rela.iloc[i,2]
    diff = reader_exer_rela.iloc[i,3]
    prer = reader_exer_rela.iloc[i,4]

    exer_id_1 = entity_all[exer_name_1]
    exer_id_2 = entity_all[exer_name_2]

    simi_rela = [exer_id_1, exer_id_2, 0, simi]
    diff_rela = [exer_id_1, exer_id_2, 1, diff]
    prer_rela = [exer_id_1, exer_id_2, 2, prer]

    if simi_rela not in relation_all:
        relation_all.append(simi_rela)
    if diff_rela not in relation_all:
        relation_all.append(diff_rela)
    if prer_rela not in relation_all:
        relation_all.append(prer_rela)

del reader_exer_rela

relation_all = pd.DataFrame(relation_all,columns=['entity1','entity2','relation_type','value'])
relation_all.to_csv('relations.csv',index=False)

del relation_all

print('reletion file precessed')
print('processing log file...')

reader_log = pd.read_csv('../原数据/JunYi/junyi_ProblemLog_original.csv',usecols=['user_id','exercise','correct'], header=0, on_bad_lines='skip')

# exer_all_dict = {}
# for i in range(len(exer_all)):
#     exer_all_dict[exer_all[i]] = i
# del exer_all

len_row,len_col = reader_log.shape
log_all = []
stu_all = {}
stu_i = 0
for i in tqdm(range(len_row)):
    stu_name = int(reader_log.iloc[i,0])
    if stu_name not in stu_all.keys():
        stu_all[stu_name] = stu_i
        stu_i += 1
    stu_id = stu_all[stu_name]
    exer_id = entity_all[reader_log.iloc[i,1]]
    correct = 1 if reader_log.iloc[i,2] else 0
    log_all.append([stu_id, exer_id, correct])

del stu_all

len_log = len(log_all)
offset = int(len_log * 0.7)
random.shuffle(log_all)
log_train = log_all[:offset]
log_test = log_all[offset:]
del log_all

print('data processed and divided,sorting...')
log_train.sort(key=get_First)
log_test.sort(key=get_First)

print('data sorted')

log_train = pd.DataFrame(log_train,columns=['student','exercise','correct'])
log_test = pd.DataFrame(log_test,columns=['student','exercise','correct'])

log_train.to_csv('train.csv',index=False)
log_test.to_csv('test.csv',index=False)

print('done')
