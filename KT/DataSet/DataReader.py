class DataReader():
    def __init__(self,dir):
        self.dir = dir

# 数据集中一行为一次做题数据
# 0	0	1,2	1.0000,1.0000	1
# 依次为学生id，题目id，涉及知识点id，知识点正确率，本次正确与否

    def load_data(self):
        data = []
        data_f = open(self.dir,'r')
        line = data_f.readline()
        line = line.strip('\n')
        line = line.split('\t')
        max_pro = int(line[0])
        kc_num = int(line[1])
        line = data_f.readline()
        current_stu_id = ''
        temp = []
        while line:
            line = line.strip('\n')
            line = line.split('\t')
            if line[0] != current_stu_id:
                if current_stu_id:
                    data.append(temp)
                current_stu_id = line[0]
                temp = []
                temp.append(int(current_stu_id))
                for i in range(4):
                    temp.append([])
            
            temp[1].append(int(line[1]))

            temp_item = line[2].split(',')
            for i in range(len(temp_item)):
                temp_item[i] = int(temp_item[i]) - 1
            temp[2].append(temp_item)

            temp[3].append(float(line[3]))
            temp_item = line[4].split(',')
            for i in range(len(temp_item)):
                temp_item[i] = float(temp_item[i])
            temp[4].append(temp_item)
            line = data_f.readline()
        data_f.close()
        return data,max_pro,kc_num

    