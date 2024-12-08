from tqdm import tqdm

file_dirs = [
    '../Data/algebra_5_6/'
    ,'../Data//algebra_6_7/'
]

for i in range(len(file_dirs)):
    print(f'heading file {i+1}')

    train_file_read = open(file_dirs[i] + 'train.txt', 'r')
    master_file_read = open(file_dirs[i] + 'master.txt', 'r')

    train_file_write = open(file_dirs[i] + 'train_processed.txt', 'w')
    master_file_write = open(file_dirs[i] + 'master_processed.txt', 'w')

    print('getting pre info...')

    train_data_origin = train_file_read.readlines()
    master_data_origin = master_file_read.readlines()

    firstline = train_data_origin[0].split("\t")
    stu_id_loc = firstline.index("Anon Student Id")
    prob_id_loc = firstline.index("Problem Name")
    cor_ft_loc = firstline.index("Correct First Attempt")
    incors_nums_loc = firstline.index("Incorrects")
    kc_loc = 0

    for item in firstline:
        if 'KC' in item:
            kc_loc = firstline.index(item)

    train_data_origin.pop(0)
    master_data_origin.pop(0)

    max_pro_singlestu = 0
    kc_all = []
    stu_id_all = []
    pro_id_all = []

    train_data = []
    master_data = []

    read_files = [train_file_read, master_file_read]
    write_files = [train_file_write, master_file_write]

    origin_data_s = [train_data_origin, master_data_origin]
    data_store = [train_data,master_data]

    for i in range(len(origin_data_s)):
        last_stu_id = ''
        data_to = data_store[i]
        current_stu_loc = {}
        j = 0

        tbar = tqdm(origin_data_s[i])

        for line in tbar:
            line_data = line.split('\t')
            if line_data[kc_loc] == '':
                continue
            stuid = line_data[stu_id_loc]
            if stuid not in stu_id_all:
                stu_id_all.append(stuid)
                stu_id_num = len(stu_id_all) # 编号，从1开始
                # data_to.append([stu_id_num,[],[],[],[]]) # 学生编号，题目编号，kc，正确与否，正确率
            else:
                stu_id_num = stu_id_all.index(stuid) + 1

            if stuid in current_stu_loc.keys():
                stu_loc = current_stu_loc[stuid]
            else:
                data_to.append([stu_id_num,[],[],[],[]]) # 学生编号，题目编号，kc，正确与否，正确率
                current_stu_loc[stuid] = j
                stu_loc = j
                j += 1
            if stuid != last_stu_id:
                if last_stu_id:
                    pro_singlestu = len(data_to[current_stu_loc[last_stu_id]][1])
                    if pro_singlestu > max_pro_singlestu:
                        max_pro_singlestu = pro_singlestu
                last_stu_id = stuid

            proid = line_data[prob_id_loc]
            if  proid not in pro_id_all:
                pro_id_all.append(proid)
                proid = len(pro_id_all) # 编号，从1开始
                # proid -= 1
            else:
                proid = pro_id_all.index(proid) + 1
            # data_to[stuid][1].append(proid)
            kcs_origin = line_data[kc_loc].split('~~')
            temp_kc = []
            for kc_origin in kcs_origin:
                if "SkillRule: " in kc_origin:
                    start_loc = 12
                    end_loc = kc_origin.find(';')
                    kc_now = kc_origin[start_loc:end_loc]
                else:
                    kc_now = kc_origin.replace("-sp","")
                if kc_now in kc_all:
                    kc_id_now = kc_all.index(kc_now) + 1
                else:
                    kc_all.append(kc_now)
                    kc_id_now = len(kc_all)
                temp_kc.append(kc_id_now)
            # data_to[stuid][2].append(temp_kc)
            for i in range(int(line_data[incors_nums_loc])):
                data_to[stu_loc][1].append(proid)
                data_to[stu_loc][2].append(temp_kc)
                data_to[stu_loc][3].append(0)
            data_to[stu_loc][1].append(proid)
            data_to[stu_loc][2].append(temp_kc)
            data_to[stu_loc][3].append(1)
    
    for item in origin_data_s:
        del item
    for item in read_files:
        item.close()
    
    print(f'max:{max_pro_singlestu},kc_num:{len(kc_all)}')
    print('pre info got,gett    ing rate...')
                
    for data in data_store:
        for line in data:
            sum_num = [0] * len(kc_all)
            cor_num = [0] * len(kc_all)
            for i in range(len(line[1])):
                line[4].append([0] * len(kc_all))
                for kc in line[2][i]:
                    sum_num[kc - 1] += 1
                    if line[3][i] == 1:
                        cor_num[kc - 1] += 1
                    line[4][i][kc - 1] = float(cor_num[kc - 1] / sum_num[kc - 1])
    
    print('rate got,writting file...')

    for i in range(len(write_files)):

        print(f'writing file {i+1}')

        write_files[i].write(str(max_pro_singlestu) + '\t' + str(len(kc_all)) + '\n')

        tbar = tqdm(data_store[i])
        for line in tbar:
            for j in range(len(line[1])):
                current_line = ''
                current_line = str(line[0]) + '\t' + str(line[1][j]) + '\t'
                temp_line = ''
                for k in range(len(line[2][j])):
                    temp_line += str(line[2][j][k]) + ','
                current_line += temp_line[:-1] + '\t' + str(line[3][j]) + '\t'
                temp_line = ''
                for k in range(len(kc_all)):
                    temp_line += '{:.4f}'.format(line[4][j][k]) + ','
                current_line += temp_line[:-1] + '\n'
                write_files[i].write(current_line)




