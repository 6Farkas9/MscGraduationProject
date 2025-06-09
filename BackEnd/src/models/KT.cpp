#include "KT.h"

KT::KT(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop)
{
    auto twotime = MLSTimer().getCurrentand30daysTime();
    now_time = twotime[0];
    thirty_days_ago_time = twotime[1];

    // std::cout << now_time << std::endl;
    // std::cout << thirty_days_ago_time << std::endl;
}

KT::~KT(){

}

/*
    1.根据当前时间获取一个月前的时间得到两个时间界限
    2.根据are_uid获取对应的pt文件
    3.加载pt文件
    4.根据are_uid、lrn_uid、时间界限，获取在该领域内的时间界限内的该学生的学习记录
    5.输入到模型中获得结果
    6.获取最后一个的预测结果作为返回
    */
std::vector<float> KT::forward(const std::string &are_uid, const std::string &lrn_uid){
    
    torch::jit::Module IPDKT;
    // are_uid获取对应的PT文件先暂时略过，没想好怎么去组织这个文件格式
    std::string pt_path = R"(\KT\PT\)" + are_uid + "_use.pt";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    std::cout << pt_path << std::endl;
    IPDKT = torch::jit::load(pt_path);
    // IPDKT.to(torch::kCPU);
    IPDKT.eval();
    
    // torch::Device model_device = (*IPDKT.parameters().begin()).device();

    // 构建输入数据
    // 获取当前领域的所有知识点
    auto cpt_uids = mysqlop.get_cpt_uid_id_of_area(are_uid);
    // 获取当前领域的知识点的该学生的一个月内的交互数据
    auto interacts = mysqlop.get_Are_lrn_Interacts_Time(
        are_uid,
        lrn_uid,
        thirty_days_ago_time,
        now_time
    );
    // 获取交互过的所有场景uid
    std::unordered_set<std::string> scn_uids;
    for(auto & interact : interacts) {
        scn_uids.insert(interact[0]);
    }
    // 获取每个场景所涉及的知识点
    auto scn_cpt = mysqlop.get_Cpt_of_Scn(scn_uids);

    int interact_num = interacts.size();
    int cpt_num = cpt_uids.size();


    torch::Tensor input_tensor = torch::zeros({1, interact_num, cpt_num * 2}, torch::kFloat32);

    auto input_tensor_accessor = input_tensor.accessor<float, 3>();
    for (int i = 0; i < interact_num; ++i){
        int interact_result;
        std::istringstream(interacts[i][1]) >> interact_result;
        int skip = (1 - interact_result) * cpt_num;
        for (auto & cpt_uid : scn_cpt[interacts[i][0]]){
            input_tensor_accessor[0][i][cpt_uids[cpt_uid] + skip] = 1.0;
        }
    }

    std::vector<torch::jit::IValue> input_data;
    input_data.push_back(input_tensor);
    torch::jit::IValue output_data = IPDKT.forward(input_data);
    torch::Tensor output_tensor = output_data.toTensor();

    std::vector<float> ans(cpt_num);

    auto output_tensor_accessor = output_tensor.accessor<float, 3>();
    for(int i = 0; i < cpt_num; ++i){
        ans[i] = output_tensor_accessor[0][interact_num - 1][i];
    }

    return ans;
}