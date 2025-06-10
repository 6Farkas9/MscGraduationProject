#include "KT.h"

KT::KT(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {
    
}

KT::~KT(){

}

std::vector<float> KT::forward(
    const std::string &are_uid, 
    const std::vector<std::vector<int>> &interacts, 
    const int &cpt_num  
) {
    // 加载模型
    std::string pt_path = R"(\KT\PT\)" + are_uid + "_use.pt";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    IPDKT = torch::jit::load(pt_path);
    IPDKT.eval();

    // 构建输入
    int interact_num = interacts.size();
    torch::Tensor input_tensor = torch::zeros({1, interact_num, cpt_num * 2}, torch::kFloat32);

    auto input_tensor_accessor = input_tensor.accessor<float, 3>();
    for (int i = 0; i < interact_num; ++i){
        for (int cpt_idx : interacts[i]){
            input_tensor_accessor[0][i][cpt_idx] = 1.0;
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