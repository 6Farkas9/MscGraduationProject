#include "CD.h"

CD::CD(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {

}

CD::~CD(){

}

std::vector<float> CD::forward(
    const std::string &lrn_uid,
    const std::string &are_uid, 
    const std::vector<std::vector<float>> &interact_unt_emb,
    const std::vector<std::vector<float>> &unt_emb,
    const std::vector<std::vector<float>> &cpt_emb
) {
    // 构造pt路径
    std::string pt_path = R"(\CD\PT\)" + are_uid + R"(\)" + lrn_uid + "_use.pt";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    // 加载模型
    model_cd = torch::jit::load(pt_path);
    model_cd.eval();
    // 计算h_lrn
    // 根据interact_unt_emb构建对应的tensor
    std::vector<torch::Tensor> interact_h_unt;
    for (const auto &unt_e : interact_unt_emb){
        interact_h_unt.push_back(torch::from_blob(
            const_cast<float*>(unt_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(unt_e.size())},
            torch::kFloat32
        ));
    }
    // 计算出h_lrn
    torch::Tensor h_lrn = torch::sum(torch::stack(interact_h_unt), 0);
    // 计算h_unt
    std::vector<torch::Tensor> h_unt_stack;
    for (const auto &unt_e : unt_emb){
        h_unt_stack.push_back(torch::from_blob(
            const_cast<float*>(unt_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(unt_e.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor h_unt = torch::stack(h_unt_stack);
    // 计算h_cpt
    std::vector<torch::Tensor> h_cpt_stack;
    for (const auto &cpt_e : cpt_emb){
        h_cpt_stack.push_back(torch::from_blob(
            const_cast<float*>(cpt_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(cpt_e.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor h_cpt = torch::stack(h_cpt_stack);
    // 构建0-special_unt_num - 1的tensor：index和全1tensormask
    int unt_num = unt_emb.size();
    torch::Tensor unt_index = torch::arange(unt_num, torch::kLong);
    torch::Tensor unt_mask = torch::ones(unt_num, torch::kFloat32);
    // 构建输入数据
    unt_index = unt_index.unsqueeze(0);
    unt_mask = unt_mask.unsqueeze(0);
    h_lrn = h_lrn.unsqueeze(0);
    std::vector<torch::jit::IValue> input_data;
    input_data.push_back(unt_index);
    input_data.push_back(unt_mask);
    input_data.push_back(h_lrn);
    input_data.push_back(h_unt);
    input_data.push_back(h_cpt);
    // 输入model获得r_pred
    torch::jit::IValue output_data = model_cd.forward(input_data);
    // 构建结果
    torch::Tensor r_pred = output_data.toTensor();
    auto r_pred_accessor = r_pred.accessor<float, 2>();
    std::vector<float> ans;
    for (int i = 0; i < unt_num; ++i) {
        ans.emplace_back(r_pred_accessor[0][i]);
    }
    return ans;
}

