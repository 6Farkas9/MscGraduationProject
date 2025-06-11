#include "CD.h"

CD::CD(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {

}

CD::~CD(){

}

std::vector<float> CD::forward(
    const std::string &are_uid, 
    const std::vector<std::vector<float>> &interact_scn_emb,
    const std::vector<std::vector<float>> &scn_emb,
    const std::vector<std::vector<float>> &cpt_emb
) {
    // 构造pt路径
    std::string pt_path = R"(\CD\PT\)" + are_uid + "_use.pt";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    // 加载模型
    model_cd = torch::jit::load(pt_path);
    model_cd.eval();
    // 计算h_lrn
    // 根据interact_scn_emb构建对应的tensor
    std::vector<torch::Tensor> interact_h_scn;
    for (const auto &scn_e : interact_scn_emb){
        interact_h_scn.push_back(torch::from_blob(
            const_cast<float*>(scn_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(scn_e.size())},
            torch::kFloat32
        ));
    }
    // 计算出h_lrn
    torch::Tensor h_lrn = torch::sum(torch::stack(interact_h_scn), 0);
    // 计算h_scn
    std::vector<torch::Tensor> h_scn_stack;
    for (const auto &scn_e : scn_emb){
        h_scn_stack.push_back(torch::from_blob(
            const_cast<float*>(scn_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(scn_e.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor h_scn = torch::stack(h_scn_stack);
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
    // 构建0-special_scn_num - 1的tensor：index和全1tensormask
    int scn_num = scn_emb.size();
    torch::Tensor scn_index = torch::arange(scn_num, torch::kLong);
    torch::Tensor scn_mask = torch::ones(scn_num, torch::kFloat32);
    // 构建输入数据
    scn_index = scn_index.unsqueeze(0);
    scn_mask = scn_mask.unsqueeze(0);
    h_lrn = h_lrn.unsqueeze(0);
    std::vector<torch::jit::IValue> input_data;
    input_data.push_back(scn_index);
    input_data.push_back(scn_mask);
    input_data.push_back(h_lrn);
    input_data.push_back(h_scn);
    input_data.push_back(h_cpt);
    // 输入model获得r_pred
    torch::jit::IValue output_data = model_cd.forward(input_data);
    // 构建结果
    torch::Tensor r_pred = output_data.toTensor();
    auto r_pred_accessor = r_pred.accessor<float, 2>();
    std::vector<float> ans;
    for (int i = 0; i < scn_num; ++i) {
        ans.emplace_back(r_pred_accessor[0][i]);
    }
    return ans;
}

