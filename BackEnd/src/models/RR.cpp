#include "RR.h"

RR::RR(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {
    
}

RR::~RR(){

}

std::vector<float> RR::forward(
    const std::vector<float> &lrn_emb_in,
    const std::vector<std::vector<float>> &scn_emb_in,
    const std::vector<std::vector<float>> &cpt_emb_in,
    const std::vector<int> &scn_index_vec
) {
    // 构造pt路径
    std::string pt_path = R"(\RR\PT\RR_use.pt)";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    // 加载模型
    model_rr = torch::jit::load(pt_path);
    model_rr.eval();
    // 构建lrn_emb的tensor
    torch::Tensor lrn_emb = torch::from_blob(
        const_cast<float*>(lrn_emb_in.data()),  // 避免拷贝数据
        {static_cast<int64_t>(lrn_emb_in.size())},
        torch::kFloat32
    );
    // 构造h_scn的tensor
    std::vector<torch::Tensor> scn_emb_vec;
    int idx = 0;
    for (const auto &scn_e : scn_emb_in){
        scn_emb_vec.push_back(torch::from_blob(
            const_cast<float*>(scn_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(scn_e.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor scn_emb = torch::stack(scn_emb_vec);
    // 构造h_cpt
    std::vector<torch::Tensor> cpt_emb_vec;
    for (const auto &cpt_e : cpt_emb_in){
        cpt_emb_vec.push_back(torch::from_blob(
            const_cast<float*>(cpt_e.data()),  // 避免拷贝数据
            {static_cast<int64_t>(cpt_e.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor cpt_emb = torch::stack(cpt_emb_vec);
    // 构造scn_index和scn_mask
    int interact_num = scn_index_vec.size();
    torch::Tensor scn_index = torch::tensor(scn_index_vec, torch::kLong);
    torch::Tensor scn_mask = torch::ones(interact_num, torch::kFloat32);
    // 构建输入
    lrn_emb = lrn_emb.unsqueeze(0);
    scn_index = scn_index.unsqueeze(0);
    scn_mask = scn_mask.unsqueeze(0);
    std::vector<torch::jit::IValue> input_data;
    input_data.push_back(lrn_emb);
    input_data.push_back(scn_emb);
    input_data.push_back(scn_index);
    input_data.push_back(scn_mask);
    input_data.push_back(cpt_emb);
    // 数据输入模型
    torch::jit::IValue output_data = model_rr.forward(input_data);
    // 构建输出
    torch::Tensor r_pred = output_data.toTuple()->elements()[0].toTensor();
    // 构建结果
    auto r_pred_accessor = r_pred.accessor<float, 2>();
    int cpt_num = cpt_emb_in.size();
    std::vector<float> ans;
    for (int i = 0; i < cpt_num; ++i) {
        ans.emplace_back(r_pred_accessor[0][i]);
    }
    return ans;
}

