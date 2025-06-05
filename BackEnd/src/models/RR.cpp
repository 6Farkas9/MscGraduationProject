#include "RR.h"

RR::RR(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop)
{
    auto twotime = MLSTimer().getCurrentand30daysTime();
    now_time = twotime[0];
    thirty_days_ago_time = twotime[1];
}

RR::~RR(){

}

std::unordered_map<std::string, float> RR::forward(const std::string are_uid, const std::string lrn_uid){
    // 获取近30天内关于are_uid的交互记录
    auto interacts = mysqlop.get_Are_lrn_Interacts_Time(are_uid, lrn_uid, thirty_days_ago_time, now_time);
    // 从交互记录中获取交互的scn_uid
    std::unordered_set<std::string> scn_uids, cpt_uids;
    for(auto & interact : interacts) {
        scn_uids.insert(interact[0]);
    }
    // 获取对应scn_uid的KCGE_Emb
    std::unordered_map<std::string, std::vector<float>> interact_scn_emb = mongodbop.get_scn_kcge_by_scn_uid(scn_uids);
    // 根据interact_scn_emb构建对应的tensor
    std::vector<torch::Tensor> interact_h_scn;
    for (const auto &kv : interact_scn_emb){
        interact_h_scn.push_back(torch::from_blob(
            const_cast<float*>(kv.second.data()),  // 避免拷贝数据
            {static_cast<int64_t>(kv.second.size())},
            torch::kFloat32
        ));
    }
    // 计算出h_lrn
    torch::Tensor h_lrn = torch::sum(torch::stack(interact_h_scn), 0);
    interact_scn_emb.clear();
    interact_h_scn.clear();
    scn_uids.clear();
    // 获取are_uid相关的所有special_scn及其对应的cpt
    std::unordered_map<std::string, std::string> special_scn_cpt = mysqlop.get_special_scn_cpt_uid_of_are(are_uid);
    // 获取special_scn和cpt的KCGE_Emb - h_scn和h_cpt
    for (auto &scn_cpt : special_scn_cpt) {
        scn_uids.insert(scn_cpt.first);
        cpt_uids.insert(scn_cpt.second);
    }
    std::unordered_map<std::string, std::vector<float>> scn_emb = mongodbop.get_scn_kcge_by_scn_uid(scn_uids);
    std::unordered_map<std::string, std::vector<float>> cpt_emb = mongodbop.get_cpt_kcge_by_cpt_uid(cpt_uids);
    std::vector<std::string> ordered_scn_uids, ordered_cpt_uids;
    std::vector<torch::Tensor> h_scn_stack;
    for (const auto &kv : scn_emb){
        ordered_scn_uids.emplace_back(kv.first);
        h_scn_stack.push_back(torch::from_blob(
            const_cast<float*>(kv.second.data()),  // 避免拷贝数据
            {static_cast<int64_t>(kv.second.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor h_scn = torch::stack(h_scn_stack);
    std::vector<torch::Tensor> h_cpt_stack;
    for (const auto &kv : cpt_emb){
        ordered_cpt_uids.emplace_back(kv.first);
        h_cpt_stack.push_back(torch::from_blob(
            const_cast<float*>(kv.second.data()),  // 避免拷贝数据
            {static_cast<int64_t>(kv.second.size())},
            torch::kFloat32
        ));
    }
    torch::Tensor h_cpt = torch::stack(h_cpt_stack);
    scn_emb.clear();
    cpt_emb.clear();
    // 构建0-special_scn_num - 1的tensor：index和全1tensormask
    int scn_num = ordered_scn_uids.size();
    torch::Tensor scn_index = torch::arange(scn_num, torch::kLong);
    torch::Tensor scn_mask = torch::ones(scn_num, torch::kFloat32);
    // 加载model
    torch::jit::Module model_cd;
    std::string pt_path = R"(\RR\PT\)" + are_uid + "_use.pt";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    model_cd = torch::jit::load(pt_path);
    model_cd.eval();
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
    torch::jit::IValue output_data = model_cd.forward(input_data);
    // 输入model获得r_pred
    torch::Tensor r_pred = output_data.toTensor();
    // 构建结果
    std::unordered_map<std::string, float> ans;
    auto r_pred_accessor = r_pred.accessor<float, 2>();
    int idx = -1;
    for (auto & cpt_uid : ordered_cpt_uids) {
        ans[cpt_uid] = r_pred_accessor[0][++idx];
    }
    return ans;
}

