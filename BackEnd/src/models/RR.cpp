#include "RR.h"

RR::RR(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop)
{
    auto twotime = MLSTimer::getCurrentand30daysTime();
    now_time = twotime[0];
    thirty_days_ago_time = twotime[1];
}

RR::~RR(){

}

std::unordered_map<std::string, float> RR::forward(const std::string lrn_uid){
    // 获取指定lrn的HGC_Emb
    std::unordered_set<std::string> lrn_uids, scn_uids, cpt_uids;
    lrn_uids.insert(lrn_uid);
    std::unordered_map<std::string, std::vector<float>> lrn_emb_map = mongodbop.get_lrn_hgc_by_lrn_uid(lrn_uids);
    // 构建lrn_emb的tensor
    torch::Tensor lrn_emb = torch::from_blob(
        const_cast<float*>(lrn_emb_map[lrn_uid].data()),  // 避免拷贝数据
        {static_cast<int64_t>(lrn_emb_map[lrn_uid].size())},
        torch::kFloat32
    );
    // 获取近30天内lrn_uid的交互记录
    auto interacts = mysqlop.get_lrn_interacts_time(lrn_uid, thirty_days_ago_time, now_time);
    // 获取交互记录中的scn_uids
    for(auto & interact : interacts) {
        scn_uids.insert(interact[0]);
    }
    // 获取scn_uids对应的HGC_Emb
    std::unordered_map<std::string, std::vector<float>> scn_emb_map = mongodbop.get_scn_hgc_by_scn_uid(scn_uids);
    std::unordered_map<std::string, int> scn_uid2idx;
    std::vector<torch::Tensor> scn_emb_vec;
    int idx = 0;
    for (const auto &kv : scn_emb_map){
        scn_emb_vec.push_back(torch::from_blob(
            const_cast<float*>(kv.second.data()),  // 避免拷贝数据
            {static_cast<int64_t>(kv.second.size())},
            torch::kFloat32
        ));
        scn_uid2idx[kv.first] = idx;
        ++idx;
    }
    torch::Tensor scn_emb = torch::stack(scn_emb_vec);
    // 构建scn_index和scn_mask
    int interact_num = interacts.size();
    std::vector<int> scn_index_vec;
    for (auto &interact : interacts) {
        scn_index_vec.emplace_back(scn_uid2idx[interact[0]]);
    }
    torch::Tensor scn_index = torch::tensor(scn_index_vec, torch::kLong);
    torch::Tensor scn_mask = torch::ones(interact_num, torch::kFloat32);
    // 获取所有知识点（涉及推荐范围）的HGC_Emb
    std::unordered_map<std::string, std::vector<float>> cpt_emb_map = mongodbop.get_all_cpt_hgc();
    std::vector<std::string> ordered_cpt_uid;
    std::vector<torch::Tensor> cpt_emb_vec;
    for (const auto &kv : cpt_emb_map){
        cpt_emb_vec.push_back(torch::from_blob(
            const_cast<float*>(kv.second.data()),  // 避免拷贝数据
            {static_cast<int64_t>(kv.second.size())},
            torch::kFloat32
        ));
        ordered_cpt_uid.emplace_back(kv.first);
    }
    torch::Tensor cpt_emb = torch::stack(cpt_emb_vec);
    // 加载模型
    torch::jit::Module model_rr;
    std::string pt_path = R"(\RR\PT\RR_use.pt)";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    model_rr = torch::jit::load(pt_path);
    model_rr.eval();
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
    std::unordered_map<std::string, float> ans;
    auto r_pred_accessor = r_pred.accessor<float, 2>();
    idx = -1;
    for (auto & cpt_uid : ordered_cpt_uid) {
        ans[cpt_uid] = r_pred_accessor[0][++idx];
    }
    return ans;
}

