#include "KCGE.h"

KCGE::KCGE(MySQLOperator &mysqlop, MongoDBOperator &mongodbop) :
    mysqlop(mysqlop),
    mongodbop(mongodbop) {

}

KCGE::~KCGE(){

}

std::vector<std::vector<float>> KCGE::forward(
    const std::vector<std::vector<float>> &x_vec,
    const std::vector<std::vector<int>> &edge_index_vec,
    const std::vector<int> &edge_type_vec,
    const std::vector<float> &edge_attr_vec
) {
    // 构造pt路径
    std::string pt_path = R"(\KCGE\PT\KCGE_use.pt)";
    pt_path = DEEPLEARNING_ROOT + pt_path;
    // 加载模型
    model_kcge = torch::jit::load(pt_path);
    model_kcge.eval();
    // 构造x的tensor
    std::vector<torch::Tensor> x_stack;
    for (const auto &x_e : x_vec){
        x_stack.push_back(
            torch::from_blob(
                const_cast<float*>(x_e.data()),  // 避免拷贝数据
                {static_cast<int64_t>(x_e.size())},
                torch::kFloat32
            )
        );
    }
    // 计算出x
    torch::Tensor x = torch::sum(torch::stack(x_stack), 0);


}