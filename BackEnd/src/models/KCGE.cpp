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
    try{
        model_kcge = torch::jit::load(pt_path, torch::kCPU);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
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
    torch::Tensor x = torch::stack(x_stack);
    // 构造edge_index
    std::vector<torch::Tensor> edge_index_stack;
    for (const auto &edge_index_e : edge_index_vec){
        edge_index_stack.push_back(
            torch::from_blob(
                const_cast<int*>(edge_index_e.data()),  // 避免拷贝数据
                {static_cast<int64_t>(edge_index_e.size())},
                torch::kLong
            )
        );
    }
    torch::Tensor edge_index = torch::stack(x_stack);
    // 构造edge_type
    torch::Tensor edge_type = torch::from_blob(
        const_cast<int*>(edge_type_vec.data()),  // 避免拷贝数据
        {static_cast<int64_t>(edge_type_vec.size())},
        torch::kLong
    );
    // 构造edge_attr
    torch::Tensor edge_attr = torch::from_blob(
        const_cast<float*>(edge_attr_vec.data()),  // 避免拷贝数据
        {static_cast<int64_t>(edge_attr_vec.size())},
        torch::kFloat32
    );
    // 构造输入
    std::vector<torch::jit::IValue> input_data;
    input_data.push_back(x);
    input_data.push_back(edge_index);
    input_data.push_back(edge_type);
    input_data.push_back(edge_attr);
    // 输入模型
    torch::jit::IValue output_data = model_kcge.forward(input_data);
    // 构建结果
    torch::Tensor r_pred = output_data.toTensor();
    auto r_pred_accessor = r_pred.accessor<float, 2>();
    int item_num = x_vec.size();
    int item_dim = x_vec[0].size();
    std::vector<std::vector<float>> ans(item_num, std::vector<float>(item_dim, 0));
    for (int i = 0; i < item_num; ++i) {
        for (int j = 0; j < item_dim; ++j) {
            ans[i][j] = r_pred_accessor[i][j];
        }
    }
    return ans;
}