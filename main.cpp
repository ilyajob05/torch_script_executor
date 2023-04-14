#include <memory>
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include "torch_script_executor.hpp"


using namespace std;

int main()
{
    auto tse = TSE("/home/ilya/PRJ/CarClassify/div_tools/traced_qsc.zip");
    at::Tensor data = torch::randn({1, 3, 224, 224});
    int class_num = tse.predict(data);
    cout << "class num: " << class_num << endl;
    return 0;
}
