#include <torch/torch.h>


struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(8, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, 0.5);
    x = torch::sigmoid(fc2->forward(x));

    return x;
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

Net net;

auto data_loader = torch::data::data_loader(torch::data::datasets::MNIST("./data"));

torch::optim::SGD optimizer(net.parameters(), 0.1);

for(size_t epoch=1; epoch<=10; ++epoch) {
  size_t batch_index = 0;
  for(auto batch : data_loader) {
    optimizer.zero_grad();
    auto prediction = model.forward(batch.data);
    auto loss = torch::binary_cross_entropy(prediction, batch.label);
    loss.backward();
    optimizer.step();

    if(batch_index++ % 10 == 0) {
      std::cout << "Epoch: " << epoch << "| Loss: " << loss << std::endl;
      torch::save(net, "net.pt")
    }
  }
}
