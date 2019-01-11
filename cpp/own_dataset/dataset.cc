#include <dirent.h>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <torch/torch.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>


class ImageDataset : public torch::data::datasets::Dataset<ImageDataset> {
  private:
    std::string root;
      std::vector<std::string> files;
      std::vector<unsigned int> labels;

  public:
    explicit ImageDataset(const std::string root, const std::string labelfile) : root(root) {
      // get files
      auto p = opendir(root.c_str());
      dirent* entry;
      if(p != nullptr) {
        do {
          entry = readdir(p);

          if(entry != nullptr) {
            if(strcmp(entry->d_name, ".\0") == 0 || strcmp(entry->d_name, "..\0") == 0) continue;
            files.push_back(root + entry->d_name);
          }
        } while(entry != nullptr);
      }

      // get labels
      std::ifstream fs(labelfile);

      int buf;
      while(fs >> buf) {
        labels.push_back(buf);
      }
    }

    torch::data::Example<> get(size_t index) override {
      std::string fname = this->files[index];
      std::cout << fname << std::endl;
      int label = this->labels[index];

      cv::Mat image = cv::imread(fname, 1);
      std::vector<int64_t> sizes = {1, 3, image.rows, image.cols};

      at::Tensor tensor_image = torch::from_blob(image.data, at::IntList(sizes), at::ScalarType::Byte);
      at::Tensor tensor_label = torch::tensor({label}, torch::dtype(torch::kUInt8));

      tensor_image = tensor_image.toType(at::kFloat);

      return {tensor_image, tensor_label};
    }

    at::optional<size_t> size() const override {
      return this->files.size();
    }
};


int main(int argc, char **argv) {
  std::string root = argv[1];
  std::string labelfile = argv[2];

  ImageDataset dataset(root, labelfile);

  auto batch = dataset.get(0);

  std::cout << "input: " << batch.data << std::endl;
  std::cout << "target: " << batch.target << std::endl;


  return 0;
}
