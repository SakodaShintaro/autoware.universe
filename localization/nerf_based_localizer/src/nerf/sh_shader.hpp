//
// Created by ppwang on 2022/10/8.
//

#ifndef NERF_BASED_LOCALIZER__SH_SHADER_HPP_
#define NERF_BASED_LOCALIZER__SH_SHADER_HPP_

#include <torch/torch.h>

class SHShader : public torch::nn::Module
{
  using Tensor = torch::Tensor;

public:
  SHShader();

  Tensor query(const Tensor & feats, const Tensor & dirs);

  std::vector<torch::optim::OptimizerParamGroup> optim_param_groups(float lr);

private:
  static constexpr int DEGREE = 4;

  Tensor encode(const Tensor & dirs);

  torch::nn::Sequential mlp_ = nullptr;
};

#endif  // NERF_BASED_LOCALIZER__SH_SHADER_HPP_
