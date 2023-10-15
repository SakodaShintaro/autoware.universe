//
// Created by ppwang on 2022/10/5.
//

#ifndef NERF_BASED_LOCALIZER__CUSTOM_OPS_HPP_
#define NERF_BASED_LOCALIZER__CUSTOM_OPS_HPP_

#include <torch/torch.h>

namespace torch::autograd
{

class TruncExp : public Function<TruncExp>
{
public:
  static variable_list forward(AutogradContext * ctx, Tensor input);

  static variable_list backward(AutogradContext * ctx, variable_list grad_output);
};

}  // namespace torch::autograd

namespace CustomOps
{

torch::Tensor WeightVar(torch::Tensor weights, torch::Tensor idx_start_end);

}

#endif  // NERF_BASED_LOCALIZER__CUSTOM_OPS_HPP_
