//
// Created by ppwang on 2023/2/11.
//

#ifndef NERF_BASED_LOCALIZER__FLEX_OPS_HPP_
#define NERF_BASED_LOCALIZER__FLEX_OPS_HPP_

#include "../common.hpp"

#include <torch/torch.h>

namespace FlexOps
{

torch::Tensor Sum(torch::Tensor val, torch::Tensor idx_start_end);
torch::Tensor AccumulateSum(torch::Tensor val, torch::Tensor idx_start_end, bool include_this);

}  // namespace FlexOps

#endif  // NERF_BASED_LOCALIZER__FLEX_OPS_HPP_
