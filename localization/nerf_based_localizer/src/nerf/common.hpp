//
// Created by ppwang on 2022/5/8.
//

#ifndef NERF_BASED_LOCALIZER__COMMON_HPP_
#define NERF_BASED_LOCALIZER__COMMON_HPP_

#include <torch/torch.h>

using Slc = torch::indexing::Slice;
const auto CUDAFloat = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

#endif  // NERF_BASED_LOCALIZER__COMMON_HPP_
