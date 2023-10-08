//
// Created by ppwang on 2022/5/11.
//

#ifndef NERF_BASED_LOCALIZER__UTILS_HPP_
#define NERF_BASED_LOCALIZER__UTILS_HPP_

#include <torch/torch.h>

#include <string>

namespace utils
{
using Tensor = torch::Tensor;

Tensor read_image_tensor(const std::string & path);
bool write_image_tensor(const std::string & path, Tensor img);
Tensor resize_image(Tensor image, const int resize_height, const int resize_width);
float calc_loss(Tensor pred_image, Tensor gt_image);

}  // namespace utils

#endif  // NERF_BASED_LOCALIZER__UTILS_HPP_
