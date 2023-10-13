//
// Created by ppwang on 2022/5/8.
//

#ifndef NERF_BASED_LOCALIZER__COMMON_CUDA_HPP_
#define NERF_BASED_LOCALIZER__COMMON_CUDA_HPP_

#include <cuda_runtime.h>

inline unsigned int DivUp(const unsigned int x, const unsigned int y)
{
  return (x + y - 1) / y;
}

constexpr unsigned int THREAD_CAP = 512;
constexpr dim3 LIN_BLOCK_DIM = {THREAD_CAP, 1, 1};

inline dim3 LIN_GRID_DIM(const int x)
{
  return dim3{DivUp(x, THREAD_CAP), 1, 1};
}

#endif  // NERF_BASED_LOCALIZER__COMMON_CUDA_HPP_
