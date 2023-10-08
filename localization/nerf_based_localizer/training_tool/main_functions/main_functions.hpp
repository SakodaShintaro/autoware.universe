#ifndef NERF_BASED_LOCALIZER__MAIN_FUNCTIONS_HPP_
#define NERF_BASED_LOCALIZER__MAIN_FUNCTIONS_HPP_

#include <string>

void walk(const std::string & train_result_dir);
void test(const std::string & train_result_dir, const std::string & dataset_dir);
void infer(const std::string & train_result_dir, const std::string & dataset_dir);

#endif  // NERF_BASED_LOCALIZER__MAIN_FUNCTIONS_HPP_
