// Copyright 2024 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "localization_util/covariance_ellipse.hpp"

#include <tf2/utils.h>
#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

namespace autoware::localization_util
{

Ellipse calculate_xy_ellipse(
  const geometry_msgs::msg::PoseWithCovariance & pose_with_covariance, const double scale)
{
  // input geometry_msgs::PoseWithCovariance contain 6x6 matrix
  Eigen::Matrix2d xy_covariance;
  const auto cov = pose_with_covariance.covariance;
  xy_covariance(0, 0) = cov[0 * 6 + 0];
  xy_covariance(0, 1) = cov[0 * 6 + 1];
  xy_covariance(1, 0) = cov[1 * 6 + 0];
  xy_covariance(1, 1) = cov[1 * 6 + 1];

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(xy_covariance);

  Ellipse ellipse;
  ellipse.P = xy_covariance;
  const Eigen::Matrix2d & p_inv = ellipse.P.inverse();

  // eigen values and vectors are sorted in ascending order
  ellipse.long_radius = scale * std::sqrt(eigensolver.eigenvalues()(1));
  ellipse.short_radius = scale * std::sqrt(eigensolver.eigenvalues()(0));

  // principal component vector
  const Eigen::Vector2d pc_vector = eigensolver.eigenvectors().col(1);
  ellipse.yaw = std::atan2(pc_vector.y(), pc_vector.x());

  const double yaw_vehicle = tf2::getYaw(pose_with_covariance.pose.orientation);

  // ellipse size along lateral direction (body-frame)
  Eigen::MatrixXd rot_to_lat(2, 1);
  rot_to_lat(0, 0) = std::cos(yaw_vehicle);
  rot_to_lat(1, 0) = std::sin(yaw_vehicle);
  const double d_lat =
    std::sqrt((rot_to_lat.transpose() * p_inv * rot_to_lat)(0, 0) / p_inv.determinant());
  ellipse.size_lateral_direction = scale * d_lat;

  // ellipse size along longitudinal direction (body-frame)
  Eigen::MatrixXd rot_to_lon(2, 1);
  rot_to_lon(0, 0) = std::cos(yaw_vehicle + M_PI_2);
  rot_to_lon(1, 0) = std::sin(yaw_vehicle + M_PI_2);
  const double d_lon =
    std::sqrt((rot_to_lon.transpose() * p_inv * rot_to_lon)(0, 0) / p_inv.determinant());
  ellipse.size_longitudinal_direction = scale * d_lon;

  return ellipse;
}

visualization_msgs::msg::Marker create_ellipse_marker(
  const Ellipse & ellipse, const std_msgs::msg::Header & header,
  const geometry_msgs::msg::PoseWithCovariance & pose_with_covariance)
{
  tf2::Quaternion quat;
  quat.setEuler(0, 0, ellipse.yaw);

  const double ellipse_long_radius = std::min(ellipse.long_radius, 30.0);
  const double ellipse_short_radius = std::min(ellipse.short_radius, 30.0);
  visualization_msgs::msg::Marker marker;
  marker.header = header;
  marker.ns = "error_ellipse";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::SPHERE;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose = pose_with_covariance.pose;
  marker.pose.orientation = tf2::toMsg(quat);
  marker.scale.x = ellipse_long_radius * 2;
  marker.scale.y = ellipse_short_radius * 2;
  marker.scale.z = 0.01;
  marker.color.a = 0.1;
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 1.0;
  return marker;
}

}  // namespace autoware::localization_util
