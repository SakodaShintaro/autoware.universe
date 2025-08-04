// Copyright 2025 TIER IV, Inc.
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

#ifndef AUTOWARE__DIFFUSION_PLANNER__UTILS__FEATURE_CREATION_HPP_
#define AUTOWARE__DIFFUSION_PLANNER__UTILS__FEATURE_CREATION_HPP_

#include "autoware/diffusion_planner/preprocessing/lane_segments.hpp"
#include "autoware/diffusion_planner/preprocessing/traffic_signals.hpp"
#include "autoware/diffusion_planner/utils/utils.hpp"

#include <Eigen/Dense>
#include <autoware/route_handler/route_handler.hpp>
#include <autoware_vehicle_info_utils/vehicle_info.hpp>

#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <lanelet2_core/LaneletMap.h>

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace autoware::diffusion_planner::utils
{

using autoware_perception_msgs::msg::TrackedObjects;
using geometry_msgs::msg::AccelWithCovarianceStamped;
using nav_msgs::msg::Odometry;

/**
 * @brief Create ego agent past feature from ego history
 */
std::vector<float> create_ego_agent_past(
  const std::deque<Odometry> & ego_history, const Eigen::Matrix4f & map_to_ego_transform);

/**
 * @brief Create ego current state feature
 */
std::vector<float> create_ego_current_state(
  const Odometry & ego_kinematic_state, const AccelWithCovarianceStamped & ego_acceleration,
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info);

/**
 * @brief Create neighbor agents past feature
 */
std::vector<float> create_neighbor_agents_past(
  const TrackedObjects & objects, const Eigen::Matrix4f & map_to_ego_transform);

/**
 * @brief Create static objects feature (placeholder for now)
 */
std::vector<float> create_static_objects();

/**
 * @brief Create lanes feature from map data
 */
std::vector<float> create_lanes_feature(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  const preprocess::ColLaneIDMaps & col_id_mapping,
  const std::map<lanelet::Id, preprocess::TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y);

/**
 * @brief Create lanes speed limit feature
 */
std::vector<float> create_lanes_speed_limit(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  const preprocess::ColLaneIDMaps & col_id_mapping,
  const std::map<lanelet::Id, preprocess::TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y);

/**
 * @brief Create route lanes feature
 */
std::pair<std::vector<float>, std::vector<float>> create_route_lanes_feature(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  const preprocess::ColLaneIDMaps & col_id_mapping,
  const std::map<lanelet::Id, preprocess::TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler,
  const geometry_msgs::msg::Pose & current_pose);

/**
 * @brief Create goal pose feature in ego frame
 */
std::vector<float> create_goal_pose_feature(
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler,
  const Eigen::Matrix4f & map_to_ego_transform);

/**
 * @brief Create ego shape feature
 */
std::vector<float> create_ego_shape_feature(
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info);

}  // namespace autoware::diffusion_planner::utils

#endif  // AUTOWARE__DIFFUSION_PLANNER__UTILS__FEATURE_CREATION_HPP_