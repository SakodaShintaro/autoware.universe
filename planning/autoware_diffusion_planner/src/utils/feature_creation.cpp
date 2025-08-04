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

#include "autoware/diffusion_planner/utils/feature_creation.hpp"

#include "autoware/diffusion_planner/conversion/agent.hpp"
#include "autoware/diffusion_planner/conversion/ego.hpp"
#include "autoware/diffusion_planner/dimensions.hpp"

namespace autoware::diffusion_planner::utils
{

std::vector<float> create_ego_agent_past(
  const std::deque<Odometry> & ego_history, const Eigen::Matrix4f & map_to_ego_transform)
{
  std::vector<float> ego_past_data;
  ego_past_data.reserve(ego_history.size() * 4);  // x, y, cos_yaw, sin_yaw per timestep

  for (const auto & ego_state : ego_history) {
    // Convert ego pose to matrix
    const Eigen::Matrix4f ego_pose_map = pose_to_matrix4f(ego_state.pose.pose);

    // Transform to ego frame
    const Eigen::Matrix4f ego_pose_ego = map_to_ego_transform * ego_pose_map;

    // Extract position
    const float x = ego_pose_ego(0, 3);
    const float y = ego_pose_ego(1, 3);

    // Extract heading as cos/sin
    const auto [cos_yaw, sin_yaw] = rotation_matrix_to_cos_sin(ego_pose_ego.block<3, 3>(0, 0));

    ego_past_data.push_back(x);
    ego_past_data.push_back(y);
    ego_past_data.push_back(cos_yaw);
    ego_past_data.push_back(sin_yaw);
  }

  return ego_past_data;
}

std::vector<float> create_ego_current_state(
  const Odometry & ego_kinematic_state, const AccelWithCovarianceStamped & ego_acceleration,
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info)
{
  EgoState ego_state(
    ego_kinematic_state, ego_acceleration, static_cast<float>(vehicle_info.wheel_base_m));
  return ego_state.as_array();
}

std::vector<float> create_neighbor_agents_past(
  const TrackedObjects & objects, const Eigen::Matrix4f & map_to_ego_transform)
{
  AgentData agent_data(objects, NEIGHBOR_SHAPE[1], NEIGHBOR_SHAPE[2]);
  agent_data.apply_transform(map_to_ego_transform);
  return agent_data.as_vector();
}

std::vector<float> create_static_objects()
{
  // TODO(Daniel): add static objects
  return create_float_data(
    std::vector<int64_t>(STATIC_OBJECTS_SHAPE.begin(), STATIC_OBJECTS_SHAPE.end()), 0.0f);
}

std::vector<float> create_lanes_feature(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  const preprocess::ColLaneIDMaps & col_id_mapping,
  const std::map<lanelet::Id, preprocess::TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y)
{
  std::tuple<Eigen::MatrixXf, preprocess::ColLaneIDMaps> matrix_mapping_tuple =
    preprocess::transform_and_select_rows(
      map_lane_segments_matrix, map_to_ego_transform, col_id_mapping, traffic_light_id_map,
      lanelet_map_ptr, center_x, center_y, LANES_SHAPE[1]);
  const Eigen::MatrixXf & ego_centric_lane_segments = std::get<0>(matrix_mapping_tuple);
  return preprocess::extract_lane_tensor_data(ego_centric_lane_segments);
}

std::vector<float> create_lanes_speed_limit(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  const preprocess::ColLaneIDMaps & col_id_mapping,
  const std::map<lanelet::Id, preprocess::TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, float center_x, float center_y)
{
  std::tuple<Eigen::MatrixXf, preprocess::ColLaneIDMaps> matrix_mapping_tuple =
    preprocess::transform_and_select_rows(
      map_lane_segments_matrix, map_to_ego_transform, col_id_mapping, traffic_light_id_map,
      lanelet_map_ptr, center_x, center_y, LANES_SHAPE[1]);
  const Eigen::MatrixXf & ego_centric_lane_segments = std::get<0>(matrix_mapping_tuple);
  return preprocess::extract_lane_speed_tensor_data(ego_centric_lane_segments);
}

std::pair<std::vector<float>, std::vector<float>> create_route_lanes_feature(
  const Eigen::MatrixXf & map_lane_segments_matrix, const Eigen::Matrix4f & map_to_ego_transform,
  const preprocess::ColLaneIDMaps & col_id_mapping,
  const std::map<lanelet::Id, preprocess::TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler,
  const geometry_msgs::msg::Pose & current_pose)
{
  constexpr double backward_path_length{200.0};  // constants::BACKWARD_PATH_LENGTH_M
  constexpr double forward_path_length{300.0};   // constants::FORWARD_PATH_LENGTH_M
  lanelet::ConstLanelet current_preferred_lane;

  if (
    !route_handler->isHandlerReady() ||
    !route_handler->getClosestPreferredLaneletWithinRoute(current_pose, &current_preferred_lane)) {
    return std::make_pair(std::vector<float>(), std::vector<float>());
  }

  auto current_lanes = route_handler->getLaneletSequence(
    current_preferred_lane, backward_path_length, forward_path_length);

  return preprocess::get_route_segments(
    map_lane_segments_matrix, map_to_ego_transform, col_id_mapping, traffic_light_id_map,
    lanelet_map_ptr, current_lanes);
}

std::vector<float> create_goal_pose_feature(
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler,
  const Eigen::Matrix4f & map_to_ego_transform)
{
  const auto & goal_pose = route_handler->getGoalPose();

  // Convert goal pose to 4x4 transformation matrix
  const Eigen::Matrix4f goal_pose_map_4x4 = pose_to_matrix4f(goal_pose);

  // Transform to ego frame
  const Eigen::Matrix4f goal_pose_ego_4x4 = map_to_ego_transform * goal_pose_map_4x4;

  // Extract relative position
  const float x = goal_pose_ego_4x4(0, 3);
  const float y = goal_pose_ego_4x4(1, 3);

  // Extract heading as cos/sin from rotation matrix
  const auto [cos_yaw, sin_yaw] = rotation_matrix_to_cos_sin(goal_pose_ego_4x4.block<3, 3>(0, 0));

  return std::vector<float>{x, y, cos_yaw, sin_yaw};
}

std::vector<float> create_ego_shape_feature(
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info)
{
  const float wheel_base = static_cast<float>(vehicle_info.wheel_base_m);
  const float vehicle_length = static_cast<float>(
    vehicle_info.front_overhang_m + vehicle_info.wheel_base_m + vehicle_info.rear_overhang_m);
  const float vehicle_width = static_cast<float>(
    vehicle_info.left_overhang_m + vehicle_info.wheel_tread_m + vehicle_info.right_overhang_m);
  return std::vector<float>{wheel_base, vehicle_length, vehicle_width};
}

}  // namespace autoware::diffusion_planner::utils