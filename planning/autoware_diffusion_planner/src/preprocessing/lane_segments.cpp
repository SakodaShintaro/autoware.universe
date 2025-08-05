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

#include "autoware/diffusion_planner/preprocessing/lane_segments.hpp"

#include "autoware/diffusion_planner/dimensions.hpp"

#include <autoware_lanelet2_extension/regulatory_elements/road_marking.hpp>  // for lanelet::autoware::RoadMarking
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>

#include <Eigen/src/Core/Matrix.h>
#include <lanelet2_core/Forward.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace autoware::diffusion_planner::preprocess
{
std::vector<ColWithDistance> compute_distances(
  const std::map<int64_t, LaneSegment> & lane_segments, const Eigen::Matrix4f & transform_matrix,
  const float center_x, const float center_y, const float mask_range)
{
  auto compute_squared_distance = [](float x, float y, const Eigen::Matrix4f & transform_matrix) {
    Eigen::Vector4f p(x, y, 0.0f, 1.0f);
    Eigen::Vector4f p_transformed = transform_matrix * p;
    return p_transformed.head<2>().squaredNorm();
  };

  std::vector<ColWithDistance> distances;
  distances.reserve(lane_segments.size());
  for (const auto & [id, segment] : lane_segments) {
    // Directly access input matrix as raw memory
    float x = segment.meanX();
    float y = segment.meanY();
    bool inside =
      (x > center_x - mask_range * 1.1 && x < center_x + mask_range * 1.1 &&
       y > center_y - mask_range * 1.1 && y < center_y + mask_range * 1.1);

    const auto distance_squared = [&]() {
      float x_first = segment.polyline.waypoints().front().x();
      float y_first = segment.polyline.waypoints().front().y();
      float x_last = segment.polyline.waypoints().back().x();
      float y_last = segment.polyline.waypoints().back().y();
      float distance_squared_first = compute_squared_distance(x_first, y_first, transform_matrix);
      float distance_squared_last = compute_squared_distance(x_last, y_last, transform_matrix);
      return std::min(distance_squared_last, distance_squared_first);
    }();

    distances.push_back({id, distance_squared, inside});
  }

  return distances;
}

void transform_selected_rows(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, int64_t num_segments,
  int64_t row_idx, bool do_translation)
{
  Eigen::MatrixXf xy_block(4, num_segments * POINTS_PER_SEGMENT);
  xy_block.setZero();
  xy_block.block(0, 0, 2, num_segments * POINTS_PER_SEGMENT) =
    output_matrix.block(row_idx, 0, 2, num_segments * POINTS_PER_SEGMENT);

  xy_block.row(3) = do_translation ? Eigen::MatrixXf::Ones(1, num_segments * POINTS_PER_SEGMENT)
                                   : Eigen::MatrixXf::Zero(1, num_segments * POINTS_PER_SEGMENT);

  Eigen::MatrixXf transformed_block = transform_matrix * xy_block;
  output_matrix.block(row_idx, 0, 2, num_segments * POINTS_PER_SEGMENT) =
    transformed_block.block(0, 0, 2, num_segments * POINTS_PER_SEGMENT);
}

int64_t get_traffic_light_status_int(
  const lanelet::Id & lane_id,
  const std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr)
{
  const auto assigned_lanelet = lanelet_map_ptr->laneletLayer.get(lane_id);
  const auto tl_reg_elems = assigned_lanelet.regulatoryElementsAs<const lanelet::TrafficLight>();
  if (tl_reg_elems.empty()) {
    return TRAFFIC_LIGHT_NO_TRAFFIC_LIGHT;
  }
  const auto & tl_reg_elem = tl_reg_elems.front();
  const auto traffic_light_stamped_info_itr = traffic_light_id_map.find(tl_reg_elem->id());
  if (traffic_light_stamped_info_itr == traffic_light_id_map.end()) {
    return TRAFFIC_LIGHT_WHITE;
  }

  const auto & signal = traffic_light_stamped_info_itr->second.signal;

  using namespace autoware::traffic_light_utils;
  using namespace autoware_perception_msgs::msg;

  if (hasTrafficLightCircleColor(signal.elements, TrafficLightElement::GREEN)) {
    return TRAFFIC_LIGHT_GREEN;
  } else if (hasTrafficLightCircleColor(signal.elements, TrafficLightElement::AMBER)) {
    return TRAFFIC_LIGHT_YELLOW;
  } else if (hasTrafficLightCircleColor(signal.elements, TrafficLightElement::RED)) {
    return TRAFFIC_LIGHT_RED;
  } else {
    return TRAFFIC_LIGHT_WHITE;
  }
}

void apply_transforms(
  const Eigen::Matrix4f & transform_matrix, Eigen::MatrixXf & output_matrix, int64_t num_segments)
{
  // transform the x and y coordinates
  transform_selected_rows(transform_matrix, output_matrix, num_segments, X);
  // the dx and dy coordinates do not require translation
  transform_selected_rows(transform_matrix, output_matrix, num_segments, dX, false);
  transform_selected_rows(transform_matrix, output_matrix, num_segments, LB_X);
  transform_selected_rows(transform_matrix, output_matrix, num_segments, RB_X);

  // subtract center from boundaries
  output_matrix.row(LB_X) = output_matrix.row(LB_X) - output_matrix.row(X);
  output_matrix.row(LB_Y) = output_matrix.row(LB_Y) - output_matrix.row(Y);
  output_matrix.row(RB_X) = output_matrix.row(RB_X) - output_matrix.row(X);
  output_matrix.row(RB_Y) = output_matrix.row(RB_Y) - output_matrix.row(Y);
}

Eigen::MatrixXf transform_and_select_rows(
  const std::map<int64_t, LaneSegment> & lane_segments, const Eigen::Matrix4f & transform_matrix,
  const std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr, const float center_x,
  const float center_y, const int64_t m)
{
  const float mask_range = 100.0f;

  // Compute distances and sort by distance
  auto compute_squared_distance = [](float x, float y, const Eigen::Matrix4f & transform_matrix) {
    Eigen::Vector4f p(x, y, 0.0f, 1.0f);
    Eigen::Vector4f p_transformed = transform_matrix * p;
    return p_transformed.head<2>().squaredNorm();
  };

  std::vector<std::pair<float, LaneSegment>> distance_and_segment_list;
  for (const auto & [id, segment] : lane_segments) {
    float x = segment.meanX();
    float y = segment.meanY();
    bool inside =
      (x > center_x - mask_range * 1.1 && x < center_x + mask_range * 1.1 &&
       y > center_y - mask_range * 1.1 && y < center_y + mask_range * 1.1);

    if (!inside) {
      continue;
    }

    const auto distance_squared = [&]() {
      float x_first = segment.polyline.waypoints().front().x();
      float y_first = segment.polyline.waypoints().front().y();
      float x_last = segment.polyline.waypoints().back().x();
      float y_last = segment.polyline.waypoints().back().y();
      float distance_squared_first = compute_squared_distance(x_first, y_first, transform_matrix);
      float distance_squared_last = compute_squared_distance(x_last, y_last, transform_matrix);
      return std::min(distance_squared_last, distance_squared_first);
    }();

    distance_and_segment_list.push_back({distance_squared, segment});
  }

  // Sort indices by distance
  std::sort(
    distance_and_segment_list.begin(), distance_and_segment_list.end(),
    [](const auto & a, const auto & b) { return a.first < b.first; });

  // Apply transformation to selected rows
  const int64_t n_total_segments = static_cast<int64_t>(lane_segments.size());
  const int64_t num_segments = std::min(m, n_total_segments);

  Eigen::MatrixXf output_matrix(FULL_MATRIX_ROWS, m * POINTS_PER_SEGMENT);
  output_matrix.setZero();

  int64_t added_segments = 0;
  ColLaneIDMaps new_col_id_mapping;
  for (const auto & [distance, segment] : distance_and_segment_list) {
    const auto lane_id = segment.id;

    // get POINTS_PER_SEGMENT rows corresponding to a single segment
    output_matrix.block<FULL_MATRIX_ROWS, POINTS_PER_SEGMENT>(
      0, added_segments * POINTS_PER_SEGMENT) = process_segment_to_matrix(segment);

    const int64_t traffic_light_status =
      get_traffic_light_status_int(lane_id, traffic_light_id_map, lanelet_map_ptr);

    for (int64_t i = 0; i < POINTS_PER_SEGMENT; i++) {
      output_matrix(TRAFFIC_LIGHT + traffic_light_status, added_segments * POINTS_PER_SEGMENT + i) =
        1.0;
    }

    ++added_segments;
    if (added_segments >= num_segments) {
      break;
    }
  }

  // Apply transforms
  // transform the x and y coordinates
  transform_selected_rows(transform_matrix, output_matrix, added_segments, X);
  // the dx and dy coordinates do not require translation
  transform_selected_rows(transform_matrix, output_matrix, added_segments, dX, false);
  transform_selected_rows(transform_matrix, output_matrix, added_segments, LB_X);
  transform_selected_rows(transform_matrix, output_matrix, added_segments, RB_X);

  // subtract center from boundaries
  output_matrix.row(LB_X) = output_matrix.row(LB_X) - output_matrix.row(X);
  output_matrix.row(LB_Y) = output_matrix.row(LB_Y) - output_matrix.row(Y);
  output_matrix.row(RB_X) = output_matrix.row(RB_X) - output_matrix.row(X);
  output_matrix.row(RB_Y) = output_matrix.row(RB_Y) - output_matrix.row(Y);

  return output_matrix;
}

Eigen::MatrixXf process_segment_to_matrix(const LaneSegment & segment)
{
  if (
    segment.polyline.is_empty() || segment.left_boundaries.empty() ||
    segment.right_boundaries.empty()) {
    return {};
  }
  const auto & centerlines = segment.polyline.waypoints();
  const auto & left_boundaries = segment.left_boundaries.front().waypoints();
  const auto & right_boundaries = segment.right_boundaries.front().waypoints();

  if (
    centerlines.size() != POINTS_PER_SEGMENT || left_boundaries.size() != POINTS_PER_SEGMENT ||
    right_boundaries.size() != POINTS_PER_SEGMENT) {
    throw std::runtime_error(
      "Segment data size mismatch: centerlines, left boundaries, and right boundaries must have "
      "POINTS_PER_SEGMENT points");
  }

  Eigen::MatrixXf segment_data(POINTS_PER_SEGMENT, FULL_MATRIX_ROWS);
  segment_data.setZero();

  // Build each row
  for (int64_t i = 0; i < POINTS_PER_SEGMENT; ++i) {
    segment_data(i, X) = centerlines[i].x();
    segment_data(i, Y) = centerlines[i].y();
    segment_data(i, dX) =
      i < POINTS_PER_SEGMENT - 1 ? centerlines[i + 1].x() - centerlines[i].x() : 0.0f;
    segment_data(i, dY) =
      i < POINTS_PER_SEGMENT - 1 ? centerlines[i + 1].y() - centerlines[i].y() : 0.0f;
    segment_data(i, LB_X) = left_boundaries[i].x();
    segment_data(i, LB_Y) = left_boundaries[i].y();
    segment_data(i, RB_X) = right_boundaries[i].x();
    segment_data(i, RB_Y) = right_boundaries[i].y();
    segment_data(i, SPEED_LIMIT) = segment.speed_limit_mps.value_or(0.0f);
    segment_data(i, LANE_ID) = static_cast<float>(segment.id);
  }

  return segment_data;
}

std::vector<float> extract_lane_tensor_data(const Eigen::MatrixXf & lane_segments_matrix)
{
  const auto total_lane_points = LANES_SHAPE[1] * POINTS_PER_SEGMENT;
  Eigen::MatrixXf lane_matrix(SEGMENT_POINT_DIM, total_lane_points);
  lane_matrix.block(0, 0, SEGMENT_POINT_DIM, total_lane_points) =
    lane_segments_matrix.block(0, 0, SEGMENT_POINT_DIM, total_lane_points);
  return {lane_matrix.data(), lane_matrix.data() + lane_matrix.size()};
}

std::vector<float> extract_lane_speed_tensor_data(const Eigen::MatrixXf & lane_segments_matrix)
{
  const auto total_lane_points = LANES_SPEED_LIMIT_SHAPE[1];
  std::vector<float> lane_speed_vector(total_lane_points);
  for (int64_t i = 0; i < total_lane_points; ++i) {
    lane_speed_vector[i] = lane_segments_matrix(SPEED_LIMIT, i * POINTS_PER_SEGMENT);
  }
  return lane_speed_vector;
}

std::pair<std::vector<float>, std::vector<float>> get_route_segments(
  const std::map<int64_t, LaneSegment> & lane_segments_map,
  const Eigen::Matrix4f & transform_matrix,
  const std::map<lanelet::Id, TrafficSignalStamped> & traffic_light_id_map,
  const std::shared_ptr<lanelet::LaneletMap> & lanelet_map_ptr,
  const lanelet::ConstLanelets & current_lanes)
{
  const auto total_route_points = ROUTE_LANES_SHAPE[1] * POINTS_PER_SEGMENT;
  Eigen::MatrixXf full_route_segment_matrix(SEGMENT_POINT_DIM, total_route_points);
  full_route_segment_matrix.setZero();
  int64_t added_route_segments = 0;

  std::vector<float> speed_limit_vector(ROUTE_LANES_SHAPE[1]);

  // Add traffic light one-hot encoding to the route segments
  for (const auto & route_segment : current_lanes) {
    if (added_route_segments >= ROUTE_LANES_SHAPE[1]) {
      break;
    }
    if (lane_segments_map.count(route_segment.id()) == 0) {
      continue;  // Skip if the segment is not in the map
    }

    const LaneSegment & lane_segment = lane_segments_map.at(route_segment.id());
    const auto map_lane_segments_matrix = process_segment_to_matrix(lane_segment);

    full_route_segment_matrix.block(
      0, added_route_segments * POINTS_PER_SEGMENT, SEGMENT_POINT_DIM, POINTS_PER_SEGMENT) =
      map_lane_segments_matrix.block(0, 0, SEGMENT_POINT_DIM, POINTS_PER_SEGMENT);

    const int64_t traffic_light_status =
      get_traffic_light_status_int(route_segment.id(), traffic_light_id_map, lanelet_map_ptr);

    for (int64_t i = 0; i < POINTS_PER_SEGMENT; i++) {
      full_route_segment_matrix(
        added_route_segments * POINTS_PER_SEGMENT + i, TRAFFIC_LIGHT + traffic_light_status) = 1.0f;
    }

    speed_limit_vector[added_route_segments] = map_lane_segments_matrix(SPEED_LIMIT, 0);
    ++added_route_segments;
  }
  // Transform the route segments.
  apply_transforms(transform_matrix, full_route_segment_matrix, added_route_segments);
  return {
    {full_route_segment_matrix.data(),
     full_route_segment_matrix.data() + full_route_segment_matrix.size()},
    speed_limit_vector};
}

}  // namespace autoware::diffusion_planner::preprocess
