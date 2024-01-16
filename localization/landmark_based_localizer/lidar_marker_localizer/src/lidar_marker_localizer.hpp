// Copyright 2023 Autoware Foundation
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

#ifndef LIDAR_MARKER_LOCALIZER_HPP_
#define LIDAR_MARKER_LOCALIZER_HPP_

#include "diagnostics/diagnostics_module.hpp"
#include "localization_util/smart_pose_buffer.hpp"

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#endif
#include <landmark_manager/landmark_manager.hpp>

#include <geometry_msgs/msg/pose_array.hpp>

#include <pcl/point_types.h>

class LidarMarkerLocalizer : public rclcpp::Node
{
  using HADMapBin = autoware_auto_mapping_msgs::msg::HADMapBin;
  using MarkerArray = visualization_msgs::msg::MarkerArray;
  using Pose = geometry_msgs::msg::Pose;
  using PoseStamped = geometry_msgs::msg::PoseStamped;
  using PoseWithCovarianceStamped = geometry_msgs::msg::PoseWithCovarianceStamped;
  using PoseArray = geometry_msgs::msg::PoseArray;
  using TransformStamped = geometry_msgs::msg::TransformStamped;
  using Vector3 = geometry_msgs::msg::Vector3;
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using SetBool = std_srvs::srv::SetBool;
  using DiagnosticStatus = diagnostic_msgs::msg::DiagnosticStatus;

  struct Param
  {
    double resolution;
    std::vector<int64_t> intensity_pattern;
    int64_t match_intensity_difference_threshold;
    int64_t positive_match_num_threshold;
    int64_t negative_match_num_threshold;
    int64_t vote_threshold_for_detect_marker;

    double self_pose_timeout_sec;
    double self_pose_distance_tolerance_m;

    double limit_distance_from_self_pose_to_nearest_marker;
    double limit_distance_from_self_pose_to_marker;
    std::array<double, 36> base_covariance;
  };

public:
  LidarMarkerLocalizer();

private:
  void self_pose_callback(const PoseWithCovarianceStamped::ConstSharedPtr & self_pose_msg_ptr);
  void points_callback(const PointCloud2::ConstSharedPtr & points_msg_ptr);
  void map_bin_callback(const HADMapBin::ConstSharedPtr & msg);
  void service_trigger_node(
    const SetBool::Request::SharedPtr req, SetBool::Response::SharedPtr res);

  void initilize_diagnostics();
  void main_process(const PointCloud2::ConstSharedPtr & points_msg_ptr);
  std::vector<landmark_manager::Landmark> detect_landmarks(
    const PointCloud2::ConstSharedPtr & points_msg_ptr);
  landmark_manager::Landmark get_nearest_landmark(
    const geometry_msgs::msg::Pose & self_pose,
    const std::vector<landmark_manager::Landmark> & landmarks) const;
  std::array<double, 36> rotate_covariance(
    const std::array<double, 36> & src_covariance, const Eigen::Matrix3d & rotation) const;

  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  rclcpp::Subscription<PointCloud2>::SharedPtr sub_points_;
  rclcpp::Subscription<PoseWithCovarianceStamped>::SharedPtr sub_self_pose_;
  rclcpp::Subscription<HADMapBin>::SharedPtr sub_map_bin_;

  rclcpp::Publisher<PoseWithCovarianceStamped>::SharedPtr
    pub_base_link_pose_with_covariance_on_map_;
  rclcpp::Service<SetBool>::SharedPtr service_trigger_node_;
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_marker_mapped_;
  rclcpp::Publisher<PoseArray>::SharedPtr pub_marker_detected_;

  std::shared_ptr<DiagnosticsModule> diagnostics_module_;

  Param param_;
  bool is_activated_;
  bool is_detected_marker_;
  bool is_exist_marker_within_self_pose_;
  std::unique_ptr<SmartPoseBuffer> ekf_pose_buffer_;

  landmark_manager::LandmarkManager landmark_manager_;
};

#endif  // LIDAR_MARKER_LOCALIZER_HPP_