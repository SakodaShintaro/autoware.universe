#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Message types
#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

// Autoware diffusion planner headers
#include <autoware/diffusion_planner/conversion/agent.hpp>
#include <autoware/diffusion_planner/conversion/ego.hpp>
#include <autoware/diffusion_planner/conversion/lanelet.hpp>
#include <autoware/diffusion_planner/dimensions.hpp>
#include <autoware/diffusion_planner/preprocessing/lane_segments.hpp>
#include <autoware/diffusion_planner/preprocessing/traffic_signals.hpp>
#include <autoware/diffusion_planner/utils/utils.hpp>

// Additional includes for Lanelet2 and test utilities
#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_test_utils/autoware_test_utils.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

// NumPy save/load utility
#include "numpy.hpp"

namespace fs = std::filesystem;

// Message type aliases
using TrackedObjects = autoware_perception_msgs::msg::TrackedObjects;
using TrafficLightGroupArray = autoware_perception_msgs::msg::TrafficLightGroupArray;
using LaneletRoute = autoware_planning_msgs::msg::LaneletRoute;
using TurnIndicatorsReport = autoware_vehicle_msgs::msg::TurnIndicatorsReport;
using AccelWithCovarianceStamped = geometry_msgs::msg::AccelWithCovarianceStamped;
using Odometry = nav_msgs::msg::Odometry;
using LaneletMapBin = autoware_map_msgs::msg::LaneletMapBin;

struct ParseRosbagConfig
{
  std::string rosbag_path;
  std::string vector_map_path;
  std::string save_dir;
  int64_t step = 1;
  int64_t limit = -1;
  int64_t min_frames = 1700;
  bool search_nearest_route = true;
};

struct MessageCollections
{
  std::deque<Odometry> kinematic_msgs;
  std::deque<AccelWithCovarianceStamped> acceleration_msgs;
  std::deque<TrackedObjects> tracking_msgs;
  std::deque<TrafficLightGroupArray> traffic_msgs;
  std::deque<LaneletRoute> route_msgs;
  std::deque<TurnIndicatorsReport> turn_indicator_msgs;
};

struct FrameData
{
  rclcpp::Time timestamp;
  LaneletRoute route;
  TrackedObjects tracked_objects;
  Odometry kinematic_state;
  AccelWithCovarianceStamped acceleration;
  TrafficLightGroupArray traffic_signals;
  TurnIndicatorsReport turn_indicator;
};

class RosbagParser
{
public:
  RosbagParser(const ParseRosbagConfig & config) : config_(config)
  {
    // Initialize target topics
    target_topics_ = {
      "/localization/kinematic_state",
      "/localization/acceleration",
      "/perception/object_recognition/tracking/objects",
      "/perception/traffic_light_recognition/traffic_signals",
      "/planning/mission_planning/route",
      "/vehicle/status/turn_indicators_status"};

    // Initialize rosbag reader
    storage_options_.uri = config_.rosbag_path;
    storage_options_.storage_id = "sqlite3";
    converter_options_.input_serialization_format = "cdr";
    converter_options_.output_serialization_format = "cdr";

    // Initialize Lanelet2 map
    initializeLaneletMap();
  }

  void loadLaneletMap() { initializeLaneletMap(); }

  MessageCollections readRosbagMessages()
  {
    std::cout << "Starting rosbag parsing..." << std::endl;
    std::cout << "Rosbag path: " << config_.rosbag_path << std::endl;
    std::cout << "Vector map path: " << config_.vector_map_path << std::endl;

    // Open rosbag
    reader_.open(storage_options_, converter_options_);

    // Set topic filter
    rosbag2_storage::StorageFilter storage_filter;
    storage_filter.topics = target_topics_;
    reader_.set_filter(storage_filter);

    std::map<std::string, int64_t> topic_counts;
    int64_t total_messages = 0;
    int64_t processed_messages = 0;

    // First pass: count messages
    std::cout << "Counting messages..." << std::endl;
    while (reader_.has_next()) {
      rosbag2_storage::SerializedBagMessageSharedPtr bag_message = reader_.read_next();
      topic_counts[bag_message->topic_name]++;
      total_messages++;

      if (config_.limit > 0 && total_messages >= config_.limit) {
        break;
      }
    }

    // Print statistics
    std::cout << "\nMessage counts per topic:" << std::endl;
    for (const std::pair<const std::string, int64_t> & topic_count : topic_counts) {
      std::cout << "  " << topic_count.first << ": " << topic_count.second << " messages"
                << std::endl;
    }

    // Reopen rosbag for processing
    reader_.close();
    reader_.open(storage_options_, converter_options_);
    reader_.set_filter(storage_filter);

    // Process messages
    std::cout << "\nProcessing messages..." << std::endl;

    // Storage for collected messages by type (using deque for efficient operations)
    MessageCollections collections;

    while (reader_.has_next() && (config_.limit < 0 || processed_messages < config_.limit)) {
      rosbag2_storage::SerializedBagMessageSharedPtr bag_message = reader_.read_next();

      // Deserialize and store messages by type
      if (bag_message->topic_name == "/localization/kinematic_state") {
        collections.kinematic_msgs.push_back(deserializeMessage<Odometry>(bag_message));
      } else if (bag_message->topic_name == "/localization/acceleration") {
        collections.acceleration_msgs.push_back(
          deserializeMessage<AccelWithCovarianceStamped>(bag_message));
      } else if (bag_message->topic_name == "/perception/object_recognition/tracking/objects") {
        collections.tracking_msgs.push_back(deserializeMessage<TrackedObjects>(bag_message));
      } else if (
        bag_message->topic_name == "/perception/traffic_light_recognition/traffic_signals") {
        collections.traffic_msgs.push_back(deserializeMessage<TrafficLightGroupArray>(bag_message));
      } else if (bag_message->topic_name == "/planning/mission_planning/route") {
        collections.route_msgs.push_back(deserializeMessage<LaneletRoute>(bag_message));
      } else if (bag_message->topic_name == "/vehicle/status/turn_indicators_status") {
        collections.turn_indicator_msgs.push_back(
          deserializeMessage<TurnIndicatorsReport>(bag_message));
      }

      processed_messages++;

      if (processed_messages % 1000 == 0) {
        std::cout << "Processed " << processed_messages << "/" << total_messages << " messages..."
                  << std::endl;
      }
    }

    reader_.close();

    std::cout << "\nCollected messages:" << std::endl;
    std::cout << "  Kinematic: " << collections.kinematic_msgs.size() << std::endl;
    std::cout << "  Acceleration: " << collections.acceleration_msgs.size() << std::endl;
    std::cout << "  Tracking: " << collections.tracking_msgs.size() << std::endl;
    std::cout << "  Traffic: " << collections.traffic_msgs.size() << std::endl;
    std::cout << "  Route: " << collections.route_msgs.size() << std::endl;
    std::cout << "  Turn indicator: " << collections.turn_indicator_msgs.size() << std::endl;

    return collections;
  }

  std::vector<FrameData> createDataList(MessageCollections & collections)
  {
    std::vector<FrameData> data_list;

    const size_t n = collections.tracking_msgs.size();

    // Use tracking messages as base timing (like Python version)
    for (size_t i = 0; i < n; ++i) {
      const TrackedObjects & tracking = collections.tracking_msgs[i];
      const rclcpp::Time timestamp = rclcpp::Time(tracking.header.stamp);

      bool ok = true;
      FrameData frame_data;
      frame_data.timestamp = timestamp;
      frame_data.tracked_objects = tracking;

      // Find nearest messages with pop like Python version
      if (!collections.kinematic_msgs.empty()) {
        std::optional<Odometry> result =
          getNearestMessageWithPop(collections.kinematic_msgs, timestamp);
        if (!result) {
          std::cout << "Cannot find kinematic_state msg at frame " << i << std::endl;
          ok = false;
        } else {
          frame_data.kinematic_state = *result;
        }
      }

      if (ok && !collections.acceleration_msgs.empty()) {
        std::optional<AccelWithCovarianceStamped> result =
          getNearestMessageWithPop(collections.acceleration_msgs, timestamp);
        if (!result) {
          std::cout << "Cannot find acceleration msg at frame " << i << std::endl;
          ok = false;
        } else {
          frame_data.acceleration = *result;
        }
      }

      if (ok && !collections.traffic_msgs.empty()) {
        std::optional<TrafficLightGroupArray> result =
          getNearestMessageWithPop(collections.traffic_msgs, timestamp);
        if (result) {
          frame_data.traffic_signals = *result;
        }
      }

      if (ok && !collections.route_msgs.empty()) {
        std::optional<LaneletRoute> result =
          getNearestMessageWithPop(collections.route_msgs, timestamp);
        if (result) {
          frame_data.route = *result;
        }
      }

      if (ok && !collections.turn_indicator_msgs.empty()) {
        std::optional<TurnIndicatorsReport> result =
          getNearestMessageWithPop(collections.turn_indicator_msgs, timestamp);
        if (result) {
          frame_data.turn_indicator = *result;
        }
      }

      if (!ok) {
        if (data_list.empty()) {
          std::cout << "Skip frame " << i << "/" << n << " (at beginning)" << std::endl;
          continue;
        } else {
          std::cout << "Finish at frame " << i << "/" << n << " (missing msg in middle)"
                    << std::endl;
          break;
        }
      }

      data_list.push_back(frame_data);
    }

    return data_list;
  }

  void processDataList(const std::vector<FrameData> & data_list)
  {
    std::cout << "\nProcessing data list..." << std::endl;

    const int64_t n = static_cast<int64_t>(data_list.size());
    // Use dimensions from dimensions.hpp
    constexpr std::array<int64_t, 3> ego_history_shape =
      autoware::diffusion_planner::EGO_HISTORY_SHAPE;
    const int64_t past_time_steps = ego_history_shape[1];                     // Shape is {1, 21, 4}
    const int64_t future_time_steps = autoware::diffusion_planner::OUTPUT_T;  // 80

    if (n <= past_time_steps + future_time_steps) {
      std::cout << "Not enough frames for processing. Need at least "
                << (past_time_steps + future_time_steps) << " frames, got " << n << std::endl;
      return;
    }

    int64_t total_frames = (n - past_time_steps - future_time_steps);
    if (config_.limit > 0) {
      total_frames = std::min(total_frames, config_.limit);
    }

    std::cout << "Processing " << total_frames << " frames..." << std::endl;

    for (int64_t i = past_time_steps; i < past_time_steps + total_frames; i += config_.step) {
      // Generate token (sequence_id + frame_id)
      std::ostringstream oss;
      oss << std::setfill('0') << std::setw(8) << 0 << std::setw(8) << i;
      std::string token = oss.str();

      // Create NPY files using FrameData from data_list
      createNPYFiles(token, data_list[i]);

      if (i % 100 == 0) {
        std::cout << "Processed frame " << i << "/" << total_frames << std::endl;
      }
    }

    std::cout << "Data list processing completed!" << std::endl;
  }

  template <typename T>
  T deserializeMessage(const rosbag2_storage::SerializedBagMessageSharedPtr & serialized_message)
  {
    T msg;
    rclcpp::Serialization<T> serialization;
    rclcpp::SerializedMessage extracted_serialized_msg(*serialized_message->serialized_data);
    serialization.deserialize_message(&extracted_serialized_msg, &msg);
    return msg;
  }

private:
  void initializeLaneletMap()
  {
    std::cout << "Loading Lanelet2 map from: " << config_.vector_map_path << std::endl;

    // Load map using test utilities (similar to lanelet_integration_test.cpp)
    if (config_.vector_map_path.find(".osm") != std::string::npos) {
      // OSM file
      map_bin_msg_ = autoware::test_utils::make_map_bin_msg(config_.vector_map_path, 1.0);
    } else {
      std::cerr << "Unsupported map format. Expected .osm file." << std::endl;
      return;
    }

    // Convert HADMapBin to lanelet map
    lanelet_map_ptr_ = std::make_shared<lanelet::LaneletMap>();
    lanelet::utils::conversion::fromBinMsg(
      map_bin_msg_, lanelet_map_ptr_, &traffic_rules_ptr_, &routing_graph_ptr_);

    // Create LaneletConverter instance
    // Use dimensions from dimensions.hpp
    constexpr std::array<int64_t, 4> lanes_shape = autoware::diffusion_planner::LANES_SHAPE;
    const size_t max_num_polyline = lanes_shape[1];  // Shape is {1, 70, 20, 13}
    const size_t max_num_point = lanes_shape[2];
    const double point_break_distance = 100.0;
    lanelet_converter_ = std::make_unique<autoware::diffusion_planner::LaneletConverter>(
      lanelet_map_ptr_, max_num_polyline, max_num_point, point_break_distance);

    std::cout << "Lanelet2 map loaded successfully!" << std::endl;
  }

  template <typename T>
  std::optional<T> getNearestMessageWithPop(std::deque<T> & msgs, const rclcpp::Time & target_time)
  {
    if (msgs.empty()) return std::nullopt;

    T best_msg = msgs.front();
    int64_t min_diff =
      std::abs(getMessageTimestamp(best_msg).nanoseconds() - target_time.nanoseconds());
    size_t best_idx = 0;

    // Find best match while checking time differences
    for (size_t i = 1; i < msgs.size(); ++i) {
      int64_t diff =
        std::abs(getMessageTimestamp(msgs[i]).nanoseconds() - target_time.nanoseconds());
      if (diff < min_diff) {
        min_diff = diff;
        best_msg = msgs[i];
        best_idx = i;
      } else {
        break;  // Messages are usually in chronological order
      }
    }

    // Check if time difference is acceptable (200ms like Python version)
    if (min_diff > 200'000'000) {  // 200ms in nanoseconds
      std::cout << "Time difference too large: " << min_diff / 1'000'000 << "ms" << std::endl;
      return std::nullopt;
    }

    // Pop elements up to and including the found message
    for (size_t i = 0; i <= best_idx; ++i) {
      msgs.pop_front();
    }

    return best_msg;
  }

  // Helper function to get timestamp from message with header
  template <typename T>
  rclcpp::Time getMessageTimestamp(const T & msg)
  {
    return rclcpp::Time(msg.header.stamp);
  }

  // Overloads for messages with stamp field instead of header
  rclcpp::Time getMessageTimestamp(const TrafficLightGroupArray & msg)
  {
    return rclcpp::Time(msg.stamp);
  }

  rclcpp::Time getMessageTimestamp(const TurnIndicatorsReport & msg)
  {
    return rclcpp::Time(msg.stamp);
  }

  void createNPYFiles(const std::string & token, const FrameData & frame_data)
  {
    std::cout << "Creating NPY files for token: " << token << std::endl;

    // 1. Create ego state using actual diffusion planner functions
    const double wheel_base = 2.79;  // Same as Python version

    // Use default acceleration if not available
    AccelWithCovarianceStamped accel_msg = frame_data.acceleration;
    if (accel_msg.header.stamp.sec == 0) {
      accel_msg.accel.accel.linear.x = 0.0;
      accel_msg.accel.accel.linear.y = 0.0;
    }

    autoware::diffusion_planner::EgoState ego_state(
      frame_data.kinematic_state, accel_msg, wheel_base);
    std::vector<float> ego_array = ego_state.as_array();
    saveEgoCurrentState(token, ego_array);

    std::cout << "Ego state dimensions: " << ego_array.size() << std::endl;

    // 2. Process tracked objects using dimensions from dimensions.hpp
    constexpr std::array<int64_t, 4> neighbor_shape = autoware::diffusion_planner::NEIGHBOR_SHAPE;
    const int64_t neighbor_num = neighbor_shape[1];  // Shape is {1, 32, 21, 11}
    const int64_t past_time_steps_neighbor = neighbor_shape[2];

    autoware::diffusion_planner::AgentData agent_data(
      frame_data.tracked_objects, neighbor_num, past_time_steps_neighbor);
    saveAgentData(token, agent_data);

    std::cout << "Agent data size: " << agent_data.size() << std::endl;

    // 3. Create lane segments if lanelet converter is available
    std::vector<autoware::diffusion_planner::LaneSegment> lane_segments;
    if (lanelet_converter_) {
      lane_segments = lanelet_converter_->convert_to_lane_segments(20);  // LANE_LEN
      saveLaneData(token, lane_segments);
      saveRouteLanes(token, lane_segments);  // Use lane_segments as placeholder for route

      std::cout << "Lane segments: " << lane_segments.size() << std::endl;
    }

    // 4. Save other data
    saveStaticObjects(token);

    // 5. Use turn indicator data
    int64_t turn_indicator_value = static_cast<int64_t>(frame_data.turn_indicator.report);
    saveTurnIndicator(token, turn_indicator_value);

    // 6. Save kinematic info as JSON like Python version
    saveKinematicInfo(token, frame_data.kinematic_state);

    std::cout << "Created NPY files for token: " << token << std::endl;
  }

  void saveEgoCurrentState(const std::string & token, const std::vector<float> & ego_array)
  {
    const std::string filename = config_.save_dir + "/ego_current_state_" + token + ".npy";
    aoba::SaveArrayAsNumpy(filename, ego_array);
    std::cout << "  Saved ego_current_state: " << filename << " (shape: " << ego_array.size() << ")"
              << std::endl;
  }

  void saveAgentData(
    const std::string & token, const autoware::diffusion_planner::AgentData & agent_data)
  {
    // Save agent data as NPY file
    const std::string npy_filename = config_.save_dir + "/neighbor_agents_past_" + token + ".npy";

    // Get the tensor data from AgentData
    const int64_t num_agents = agent_data.num_agent();
    const int64_t time_length = agent_data.time_length();
    constexpr std::array<int64_t, 4> neighbor_shape = autoware::diffusion_planner::NEIGHBOR_SHAPE;
    const int64_t feature_dim = neighbor_shape[3];  // Shape is {1, 32, 21, 11}

    // Get actual tensor data from AgentData
    const std::vector<float> agent_tensor_data = agent_data.as_vector();

    // Save as 3D array: (num_agents, time_length, feature_dim)
    const int shape[3] = {
      static_cast<int>(num_agents), static_cast<int>(time_length), static_cast<int>(feature_dim)};
    aoba::SaveArrayAsNumpy(npy_filename, 3, shape, agent_tensor_data.data());

    std::cout << "  Saved neighbor_agents_past: " << npy_filename << " (shape: " << num_agents
              << ", " << time_length << ", " << feature_dim << ")" << std::endl;
  }

  void saveLaneData(
    const std::string & token,
    const std::vector<autoware::diffusion_planner::LaneSegment> & lane_segments)
  {
    // Save lane data as NPY file
    const std::string npy_filename = config_.save_dir + "/lanes_" + token + ".npy";

    // Use dimensions from dimensions.hpp
    constexpr std::array<int64_t, 4> lanes_shape = autoware::diffusion_planner::LANES_SHAPE;
    const int64_t max_lane_num = lanes_shape[1];  // Shape is {1, 70, 20, 13}
    const int64_t max_lane_len = lanes_shape[2];
    const int64_t lane_feature_dim = lanes_shape[3];

    // Create lane tensor data with correct shape (70, 20, 13)
    std::vector<float> lane_tensor_data(max_lane_num * max_lane_len * lane_feature_dim, 0.0f);

    // Fill with actual lane data (simplified version)
    for (size_t i = 0; i < std::min(lane_segments.size(), static_cast<size_t>(max_lane_num)); ++i) {
      const autoware::diffusion_planner::LaneSegment & segment = lane_segments[i];
      const std::vector<autoware::diffusion_planner::LanePoint> & waypoints =
        segment.polyline.waypoints();

      for (size_t j = 0; j < std::min(waypoints.size(), static_cast<size_t>(max_lane_len)); ++j) {
        const autoware::diffusion_planner::LanePoint & point = waypoints[j];
        size_t base_idx = i * max_lane_len * lane_feature_dim + j * lane_feature_dim;

        if (base_idx + 2 < lane_tensor_data.size()) {
          lane_tensor_data[base_idx + 0] = point.x();  // x
          lane_tensor_data[base_idx + 1] = point.y();  // y
          lane_tensor_data[base_idx + 2] = point.z();  // z
          // Other features would be filled here in a complete implementation
        }
      }
    }

    // Save as 3D array: (max_lane_num, max_lane_len, lane_feature_dim)
    const int shape[3] = {
      static_cast<int>(max_lane_num), static_cast<int>(max_lane_len),
      static_cast<int>(lane_feature_dim)};
    aoba::SaveArrayAsNumpy(npy_filename, 3, shape, lane_tensor_data.data());

    std::cout << "  Saved lanes: " << npy_filename << " (shape: " << max_lane_num << ", "
              << max_lane_len << ", " << lane_feature_dim << ")" << std::endl;
  }

  void saveStaticObjects(const std::string & token)
  {
    const std::string filename = config_.save_dir + "/static_objects_" + token + ".npy";

    // Create static objects data using dimensions from dimensions.hpp
    constexpr std::array<int64_t, 3> static_shape =
      autoware::diffusion_planner::STATIC_OBJECTS_SHAPE;
    const int64_t num_static_objects = static_shape[1];  // Shape is {1, 5, 10}
    const int64_t object_features = static_shape[2];

    // Create placeholder static objects data
    std::vector<float> static_objects_data(num_static_objects * object_features, 0.0f);

    // Save as 2D array: (num_static_objects, object_features)
    const int shape[2] = {static_cast<int>(num_static_objects), static_cast<int>(object_features)};
    aoba::SaveArrayAsNumpy(filename, 2, shape, static_objects_data.data());

    std::cout << "  Saved static_objects: " << filename << " (shape: " << num_static_objects << ", "
              << object_features << ")" << std::endl;
  }

  void saveRouteLanes(
    const std::string & token,
    const std::vector<autoware::diffusion_planner::LaneSegment> & lane_segments)
  {
    const std::string filename = config_.save_dir + "/route_lanes_" + token + ".npy";
    // Create route lanes data using dimensions from dimensions.hpp
    constexpr std::array<int64_t, 4> route_shape = autoware::diffusion_planner::ROUTE_LANES_SHAPE;
    const int64_t num_route_lanes = route_shape[1];  // Shape is {1, 25, 20, 13}
    const int64_t route_lane_len = route_shape[2];
    const int64_t route_lane_features = route_shape[3];

    // Create route lanes tensor data
    std::vector<float> route_lanes_data(
      num_route_lanes * route_lane_len * route_lane_features, 0.0f);

    // Fill with actual route lane data (simplified version using available lane_segments)
    for (size_t i = 0; i < std::min(lane_segments.size(), static_cast<size_t>(num_route_lanes));
         ++i) {
      const autoware::diffusion_planner::LaneSegment & segment = lane_segments[i];
      const std::vector<autoware::diffusion_planner::LanePoint> & waypoints =
        segment.polyline.waypoints();

      for (size_t j = 0; j < std::min(waypoints.size(), static_cast<size_t>(route_lane_len)); ++j) {
        const autoware::diffusion_planner::LanePoint & point = waypoints[j];
        size_t base_idx = i * route_lane_len * route_lane_features + j * route_lane_features;

        if (base_idx + 2 < route_lanes_data.size()) {
          route_lanes_data[base_idx + 0] = point.x();  // x
          route_lanes_data[base_idx + 1] = point.y();  // y
          route_lanes_data[base_idx + 2] = point.z();  // z
          // Other features would be filled here in a complete implementation
        }
      }
    }

    // Save as 3D array: (num_route_lanes, route_lane_len, route_lane_features)
    const int shape[3] = {
      static_cast<int>(num_route_lanes), static_cast<int>(route_lane_len),
      static_cast<int>(route_lane_features)};
    aoba::SaveArrayAsNumpy(filename, 3, shape, route_lanes_data.data());

    std::cout << "  Saved route_lanes: " << filename << " (shape: " << num_route_lanes << ", "
              << route_lane_len << ", " << route_lane_features << ")" << std::endl;
  }

  void saveTurnIndicator(const std::string & token, int64_t turn_indicator_value)
  {
    const std::string filename = config_.save_dir + "/turn_indicator_" + token + ".npy";

    // Create turn indicator data with shape (1,) as mentioned in conversation
    std::vector<float> turn_indicator_data = {static_cast<float>(turn_indicator_value)};

    // Save as 1D array: (1,)
    aoba::SaveArrayAsNumpy(filename, turn_indicator_data);

    std::cout << "  Saved turn_indicator: " << filename << " (shape: 1)" << std::endl;
  }

  void saveKinematicInfo(const std::string & token, const Odometry & kinematic_msg)
  {
    const std::string filename = config_.save_dir + "/" + token + ".json";
    std::ofstream file(filename);
    if (file.is_open()) {
      file << "{\n";
      file << "  \"timestamp\": "
           << kinematic_msg.header.stamp.sec * 1000000000LL + kinematic_msg.header.stamp.nanosec
           << ",\n";
      file << "  \"x\": " << kinematic_msg.pose.pose.position.x << ",\n";
      file << "  \"y\": " << kinematic_msg.pose.pose.position.y << ",\n";
      file << "  \"z\": " << kinematic_msg.pose.pose.position.z << ",\n";
      file << "  \"qx\": " << kinematic_msg.pose.pose.orientation.x << ",\n";
      file << "  \"qy\": " << kinematic_msg.pose.pose.orientation.y << ",\n";
      file << "  \"qz\": " << kinematic_msg.pose.pose.orientation.z << ",\n";
      file << "  \"qw\": " << kinematic_msg.pose.pose.orientation.w << "\n";
      file << "}\n";
      file.close();
    }
  }

  ParseRosbagConfig config_;
  std::vector<std::string> target_topics_;
  rosbag2_cpp::readers::SequentialReader reader_;
  rosbag2_storage::StorageOptions storage_options_;
  rosbag2_cpp::ConverterOptions converter_options_;

  // Lanelet2 map processing
  LaneletMapBin map_bin_msg_;
  std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr_;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_ptr_;
  lanelet::routing::RoutingGraphPtr routing_graph_ptr_;
  std::unique_ptr<autoware::diffusion_planner::LaneletConverter> lanelet_converter_;
};

void printUsage(const char * program_name)
{
  std::cout << "Usage: " << program_name
            << " <rosbag_path> <vector_map_path> <save_dir> [options]\n";
  std::cout << "\nOptions:\n";
  std::cout << "  --step <int>           Step size (default: 1)\n";
  std::cout
    << "  --limit <int>          Limit number of messages to process (default: -1, no limit)\n";
  std::cout << "  --min_frames <int>     Minimum frames required (default: 1700)\n";
  std::cout << "  --search_nearest_route <0|1>  Search for nearest route (default: 1)\n";
  std::cout << "\nExample:\n";
  std::cout << "  " << program_name
            << " /path/to/rosbag /path/to/map /path/to/output --step 2 --limit 10000\n";
}

ParseRosbagConfig parseArguments(int argc, char * argv[])
{
  ParseRosbagConfig config;

  if (argc < 4) {
    printUsage(argv[0]);
    exit(1);
  }

  config.rosbag_path = argv[1];
  config.vector_map_path = argv[2];
  config.save_dir = argv[3];

  // Parse optional arguments
  for (int64_t i = 4; i < argc; i += 2) {
    if (i + 1 >= argc) break;

    std::string arg = argv[i];
    if (arg == "--step") {
      config.step = std::stoll(argv[i + 1]);
    } else if (arg == "--limit") {
      config.limit = std::stoll(argv[i + 1]);
    } else if (arg == "--min_frames") {
      config.min_frames = std::stoll(argv[i + 1]);
    } else if (arg == "--search_nearest_route") {
      config.search_nearest_route = std::stoll(argv[i + 1]) != 0;
    }
  }

  return config;
}

void printConfiguration(const ParseRosbagConfig & config)
{
  std::cout << "Configuration:" << std::endl;
  std::cout << "  Rosbag path: " << config.rosbag_path << std::endl;
  std::cout << "  Vector map path: " << config.vector_map_path << std::endl;
  std::cout << "  Save directory: " << config.save_dir << std::endl;
  std::cout << "  Step: " << config.step << std::endl;
  std::cout << "  Limit: " << config.limit << std::endl;
  std::cout << "  Min frames: " << config.min_frames << std::endl;
  std::cout << "  Search nearest route: " << (config.search_nearest_route ? "true" : "false")
            << std::endl;
}

int main(int argc, char * argv[])
{
  // 1. Initialize ROS 2
  rclcpp::init(argc, argv);

  // 2. Parse and validate arguments
  ParseRosbagConfig config = parseArguments(argc, argv);

  if (!fs::exists(config.rosbag_path)) {
    std::cerr << "Error: Rosbag path does not exist: " << config.rosbag_path << std::endl;
    return 1;
  }
  if (!fs::exists(config.vector_map_path)) {
    std::cerr << "Error: Vector map path does not exist: " << config.vector_map_path << std::endl;
    return 1;
  }

  // 3. Print configuration
  printConfiguration(config);

  // 4. Create parser and load map
  RosbagParser parser(config);
  parser.loadLaneletMap();

  // 5. Read rosbag messages
  MessageCollections message_collections = parser.readRosbagMessages();

  // 6. Process messages and create data list
  std::vector<FrameData> data_list = parser.createDataList(message_collections);

  // 7. Process data list and generate output files
  fs::create_directories(config.save_dir);
  parser.processDataList(data_list);

  std::cout << "\nRosbag parsing completed successfully!" << std::endl;

  rclcpp::shutdown();
  return 0;
}
