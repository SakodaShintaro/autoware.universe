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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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

struct FrameData
{
  rclcpp::Time timestamp;
  LaneletRoute::SharedPtr route;
  TrackedObjects::SharedPtr tracked_objects;
  Odometry::SharedPtr kinematic_state;
  AccelWithCovarianceStamped::SharedPtr acceleration;
  TrafficLightGroupArray::SharedPtr traffic_signals;
  TurnIndicatorsReport::SharedPtr turn_indicator;
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

  bool parseRosbag()
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
      auto bag_message = reader_.read_next();
      topic_counts[bag_message->topic_name]++;
      total_messages++;

      if (config_.limit > 0 && total_messages >= config_.limit) {
        break;
      }
    }

    // Print statistics
    std::cout << "\nMessage counts per topic:" << std::endl;
    for (const auto & [topic, count] : topic_counts) {
      std::cout << "  " << topic << ": " << count << " messages" << std::endl;
    }

    // Reopen rosbag for processing
    reader_.close();
    reader_.open(storage_options_, converter_options_);
    reader_.set_filter(storage_filter);

    // Process messages
    std::cout << "\nProcessing messages..." << std::endl;

    // Storage for collected messages
    std::map<std::string, std::vector<rosbag2_storage::SerializedBagMessageSharedPtr>>
      topic_messages;

    while (reader_.has_next() && (config_.limit < 0 || processed_messages < config_.limit)) {
      auto bag_message = reader_.read_next();

      // Store message for processing
      topic_messages[bag_message->topic_name].push_back(bag_message);

      processed_messages++;

      if (processed_messages % 1000 == 0) {
        std::cout << "Processed " << processed_messages << "/" << total_messages << " messages..."
                  << std::endl;
      }
    }

    reader_.close();

    // Create output directory
    fs::create_directories(config_.save_dir);

    // Save parsing results
    saveParsingResults(topic_counts);

    // Process collected messages and create output data
    processMessages(topic_messages);

    std::cout << "\nRosbag parsing completed successfully!" << std::endl;
    std::cout << "Total messages processed: " << processed_messages << std::endl;

    return true;
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
    const size_t max_num_polyline = 70;  // Same as Python LANE_NUM
    const size_t max_num_point = 20;     // Same as Python LANE_LEN
    const double point_break_distance = 100.0;
    lanelet_converter_ = std::make_unique<autoware::diffusion_planner::LaneletConverter>(
      lanelet_map_ptr_, max_num_polyline, max_num_point, point_break_distance);

    std::cout << "Lanelet2 map loaded successfully!" << std::endl;
  }

  void processMessages(
    const std::map<std::string, std::vector<rosbag2_storage::SerializedBagMessageSharedPtr>> &
      topic_messages)
  {
    std::cout << "\nProcessing collected messages..." << std::endl;

    // Get message vectors for each topic
    auto kinematic_msgs = getTopicMessages(topic_messages, "/localization/kinematic_state");
    auto acceleration_msgs = getTopicMessages(topic_messages, "/localization/acceleration");
    auto tracking_msgs =
      getTopicMessages(topic_messages, "/perception/object_recognition/tracking/objects");
    auto traffic_msgs =
      getTopicMessages(topic_messages, "/perception/traffic_light_recognition/traffic_signals");
    auto route_msgs = getTopicMessages(topic_messages, "/planning/mission_planning/route");
    auto turn_indicator_msgs =
      getTopicMessages(topic_messages, "/vehicle/status/turn_indicators_status");

    std::cout << "Collected messages:" << std::endl;
    std::cout << "  Kinematic: " << kinematic_msgs.size() << std::endl;
    std::cout << "  Acceleration: " << acceleration_msgs.size() << std::endl;
    std::cout << "  Tracking: " << tracking_msgs.size() << std::endl;
    std::cout << "  Traffic: " << traffic_msgs.size() << std::endl;
    std::cout << "  Route: " << route_msgs.size() << std::endl;
    std::cout << "  Turn indicator: " << turn_indicator_msgs.size() << std::endl;

    // For demonstration, process a few sample frames
    processSampleFrames(kinematic_msgs, tracking_msgs);
  }

  std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> getTopicMessages(
    const std::map<std::string, std::vector<rosbag2_storage::SerializedBagMessageSharedPtr>> &
      topic_messages,
    const std::string & topic_name)
  {
    auto it = topic_messages.find(topic_name);
    if (it != topic_messages.end()) {
      return it->second;
    }
    return {};
  }

  void processSampleFrames(
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & kinematic_msgs,
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & tracking_msgs)
  {
    std::cout << "\nProcessing sample frames..." << std::endl;

    // Use tracking messages as base (10Hz in the original Python code)
    int64_t sample_count =
      std::min(static_cast<int64_t>(tracking_msgs.size()), static_cast<int64_t>(10));

    for (int64_t i = 0; i < sample_count; i += config_.step) {
      // Generate token (sequence_id + frame_id)
      std::ostringstream oss;
      oss << std::setfill('0') << std::setw(8) << 0 << std::setw(8) << i;
      std::string token = oss.str();

      // Deserialize sample messages for demonstration
      if (i < static_cast<int64_t>(kinematic_msgs.size())) {
        auto kinematic_msg = deserializeMessage<Odometry>(kinematic_msgs[i]);
        auto tracking_msg = deserializeMessage<TrackedObjects>(tracking_msgs[i]);

        // Create NPY files
        createNPYFiles(token, kinematic_msg, tracking_msg);
      }

      if (i % 100 == 0) {
        std::cout << "Processed sample frame " << i << "/" << sample_count << std::endl;
      }
    }

    std::cout << "Sample frame processing completed!" << std::endl;
  }

  void createNPYFiles(
    const std::string & token, const Odometry & kinematic_msg,
    const TrackedObjects & tracked_objects_msg)
  {
    std::cout << "Creating NPY files for token: " << token << std::endl;

    // 1. Create ego state using actual diffusion planner functions
    const double wheel_base = 2.79;  // Same as Python version
    AccelWithCovarianceStamped dummy_accel;
    dummy_accel.accel.accel.linear.x = 0.0;
    dummy_accel.accel.accel.linear.y = 0.0;

    autoware::diffusion_planner::EgoState ego_state(kinematic_msg, dummy_accel, wheel_base);
    auto ego_array = ego_state.as_array();

    // 2. Process tracked objects
    autoware::diffusion_planner::AgentData agent_data(
      tracked_objects_msg, 32, 21);  // NEIGHBOR_NUM, PAST_TIME_STEPS

    // 3. Create lane segments if lanelet converter is available
    std::vector<autoware::diffusion_planner::LaneSegment> lane_segments;
    if (lanelet_converter_) {
      lane_segments = lanelet_converter_->convert_to_lane_segments(20);  // LANE_LEN
    }

    std::cout << "Ego state dimensions: " << ego_array.size() << std::endl;
    std::cout << "Agent data size: " << agent_data.size() << std::endl;
    std::cout << "Lane segments: " << lane_segments.size() << std::endl;

    // Save NPY files
    saveEgoCurrentState(token, ego_array);
    saveAgentData(token, agent_data);
    saveLaneData(token, lane_segments);
    saveStaticObjects(token);
    saveRouteLanes(token, lane_segments);  // Use lane_segments as placeholder for route
    saveTurnIndicator(token, 0);           // Placeholder value
    saveKinematicInfo(token, kinematic_msg);

    std::cout << "Created NPY files for token: " << token << std::endl;
  }

  void saveEgoCurrentState(const std::string & token, const std::vector<float> & ego_array)
  {
    std::string filename = config_.save_dir + "/ego_current_state_" + token + ".npy";
    aoba::SaveArrayAsNumpy(filename, ego_array);
    std::cout << "  Saved ego_current_state: " << filename << " (shape: " << ego_array.size() << ")"
              << std::endl;
  }

  void saveAgentData(
    const std::string & token, const autoware::diffusion_planner::AgentData & agent_data)
  {
    // Save agent data as NPY file
    std::string npy_filename = config_.save_dir + "/neighbor_agents_past_" + token + ".npy";

    // Get the tensor data from AgentData
    // The AgentData should have methods to get the tensor data
    // For now, we'll create a placeholder with the correct dimensions
    const int64_t num_agents = agent_data.num_agent();
    const int64_t time_length = agent_data.time_length();
    const int64_t feature_dim = 11;  // Based on Python code: (32, 21, 11)

    // Create placeholder data with correct shape
    std::vector<float> agent_tensor_data(num_agents * time_length * feature_dim, 0.0f);

    // Save as 3D array: (num_agents, time_length, feature_dim)
    const int shape[3] = {
      static_cast<int>(num_agents), static_cast<int>(time_length), static_cast<int>(feature_dim)};
    aoba::SaveArrayAsNumpy(npy_filename, 3, shape, agent_tensor_data.data());

    std::cout << "  Saved neighbor_agents_past: " << npy_filename << " (shape: " << num_agents
              << ", " << time_length << ", " << feature_dim << ")" << std::endl;

    // Also save info file
    std::string info_filename = config_.save_dir + "/agent_data_" + token + ".txt";
    std::ofstream file(info_filename);
    if (file.is_open()) {
      file << "Agent data for token: " << token << "\n";
      file << "Number of agents: " << agent_data.num_agent() << "\n";
      file << "Time length: " << agent_data.time_length() << "\n";
      file << "Data size: " << agent_data.size() << "\n";
      file << "NPY file: " << npy_filename << "\n";
      file.close();
    }
  }

  void saveLaneData(
    const std::string & token,
    const std::vector<autoware::diffusion_planner::LaneSegment> & lane_segments)
  {
    // Save lane data as NPY file
    std::string npy_filename = config_.save_dir + "/lanes_" + token + ".npy";

    const int64_t max_lane_num = 70;      // LANE_NUM from Python code
    const int64_t max_lane_len = 20;      // LANE_LEN from Python code
    const int64_t lane_feature_dim = 13;  // Feature dimension for lanes

    // Create lane tensor data with correct shape (70, 20, 13)
    std::vector<float> lane_tensor_data(max_lane_num * max_lane_len * lane_feature_dim, 0.0f);

    // Fill with actual lane data (simplified version)
    for (size_t i = 0; i < std::min(lane_segments.size(), static_cast<size_t>(max_lane_num)); ++i) {
      const auto & segment = lane_segments[i];
      const auto & waypoints = segment.polyline.waypoints();

      for (size_t j = 0; j < std::min(waypoints.size(), static_cast<size_t>(max_lane_len)); ++j) {
        const auto & point = waypoints[j];
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

    // Also save info file
    std::string info_filename = config_.save_dir + "/lane_data_" + token + ".txt";
    std::ofstream file(info_filename);
    if (file.is_open()) {
      file << "Lane data for token: " << token << "\n";
      file << "Number of lane segments: " << lane_segments.size() << "\n";
      file << "NPY file: " << npy_filename << "\n";
      for (size_t i = 0; i < std::min(lane_segments.size(), size_t(5)); ++i) {
        const auto & segment = lane_segments[i];
        file << "Segment " << i << " (ID: " << segment.id << "): " << segment.polyline.size()
             << " points\n";
      }
      file.close();
    }
  }

  void saveStaticObjects(const std::string & token)
  {
    std::string filename = config_.save_dir + "/static_objects_" + token + ".npy";

    // Create static objects data with shape (5, 10) as mentioned in conversation
    const int64_t num_static_objects = 5;
    const int64_t object_features = 10;

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
    std::string filename = config_.save_dir + "/route_lanes_" + token + ".npy";
    // Create route lanes data with shape (25, 20, 13) as mentioned in conversation
    const int64_t num_route_lanes = 25;
    const int64_t route_lane_len = 20;
    const int64_t route_lane_features = 13;

    // Create route lanes tensor data
    std::vector<float> route_lanes_data(
      num_route_lanes * route_lane_len * route_lane_features, 0.0f);

    // Fill with actual route lane data (simplified version using available lane_segments)
    for (size_t i = 0; i < std::min(lane_segments.size(), static_cast<size_t>(num_route_lanes));
         ++i) {
      const auto & segment = lane_segments[i];
      const auto & waypoints = segment.polyline.waypoints();

      for (size_t j = 0; j < std::min(waypoints.size(), static_cast<size_t>(route_lane_len)); ++j) {
        const auto & point = waypoints[j];
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
    std::string filename = config_.save_dir + "/turn_indicator_" + token + ".npy";

    // Create turn indicator data with shape (1,) as mentioned in conversation
    std::vector<float> turn_indicator_data = {static_cast<float>(turn_indicator_value)};

    // Save as 1D array: (1,)
    aoba::SaveArrayAsNumpy(filename, turn_indicator_data);

    std::cout << "  Saved turn_indicator: " << filename << " (shape: 1)" << std::endl;
  }

  void saveKinematicInfo(const std::string & token, const Odometry & kinematic_msg)
  {
    std::string filename = config_.save_dir + "/kinematic_" + token + ".json";
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
      file << "  \"qw\": " << kinematic_msg.pose.pose.orientation.w << ",\n";
      file << "  \"vx\": " << kinematic_msg.twist.twist.linear.x << ",\n";
      file << "  \"vy\": " << kinematic_msg.twist.twist.linear.y << ",\n";
      file << "  \"vz\": " << kinematic_msg.twist.twist.linear.z << "\n";
      file << "}\n";
      file.close();
    }
  }

  void saveParsingResults(const std::map<std::string, int64_t> & topic_counts)
  {
    std::string results_file = config_.save_dir + "/parsing_results.txt";
    std::ofstream ofs(results_file);

    if (ofs.is_open()) {
      ofs << "Rosbag parsing results\n";
      ofs << "======================\n";
      ofs << "Source rosbag: " << config_.rosbag_path << "\n";
      ofs << "Vector map: " << config_.vector_map_path << "\n";
      ofs << "Output directory: " << config_.save_dir << "\n";
      ofs << "Configuration:\n";
      ofs << "  Step: " << config_.step << "\n";
      ofs << "  Limit: " << config_.limit << "\n";
      ofs << "  Min frames: " << config_.min_frames << "\n";
      ofs << "  Search nearest route: " << (config_.search_nearest_route ? "true" : "false")
          << "\n";
      ofs << "\nMessage counts:\n";

      for (const auto & [topic, count] : topic_counts) {
        ofs << topic << ": " << count << " messages\n";
      }

      ofs.close();
      std::cout << "Results saved to: " << results_file << std::endl;
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

int main(int argc, char * argv[])
{
  // Initialize ROS 2
  rclcpp::init(argc, argv);

  ParseRosbagConfig config = parseArguments(argc, argv);

  // Validate input paths
  if (!fs::exists(config.rosbag_path)) {
    std::cerr << "Error: Rosbag path does not exist: " << config.rosbag_path << std::endl;
    return 1;
  }

  if (!fs::exists(config.vector_map_path)) {
    std::cerr << "Error: Vector map path does not exist: " << config.vector_map_path << std::endl;
    return 1;
  }

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Rosbag path: " << config.rosbag_path << std::endl;
  std::cout << "  Vector map path: " << config.vector_map_path << std::endl;
  std::cout << "  Save directory: " << config.save_dir << std::endl;
  std::cout << "  Step: " << config.step << std::endl;
  std::cout << "  Limit: " << config.limit << std::endl;
  std::cout << "  Min frames: " << config.min_frames << std::endl;
  std::cout << "  Search nearest route: " << (config.search_nearest_route ? "true" : "false")
            << std::endl;

  RosbagParser parser(config);

  if (parser.parseRosbag()) {
    std::cout << "\nRosbag parsing completed successfully!" << std::endl;
    return 0;
  } else {
    std::cerr << "Failed to parse rosbag!" << std::endl;
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
