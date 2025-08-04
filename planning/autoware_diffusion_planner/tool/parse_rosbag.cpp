#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <rosbag2_storage/storage_filter.hpp>

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
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>
#include <geometry_msgs/msg/accel_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace fs = std::filesystem;

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
  autoware_planning_msgs::msg::LaneletRoute::SharedPtr route;
  autoware_perception_msgs::msg::TrackedObjects::SharedPtr tracked_objects;
  nav_msgs::msg::Odometry::SharedPtr kinematic_state;
  geometry_msgs::msg::AccelWithCovarianceStamped::SharedPtr acceleration;
  autoware_perception_msgs::msg::TrafficLightGroupArray::SharedPtr traffic_signals;
  autoware_vehicle_msgs::msg::TurnIndicatorsReport::SharedPtr turn_indicator;
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
  }

  bool parseRosbag()
  {
    try {
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

    } catch (const std::exception & e) {
      std::cerr << "Error parsing rosbag: " << e.what() << std::endl;
      return false;
    }
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
    processSampleFrames(
      kinematic_msgs, acceleration_msgs, tracking_msgs, traffic_msgs, route_msgs,
      turn_indicator_msgs);
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
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & /* acceleration_msgs */,
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & tracking_msgs,
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & /* traffic_msgs */,
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & /* route_msgs */,
    const std::vector<rosbag2_storage::SerializedBagMessageSharedPtr> & /* turn_indicator_msgs */)
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

      try {
        // Deserialize sample messages for demonstration
        if (i < static_cast<int64_t>(kinematic_msgs.size())) {
          auto kinematic_msg = deserializeMessage<nav_msgs::msg::Odometry>(kinematic_msgs[i]);

          // Create sample NPZ data structure file
          createSampleNPZFile(token, kinematic_msg);
        }

      } catch (const std::exception & e) {
        std::cerr << "Error processing frame " << i << ": " << e.what() << std::endl;
        continue;
      }

      if (i % 100 == 0) {
        std::cout << "Processed sample frame " << i << "/" << sample_count << std::endl;
      }
    }

    std::cout << "Sample frame processing completed!" << std::endl;
  }

  void createSampleNPZFile(const std::string & token, const nav_msgs::msg::Odometry & kinematic_msg)
  {
    std::string npz_info_file = config_.save_dir + "/sample_" + token + ".npz.txt";
    std::string json_file = config_.save_dir + "/sample_" + token + ".json";

    // Create NPZ structure file
    std::ofstream npz_info(npz_info_file);
    if (npz_info.is_open()) {
      npz_info << "NPZ file structure for token: " << token << "\n";
      npz_info << "=================================\n";
      npz_info << "map_name: " << fs::path(config_.rosbag_path).stem().string() << "\n";
      npz_info << "token: " << token << "\n";
      npz_info << "ego_agent_past: (21, 3) - Past trajectory of ego vehicle\n";
      npz_info << "ego_current_state: (10,) - Current state [x, y, heading_cos, heading_sin, vx, "
                  "vy, ax, ay, curvature, wheel_base]\n";
      npz_info << "ego_agent_future: (80, 3) - Future trajectory of ego vehicle\n";
      npz_info << "neighbor_agents_past: (32, 21, 11) - Past states of neighboring agents\n";
      npz_info
        << "neighbor_agents_future: (32, 80, 3) - Future trajectories of neighboring agents\n";
      npz_info << "static_objects: (5, 10) - Static objects in the scene\n";
      npz_info << "lanes: (70, 20, 13) - Lane information\n";
      npz_info << "lanes_speed_limit: (70, 1) - Speed limits for lanes\n";
      npz_info
        << "lanes_has_speed_limit: (70, 1) - Boolean array indicating if lane has speed limit\n";
      npz_info << "route_lanes: (25, 20, 13) - Route lane information\n";
      npz_info << "route_lanes_speed_limit: (25, 1) - Speed limits for route lanes\n";
      npz_info << "route_lanes_has_speed_limit: (25, 1) - Boolean array for route speed limits\n";
      npz_info << "turn_indicator: (1,) - Turn indicator state\n";
      npz_info << "goal_pose: (3,) - Relative goal pose [x, y, yaw]\n";
      npz_info.close();
    }

    // Create JSON file with pose information from actual kinematic data
    std::ofstream json_out(json_file);
    if (json_out.is_open()) {
      json_out << "{\n";
      json_out << "  \"timestamp\": "
               << kinematic_msg.header.stamp.sec * 1000000000LL + kinematic_msg.header.stamp.nanosec
               << ",\n";
      json_out << "  \"x\": " << kinematic_msg.pose.pose.position.x << ",\n";
      json_out << "  \"y\": " << kinematic_msg.pose.pose.position.y << ",\n";
      json_out << "  \"z\": " << kinematic_msg.pose.pose.position.z << ",\n";
      json_out << "  \"qx\": " << kinematic_msg.pose.pose.orientation.x << ",\n";
      json_out << "  \"qy\": " << kinematic_msg.pose.pose.orientation.y << ",\n";
      json_out << "  \"qz\": " << kinematic_msg.pose.pose.orientation.z << ",\n";
      json_out << "  \"qw\": " << kinematic_msg.pose.pose.orientation.w << "\n";
      json_out << "}\n";
      json_out.close();
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

  try {
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

  } catch (const std::exception & e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
