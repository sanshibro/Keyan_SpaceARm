#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qingyu_api_sdk/visibility_control.hpp"

namespace qingyu_api_sdk
{

struct ClientOptions
{
  std::string node_name{"qingyu_api_sdk_client"};
  std::string command_topic{"/qingyu_api"};
  std::string pose_topic{"/qingyu_api_pose"};
  std::string serial_node_name{"/qingyu_api_node"};
  int service_timeout_ms{5000};
};

struct JointCommand
{
  float theta1{0.0F};
  float phi1{0.0F};
  float theta2{0.0F};
  float phi2{0.0F};
  float theta3{0.0F};
  float phi3{0.0F};
  bool enable{true};
};

struct PoseCommand
{
  float x{0.0F};
  float y{0.0F};
  float z{0.0F};
  float roll{0.0F};
  float pitch{0.0F};
  float yaw{0.0F};
  bool enable{true};
};

enum class ControlMode : std::uint8_t
{
  ClosedLoop = 0x01,
  OpenLoop = 0x02,
};

enum class ConfigType : std::uint8_t
{
  Float = 0x01,
  Int32 = 0x02,
  UInt8 = 0x03,
};

struct QINGYU_API_SDK_PUBLIC ConfigValue
{
  ConfigType type{ConfigType::Float};
  std::array<std::uint8_t, 4> raw{{0, 0, 0, 0}};

  static ConfigValue fromFloat(float value);
  static ConfigValue fromInt32(std::int32_t value);
  static ConfigValue fromUInt8(std::uint8_t value);

  float toFloat() const;
  std::int32_t toInt32() const;
  std::uint8_t toUInt8() const;
};

struct ConfigEntry
{
  std::uint8_t param_id{0};
  ConfigValue value{};
};

class QINGYU_API_SDK_PUBLIC Client
{
public:
  explicit Client(ClientOptions options = {});
  ~Client();

  Client(const Client &) = delete;
  Client & operator=(const Client &) = delete;
  Client(Client &&) noexcept;
  Client & operator=(Client &&) noexcept;

  bool start();
  void stop();
  bool isStarted() const;

  bool moveJoints(const JointCommand & command);
  bool movePose(const PoseCommand & command);
  bool calibrate();
  bool enableClosedLoopControl();
  bool enableOpenLoopControl();
  bool switchSerialDevice(const std::string & serial_device, std::optional<int> baudrate = std::nullopt);
  bool readConfig(std::uint8_t param_id, ConfigEntry & entry);
  bool writeConfig(std::uint8_t param_id, const ConfigValue & value);
  bool readAllConfigs(std::vector<ConfigEntry> & entries);
  bool restoreDefaultConfig();

  std::string lastError() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qingyu_api_sdk
