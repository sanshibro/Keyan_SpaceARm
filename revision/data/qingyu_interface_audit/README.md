# qingyu_api_sdk

ROS1 C++ SDK for A01 runtime.

Main features:

- publish joint commands
- publish pose commands
- switch control mode
- trigger calibration
- read and write config
- switch serial device

Example CLI usage:

```bash
rosrun qingyu_api_sdk qingyu_api_sdk_cli joint 30 0 0 0 0 0 1
rosrun qingyu_api_sdk qingyu_api_sdk_cli closed_loop
rosrun qingyu_api_sdk qingyu_api_sdk_cli read_one 0x10
rosrun qingyu_api_sdk qingyu_api_sdk_cli serial /dev/ttyACM1 115200
```
