# ODrive ros2_control Plugin

This package serves as a hardware interface to control ODrives from [ros2_control](https://control.ros.org/master/index.html).

It assumes that the ODrive is already configured and calibrated (see [docs](https://docs.odriverobotics.com/v/latest/guides/getting-started.html) for details).

## Usage

For a high level usage example, see the [BotWheel Explorer ROS2 Package](../odrive_botwheel_explorer/README.md).

## Features

- Communicates over Linux SocketCAN
- Position Control (with optional velocity and torque feedforward)
- Velocity Control (with optional torque feedforward)
- Torque Control
- Automatic control mode selection (based on which Command Interfaces are claimed by the ros2_control Controller)
- Position, velocity and torque Feedback
- Multiple ODrives

## Parameters

Top level:

- `can`: Name of the CAN interface to run on

Per joint:

- `node_id`: `node_id` of the ODrive

## Command Interfaces

(from ros2_control Controller to ODrive)

- `position`
- `velocity`
- `effort` (aka Torque)

## State Interfaces

(from ODrive to ros2_control Controller)

- `position`
- `velocity`
- `effort` (aka Torque)
