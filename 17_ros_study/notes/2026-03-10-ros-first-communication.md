# ROS 2 First Communication Practice

## Overview
Today, I practiced the basic communication workflow in ROS 2 using Docker on my MacBook Pro (Apple M2 Pro).

My goal was not to write code yet, but to understand how ROS 2 nodes communicate with each other through a topic.

## Environment
- MacBook Pro (Apple M2 Pro)
- Docker Desktop
- ROS 2 Jazzy
- Docker image: `osrf/ros:jazzy-desktop`

## What I did
Today, I completed the following steps:

1. Ran a ROS 2 Docker container
2. Checked that ROS 2 commands worked inside the container
3. Ran a demo `talker` node
4. Ran a demo `listener` node
5. Observed that the two nodes communicated through the `/chatter` topic
6. Used `ros2 topic echo /chatter` to confirm that messages were being published and received
7. Stopped the running nodes and exited the container properly

## Commands I used

### Start the ROS 2 container
```bash
docker run --platform linux/amd64 -it --rm osrf/ros:jazzy-desktop

##Check ROS 2 commands
ros2 --help
ros2 pkg list

##Run the talker node
ros2 run demo_nodes_cpp talker

##Run the listener node
ros2 run demo_nodes_cpp listener

##Check messages on the topic
ros2 topic echo /chatter

###What I learned
Today, I learned the basic roles of these ROS 2 concepts:

Node: a running program in ROS 2

Talker: a node that publishes messages

Listener: a node that subscribes to messages

Topic: a communication channel between nodes

/chatter: the topic used in this demo example

###Result
I confirmed that:

the talker node published messages,

the listener node received messages,

and the /chatter topic carried the messages correctly.


###Reflection
This was my first hands-on practice with ROS 2 communication.

I did not write ROS 2 code yet, but this session helped me understand the relationship between nodes and topics much more clearly.

###Next Steps
Next, I want to practice:

ros2 node list

ros2 node info

ros2 topic list

ros2 topic info

turtlesim tutorial
