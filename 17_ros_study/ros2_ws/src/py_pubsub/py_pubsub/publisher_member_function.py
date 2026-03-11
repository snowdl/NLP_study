import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# Define a minimal publisher node by inheriting from the ROS 2 Node class
class MinimalPublisher(Node):

    def __init__(self):
        # Initialize the parent Node class and set the node name
        super().__init__('minimal_publisher')

        # Create a publisher that sends String messages to the 'topic' topic
        # The number 10 is the queue size
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Create a timer that calls timer_callback every 0.5 seconds
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Initialize a counter variable
        self.i = 0

    def timer_callback(self):
        # Create a new String message object
        msg = String()

        # Set the message content using the current counter value
        msg.data = f'Hello World: {self.i}'

        # Publish the message to the topic
        self.publisher_.publish(msg)

        # Print a log message to the terminal
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increase the counter after publishing
        self.i += 1


def main(args=None):
    # Initialize the ROS 2 Python client library
    rclpy.init(args=args)

    # Create an instance of the publisher node
    minimal_publisher = MinimalPublisher()

    # Keep the node running so the timer callback can continue executing
    rclpy.spin(minimal_publisher)

    # Clean up the node when the program is stopped
    minimal_publisher.destroy_node()

    # Shut down ROS 2
    rclpy.shutdown()


# Run the main function only when this file is executed directly
if __name__ == '__main__':
    main()