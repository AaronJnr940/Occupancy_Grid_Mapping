import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker


class ExclusionMarkerNode(Node):

    def __init__(self):
        super().__init__("exclusion_marker_node")

        self.declare_parameter("frame_id", "velodyne")

        self.declare_parameter("x_min", 0.0)
        self.declare_parameter("x_max", 25.0)

        self.declare_parameter("y_min", -15.0)
        self.declare_parameter("y_max", 15.0)

        self.frame_id = str(
            self.get_parameter("frame_id").value
        )

        self.x_min = float(
            self.get_parameter("x_min").value
        )

        self.x_max = float(
            self.get_parameter("x_max").value
        )

        self.y_min = float(
            self.get_parameter("y_min").value
        )

        self.y_max = float(
            self.get_parameter("y_max").value
        )

        self.publisher = self.create_publisher(
            Marker,
            "/plane_exclusion_marker",
            10
        )

        self.timer = self.create_timer(
            0.5,
            self.publish_marker
        )

    def publish_marker(self):

        marker = Marker()

        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "plane_exclusion"
        marker.id = 0

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Rectangle center
        marker.pose.position.x = (
            self.x_min + self.x_max
        ) / 2.0

        marker.pose.position.y = (
            self.y_min + self.y_max
        ) / 2.0

        marker.pose.position.z = 0.0

        marker.pose.orientation.w = 1.0

        # Rectangle dimensions
        marker.scale.x = (
            self.x_max - self.x_min
        )

        marker.scale.y = (
            self.y_max - self.y_min
        )

        marker.scale.z = 0.10

        # Semi-transparent marker
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.25

        self.publisher.publish(marker)


def main(args=None):

    rclpy.init(args=args)

    node = ExclusionMarkerNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()