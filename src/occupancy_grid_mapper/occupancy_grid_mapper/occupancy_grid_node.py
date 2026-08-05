import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker


class OccupancyGridNode(Node):

    def __init__(self):
        super().__init__("occupancy_grid_node")

        # Point-cloud input
        self.declare_parameter(
            "input_topic",
            "/patchworkpp/nonground",
        )

        # Mapping/ROI area
        self.declare_parameter("x_min", -10.0)
        self.declare_parameter("x_max", 50.0)
        self.declare_parameter("y_min", -25.0)
        self.declare_parameter("y_max", 25.0)
        self.declare_parameter("z_min", -1.2)
        self.declare_parameter("z_max", 2.5)

        # Grid settings
        self.declare_parameter("resolution", 0.20)
        self.declare_parameter("min_points_per_cell", 2)
        self.declare_parameter("angle_bins", 720)

        # Simulated tug
        self.declare_parameter("tug_length", 6.0)
        self.declare_parameter("tug_width", 3.0)

        # Simulated aircraft
        self.declare_parameter("aircraft_length", 35.0)
        self.declare_parameter("aircraft_width", 30.0)
        self.declare_parameter("hitch_distance", 8.0)
        self.declare_parameter("hitch_angle_deg", 0.0)

        self.input_topic = str(
            self.get_parameter("input_topic").value
        )

        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)
        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)

        self.resolution = float(
            self.get_parameter("resolution").value
        )
        self.min_points = int(
            self.get_parameter("min_points_per_cell").value
        )
        self.angle_bins = int(
            self.get_parameter("angle_bins").value
        )

        self.tug_length = float(
            self.get_parameter("tug_length").value
        )
        self.tug_width = float(
            self.get_parameter("tug_width").value
        )

        self.aircraft_length = float(
            self.get_parameter("aircraft_length").value
        )
        self.aircraft_width = float(
            self.get_parameter("aircraft_width").value
        )
        self.hitch_distance = float(
            self.get_parameter("hitch_distance").value
        )
        self.hitch_angle = math.radians(
            float(self.get_parameter("hitch_angle_deg").value)
        )

        self.validate_parameters()

        self.width = math.ceil(
            (self.x_max - self.x_min) / self.resolution
        )
        self.height = math.ceil(
            (self.y_max - self.y_min) / self.resolution
        )

        self.sensor_x = int(
            (0.0 - self.x_min) / self.resolution
        )
        self.sensor_y = int(
            (0.0 - self.y_min) / self.resolution
        )

        # Aircraft centre relative to the tug.
        self.aircraft_x = (
            self.tug_length / 2.0
            + self.hitch_distance
            + self.aircraft_length / 2.0
        )
        self.aircraft_y = 0.0

        self.self_mask = self.create_self_mask()

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.before_pub = self.create_publisher(
            PointCloud2,
            "/mapping/points_before_mask",
            qos_profile_sensor_data,
        )

        self.after_pub = self.create_publisher(
            PointCloud2,
            "/mapping/points_after_mask",
            qos_profile_sensor_data,
        )

        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            "/occupancy_grid",
            map_qos,
        )

        self.tug_pub = self.create_publisher(
            Marker,
            "/mapping/tug_marker",
            map_qos,
        )

        self.aircraft_pub = self.create_publisher(
            Marker,
            "/mapping/aircraft_marker",
            map_qos,
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Listening to {self.input_topic}"
        )
        self.get_logger().info(
            f"Grid: {self.width} x {self.height}, "
            f"{self.resolution:.2f} m/cell"
        )

    def validate_parameters(self):
        if self.x_min >= self.x_max:
            raise ValueError("Invalid x limits")

        if self.y_min >= self.y_max:
            raise ValueError("Invalid y limits")

        if self.z_min >= self.z_max:
            raise ValueError("Invalid height limits")

        if self.resolution <= 0.0:
            raise ValueError("Resolution must be positive")

        if self.min_points < 1:
            raise ValueError(
                "min_points_per_cell must be at least 1"
            )

    def cloud_callback(self, msg):
        try:
            field_names = {field.name for field in msg.fields}

            if not {"x", "y", "z"}.issubset(field_names):
                self.get_logger().error(
                    "Point cloud does not contain x, y and z fields"
                )
                return

            points = point_cloud2.read_points_numpy(
                msg,
                field_names=["x", "y", "z"],
                skip_nans=True,
            )

            points = np.asarray(
                points,
                dtype=np.float32,
            ).reshape(-1, 3)

            if len(points) == 0:
                return

            # Keep the required region and height.
            roi = (
                (points[:, 0] >= self.x_min)
                & (points[:, 0] < self.x_max)
                & (points[:, 1] >= self.y_min)
                & (points[:, 1] < self.y_max)
                & (points[:, 2] >= self.z_min)
                & (points[:, 2] <= self.z_max)
            )

            before_mask = points[roi]

            # Ignore tug and aircraft regions.
            self_points = (
                self.inside_tug(before_mask)
                | self.inside_aircraft(before_mask)
            )

            after_mask = before_mask[~self_points]

            self.before_pub.publish(
                point_cloud2.create_cloud_xyz32(
                    msg.header,
                    before_mask,
                )
            )

            self.after_pub.publish(
                point_cloud2.create_cloud_xyz32(
                    msg.header,
                    after_mask,
                )
            )

            self.grid_pub.publish(
                self.create_grid(
                    after_mask,
                    msg.header,
                )
            )

            self.publish_markers(msg.header)

            self.get_logger().info(
                f"Input: {len(points)}, "
                f"ROI: {len(before_mask)}, "
                f"mapped: {len(after_mask)}"
            )

        except Exception as error:
            self.get_logger().error(
                f"Point-cloud processing failed: {error}"
            )

    def inside_tug(self, points):
        return (
            (np.abs(points[:, 0]) <= self.tug_length / 2.0)
            & (np.abs(points[:, 1]) <= self.tug_width / 2.0)
        )

    def inside_aircraft(self, points):
        dx = points[:, 0] - self.aircraft_x
        dy = points[:, 1] - self.aircraft_y

        cosine = math.cos(self.hitch_angle)
        sine = math.sin(self.hitch_angle)

        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy

        return (
            (np.abs(local_x) <= self.aircraft_length / 2.0)
            & (np.abs(local_y) <= self.aircraft_width / 2.0)
        )

    def create_self_mask(self):
        x = (
            self.x_min
            + (np.arange(self.width) + 0.5)
            * self.resolution
        )

        y = (
            self.y_min
            + (np.arange(self.height) + 0.5)
            * self.resolution
        )

        grid_x, grid_y = np.meshgrid(x, y)

        tug = (
            (np.abs(grid_x) <= self.tug_length / 2.0)
            & (np.abs(grid_y) <= self.tug_width / 2.0)
        )

        dx = grid_x - self.aircraft_x
        dy = grid_y - self.aircraft_y

        cosine = math.cos(self.hitch_angle)
        sine = math.sin(self.hitch_angle)

        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy

        aircraft = (
            (np.abs(local_x) <= self.aircraft_length / 2.0)
            & (np.abs(local_y) <= self.aircraft_width / 2.0)
        )

        return tug | aircraft

    def create_grid(self, points, header):
        grid = np.full(
            (self.height, self.width),
            -1,
            dtype=np.int8,
        )

        if len(points):
            grid_x = (
                (points[:, 0] - self.x_min)
                / self.resolution
            ).astype(np.int32)

            grid_y = (
                (points[:, 1] - self.y_min)
                / self.resolution
            ).astype(np.int32)

            linear = grid_y * self.width + grid_x

            counts = np.bincount(
                linear,
                minlength=self.width * self.height,
            )

            valid = counts[linear] >= self.min_points

            valid_points = points[valid]
            occupied = np.unique(linear[valid])

            self.mark_free_space(grid, valid_points)

            occupied_y = occupied // self.width
            occupied_x = occupied % self.width

            grid[occupied_y, occupied_x] = 100

        # Tug and aircraft cells stay unknown.
        grid[self.self_mask] = -1

        message = OccupancyGrid()
        message.header = header
        message.info.map_load_time = header.stamp
        message.info.resolution = self.resolution
        message.info.width = self.width
        message.info.height = self.height
        message.info.origin.position.x = self.x_min
        message.info.origin.position.y = self.y_min
        message.info.origin.orientation.w = 1.0
        message.data = grid.flatten().tolist()

        return message

    def mark_free_space(self, grid, points):
        if len(points) == 0:
            return

        angles = np.arctan2(
            points[:, 1],
            points[:, 0],
        )

        angle_ids = (
            ((angles + math.pi) / (2.0 * math.pi))
            * self.angle_bins
        ).astype(np.int32) % self.angle_bins

        distance = (
            points[:, 0] ** 2
            + points[:, 1] ** 2
        )

        order = np.argsort(distance)

        _, first = np.unique(
            angle_ids[order],
            return_index=True,
        )

        nearest_points = points[order[first]]

        for point in nearest_points:
            end_x = int(
                (point[0] - self.x_min)
                / self.resolution
            )

            end_y = int(
                (point[1] - self.y_min)
                / self.resolution
            )

            ray = self.bresenham(
                self.sensor_x,
                self.sensor_y,
                end_x,
                end_y,
            )

            for cell_x, cell_y in ray[:-1]:
                if (
                    0 <= cell_x < self.width
                    and 0 <= cell_y < self.height
                    and not self.self_mask[cell_y, cell_x]
                ):
                    grid[cell_y, cell_x] = 0

    @staticmethod
    def bresenham(x0, y0, x1, y1):
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1

        error = dx - dy

        while True:
            cells.append((x0, y0))

            if x0 == x1 and y0 == y1:
                return cells

            double_error = 2 * error

            if double_error > -dy:
                error -= dy
                x0 += step_x

            if double_error < dx:
                error += dx
                y0 += step_y

    def publish_markers(self, header):
        tug = self.create_marker(
            header,
            marker_id=0,
            centre_x=0.0,
            centre_y=0.0,
            length=self.tug_length,
            width=self.tug_width,
            yaw=0.0,
        )

        aircraft = self.create_marker(
            header,
            marker_id=1,
            centre_x=self.aircraft_x,
            centre_y=self.aircraft_y,
            length=self.aircraft_length,
            width=self.aircraft_width,
            yaw=self.hitch_angle,
        )

        self.tug_pub.publish(tug)
        self.aircraft_pub.publish(aircraft)

    @staticmethod
    def create_marker(
        header,
        marker_id,
        centre_x,
        centre_y,
        length,
        width,
        yaw,
    ):
        marker = Marker()
        marker.header = header
        marker.ns = "vehicle_geometry"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = centre_x
        marker.pose.position.y = centre_y
        marker.pose.position.z = -0.5

        marker.pose.orientation.z = math.sin(yaw / 2.0)
        marker.pose.orientation.w = math.cos(yaw / 2.0)

        marker.scale.x = length
        marker.scale.y = width
        marker.scale.z = 2.0

        marker.color.r = 1.0
        marker.color.g = 0.4
        marker.color.b = 0.1
        marker.color.a = 0.30

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridNode()

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