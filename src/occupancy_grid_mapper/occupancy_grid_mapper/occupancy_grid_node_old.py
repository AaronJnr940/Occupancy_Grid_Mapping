import math
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node 
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker


class OccupancyGridNode(Node):

    def __init__(self):
        super().__init__("occupancy_grid_node")

        # Dataset playback
        self.declare_parameter(
            "dataset_path",
            "/home/leon/KITTI_filtered/dataset",
        )
        self.declare_parameter("sequence", "00")
        self.declare_parameter("rate_hz", 1.0)
        self.declare_parameter("loop", True)

        # ROI and obstacle height
        self.declare_parameter("x_min", -10.0)
        self.declare_parameter("x_max", 50.0)
        self.declare_parameter("y_min", -25.0)
        self.declare_parameter("y_max", 25.0)
        self.declare_parameter("obstacle_z_min", -1.2)
        self.declare_parameter("obstacle_z_max", 2.5)

        # Grid
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

        self.dataset_path = Path(
            self.get_parameter("dataset_path").value
        )
        self.sequence = str(
            self.get_parameter("sequence").value
        )
        self.rate_hz = float(
            self.get_parameter("rate_hz").value
        )
        self.loop = bool(
            self.get_parameter("loop").value
        )

        self.x_min = float(self.get_parameter("x_min").value)
        self.x_max = float(self.get_parameter("x_max").value)
        self.y_min = float(self.get_parameter("y_min").value)
        self.y_max = float(self.get_parameter("y_max").value)

        self.z_min = float(
            self.get_parameter("obstacle_z_min").value
        )
        self.z_max = float(
            self.get_parameter("obstacle_z_max").value
        )

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

        self.velodyne_dir = (
            self.dataset_path
            / self.sequence
            / "velodyne"
        )
        self.label_dir = (
            self.dataset_path
            / self.sequence
            / "labels"
        )

        self.frames = sorted(self.velodyne_dir.glob("*.bin"))

        if not self.frames:
            raise RuntimeError(
                f"No KITTI frames found in {self.velodyne_dir}"
            )

        self.width = int(
            math.ceil(
                (self.x_max - self.x_min) / self.resolution
            )
        )
        self.height = int(
            math.ceil(
                (self.y_max - self.y_min) / self.resolution
            )
        )

        self.sensor_x = int(
            (0.0 - self.x_min) / self.resolution
        )
        self.sensor_y = int(
            (0.0 - self.y_min) / self.resolution
        )

        # Aircraft centre is placed in front of the tug.
        self.aircraft_centre_x = (
            self.tug_length / 2.0
            + self.hitch_distance
            + self.aircraft_length / 2.0
        )
        self.aircraft_centre_y = 0.0

        self.self_mask = self.create_self_mask()

        point_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # QOS can be changed to best effort
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.before_pub = self.create_publisher(
            PointCloud2,
            "/simplified/points_before_mask",
            point_qos,
        )
        self.after_pub = self.create_publisher(
            PointCloud2,
            "/simplified/points_after_mask",
            point_qos,
        )
        self.grid_pub = self.create_publisher(
            OccupancyGrid,
            "/simplified/occupancy_grid",
            map_qos,
        )
        self.tug_pub = self.create_publisher(
            Marker,
            "/simplified/tug_marker",
            map_qos,
        )
        self.aircraft_pub = self.create_publisher(
            Marker,
            "/simplified/aircraft_marker",
            map_qos,
        )

        self.frame_index = 0

        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self.process_next_frame,
        )

        self.get_logger().info(
            f"Loaded {len(self.frames)} KITTI frames"
        )
        self.get_logger().info(
            f"Grid: {self.width} x {self.height}, "
            f"{self.resolution:.2f} m/cell"
        )

    def validate_parameters(self):
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")

        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

        if self.x_min >= self.x_max:
            raise ValueError("Invalid x limits")

        if self.y_min >= self.y_max:
            raise ValueError("Invalid y limits")

        if self.z_min >= self.z_max:
            raise ValueError("Invalid obstacle height limits")

    def process_next_frame(self):
        if self.frame_index >= len(self.frames):
            if not self.loop:
                self.timer.cancel()
                return

            self.frame_index = 0

        bin_path = self.frames[self.frame_index]
        label_path = self.label_dir / f"{bin_path.stem}.label"

        points = np.fromfile(
            bin_path,
            dtype=np.float32,
        ).reshape(-1, 4)

        # Remove selected object classes when labels exist.
        if label_path.exists():
            labels = np.fromfile(
                label_path,
                dtype=np.uint32,
            )

            count = min(len(points), len(labels))
            points = points[:count]
            labels = labels[:count]

            semantic_ids = labels & 0xFFFF

            remove_classes = np.array(
                [13, 18, 257, 258],
                dtype=np.uint32,
            )

            points = points[
                ~np.isin(semantic_ids, remove_classes)
            ]

        # Keep only the required ROI and obstacle height.
        roi_mask = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] < self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] < self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] <= self.z_max)
        )

        before_mask = points[roi_mask]

        # Remove simulated tug and aircraft points.
        tug_mask = self.points_inside_tug(before_mask)
        aircraft_mask = self.points_inside_aircraft(before_mask)

        after_mask = before_mask[
            ~(tug_mask | aircraft_mask)
        ]

        stamp = self.get_clock().now().to_msg()

        self.before_pub.publish(
            self.make_cloud(before_mask, stamp)
        )
        self.after_pub.publish(
            self.make_cloud(after_mask, stamp)
        )

        self.grid_pub.publish(
            self.make_grid(after_mask, stamp)
        )

        self.publish_markers(stamp)

        self.get_logger().info(
            f"{bin_path.name}: "
            f"{len(before_mask)} before mask, "
            f"{len(after_mask)} after mask"
        )

        self.frame_index += 1

    def points_inside_tug(self, points):
        return (
            (np.abs(points[:, 0]) <= self.tug_length / 2.0)
            & (np.abs(points[:, 1]) <= self.tug_width / 2.0)
        )

    def points_inside_aircraft(self, points):
        dx = points[:, 0] - self.aircraft_centre_x
        dy = points[:, 1] - self.aircraft_centre_y

        cosine = math.cos(self.hitch_angle)
        sine = math.sin(self.hitch_angle)

        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy

        return (
            (np.abs(local_x) <= self.aircraft_length / 2.0)
            & (np.abs(local_y) <= self.aircraft_width / 2.0)
        )

    def create_self_mask(self):
        x_values = (
            self.x_min
            + (np.arange(self.width) + 0.5)
            * self.resolution
        )
        y_values = (
            self.y_min
            + (np.arange(self.height) + 0.5)
            * self.resolution
        )

        grid_x, grid_y = np.meshgrid(x_values, y_values)

        tug = (
            (np.abs(grid_x) <= self.tug_length / 2.0)
            & (np.abs(grid_y) <= self.tug_width / 2.0)
        )

        dx = grid_x - self.aircraft_centre_x
        dy = grid_y - self.aircraft_centre_y

        cosine = math.cos(self.hitch_angle)
        sine = math.sin(self.hitch_angle)

        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy

        aircraft = (
            (np.abs(local_x) <= self.aircraft_length / 2.0)
            & (np.abs(local_y) <= self.aircraft_width / 2.0)
        )

        return tug | aircraft

    def make_grid(self, points, stamp):
        grid = np.full(
            (self.height, self.width),
            -1,
            dtype=np.int8,
        )

        if len(points) > 0:
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

            valid_cells = counts[linear] >= self.min_points
            valid_points = points[valid_cells]
            valid_linear = linear[valid_cells]

            occupied = np.unique(valid_linear)

            # Use the nearest obstacle in each angular direction.
            if len(valid_points) > 0:
                angles = np.arctan2(
                    valid_points[:, 1],
                    valid_points[:, 0],
                )

                angle_index = (
                    (
                        (angles + math.pi)
                        / (2.0 * math.pi)
                    )
                    * self.angle_bins
                ).astype(np.int32) % self.angle_bins

                distance = (
                    valid_points[:, 0] ** 2
                    + valid_points[:, 1] ** 2
                )

                order = np.argsort(distance)
                sorted_angles = angle_index[order]

                _, first = np.unique(
                    sorted_angles,
                    return_index=True,
                )

                nearest = valid_points[order[first]]

                for point in nearest:
                    end_x = int(
                        (point[0] - self.x_min)
                        / self.resolution
                    )
                    end_y = int(
                        (point[1] - self.y_min)
                        / self.resolution
                    )

                    for cell_x, cell_y in self.bresenham(
                        self.sensor_x,
                        self.sensor_y,
                        end_x,
                        end_y,
                    )[:-1]:
                        if (
                            0 <= cell_x < self.width
                            and 0 <= cell_y < self.height
                        ):
                            grid[cell_y, cell_x] = 0

            occupied_y = occupied // self.width
            occupied_x = occupied % self.width
            grid[occupied_y, occupied_x] = 100

        # Tug and aircraft regions remain unknown.
        grid[self.self_mask] = -1

        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = "velodyne"

        message.info.map_load_time = stamp
        message.info.resolution = self.resolution
        message.info.width = self.width
        message.info.height = self.height
        message.info.origin.position.x = self.x_min
        message.info.origin.position.y = self.y_min
        message.info.origin.orientation.w = 1.0

        message.data = grid.flatten().tolist()

        return message

    @staticmethod
    def bresenham(x0, y0, x1, y1):
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx - dy

        while True:
            cells.append((x0, y0))

            if x0 == x1 and y0 == y1:
                return cells

            double_error = 2 * error

            if double_error > -dy:
                error -= dy
                x0 += sx

            if double_error < dx:
                error += dx
                y0 += sy

    @staticmethod
    def make_cloud(points, stamp):
        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = "velodyne"
        message.height = 1
        message.width = len(points)

        message.fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="intensity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]

        message.is_bigendian = False
        message.point_step = 16
        message.row_step = 16 * len(points)
        message.is_dense = False
        message.data = np.asarray(
            points,
            dtype=np.float32,
        ).tobytes()

        return message

    def publish_markers(self, stamp):
        tug = self.make_marker(
            stamp=stamp,
            marker_id=0,
            length=self.tug_length,
            width=self.tug_width,
            centre_x=0.0,
            centre_y=0.0,
            yaw=0.0,
            red=0.1,
            green=0.4,
            blue=1.0,
        )

        aircraft = self.make_marker(
            stamp=stamp,
            marker_id=1,
            length=self.aircraft_length,
            width=self.aircraft_width,
            centre_x=self.aircraft_centre_x,
            centre_y=self.aircraft_centre_y,
            yaw=self.hitch_angle,
            red=1.0,
            green=0.4,
            blue=0.1,
        )

        self.tug_pub.publish(tug)
        self.aircraft_pub.publish(aircraft)

    @staticmethod
    def make_marker(
        stamp,
        marker_id,
        length,
        width,
        centre_x,
        centre_y,
        yaw,
        red,
        green,
        blue,
    ):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "velodyne"
        marker.ns = "simulated_geometry"
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

        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
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
