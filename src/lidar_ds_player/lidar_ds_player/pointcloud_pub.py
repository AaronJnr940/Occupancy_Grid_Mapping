import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class PointCloudPublisher(Node):

    def __init__(self):
        super().__init__("pointcloud_pub")

        # Parameters

        self.declare_parameter(
            "dataset_path",
            os.path.expanduser("~/KITTI_roi_filtered/dataset")
        )

        self.declare_parameter(
            "sequence",
            "00"
        )

        # Publish files whose frame number is greater than or equal to this value.
        self.declare_parameter(
            "start_frame",
            0
        )

        self.declare_parameter(
            "rate_hz",
            1.0
        )

        self.dataset_path = os.path.expanduser(
            self.get_parameter("dataset_path").value
        )

        self.sequence = str(
            self.get_parameter("sequence").value
        )

        self.start_frame = int(
            self.get_parameter("start_frame").value
        )

        self.rate_hz = float(
            self.get_parameter("rate_hz").value
        )

        if self.rate_hz <= 0.0:
            raise ValueError(
                "rate_hz must be greater than zero"
            )

        # Dataset path
        self.velodyne_path = os.path.join(
            self.dataset_path,
            self.sequence,
            "velodyne"
        )

        if not os.path.isdir(self.velodyne_path):
            raise RuntimeError(
                f"Dataset folder does not exist: "
                f"{self.velodyne_path}"
            )

        # Find all available .bin files

        self.bin_files = []

        for filename in os.listdir(self.velodyne_path):

            if not filename.endswith(".bin"):
                continue

            frame_text = os.path.splitext(filename)[0]

            try:
                frame_number = int(frame_text)
            except ValueError:
                self.get_logger().warning(
                    f"Ignoring invalid filename: {filename}"
                )
                continue

            if frame_number >= self.start_frame:
                self.bin_files.append(
                    (frame_number, filename)
                )

        # Sort numerically, not alphabetically.
        self.bin_files.sort(
            key=lambda item: item[0]
        )

        if not self.bin_files:
            raise RuntimeError(
                f"No .bin files found in {self.velodyne_path} "
                f"from frame {self.start_frame:06d}"
            )

        self.file_index = 0

        # Publisher

        self.publisher = self.create_publisher(
            PointCloud2,
            "/points_raw",
            qos_profile_sensor_data
        )

        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self.publish_frame
        )

        self.get_logger().info(
            "PointCloud publisher started"
        )

        self.get_logger().info(
            f"Dataset: {self.velodyne_path}"
        )

        self.get_logger().info(
            f"Available frames: {len(self.bin_files)}"
        )

        self.get_logger().info(
            f"First frame: {self.bin_files[0][0]:06d}"
        )

        self.get_logger().info(
            f"Last frame: {self.bin_files[-1][0]:06d}"
        )

        self.get_logger().info(
            f"Publishing rate: {self.rate_hz:.2f} Hz"
        )

    # Publish one frame

    def publish_frame(self):

        if self.file_index >= len(self.bin_files):

            self.get_logger().info(
                "Finished sequence"
            )

            self.timer.cancel()
            return

        frame_number, filename = self.bin_files[
            self.file_index
        ]

        filepath = os.path.join(
            self.velodyne_path,
            filename
        )

        try:
            points_flat = np.fromfile(
                filepath,
                dtype=np.float32
            )

            if points_flat.size % 4 != 0:
                self.get_logger().error(
                    f"Skipping invalid KITTI file {filename}: "
                    f"{points_flat.size} float values is not "
                    "divisible by 4"
                )

                self.file_index += 1
                return

            points = points_flat.reshape(
                (-1, 4)
            )

            msg = self.create_cloud(
                points,
                frame_number
            )

            self.publisher.publish(msg)

            self.get_logger().info(
                f"Published frame {filename} "
                f"({points.shape[0]} points)"
            )

        except Exception as error:
            self.get_logger().error(
                f"Failed to publish {filename}: {error}"
            )

        self.file_index += 1

    # Convert NumPy points to PointCloud2

    def create_cloud(
        self,
        points,
        frame_number
    ):

        header = Header()

        # Preserve the real clock seconds, but store the KITTI frame number in nanosec for the index-based exporter.
        stamp = self.get_clock().now().to_msg()
        stamp.nanosec = int(frame_number)

        header.stamp = stamp
        header.frame_id = "velodyne"

        fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),

            # Filtered KITTI files still preserve the original intensity value.
            PointField(
                name="intensity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            )
        ]

        points = np.ascontiguousarray(
            points,
            dtype=np.float32
        )

        cloud = PointCloud2()

        cloud.header = header
        cloud.height = 1
        cloud.width = points.shape[0]
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 16
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = points.tobytes()
        cloud.is_dense = bool(
            np.isfinite(points[:, :3]).all()
        )

        return cloud


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = PointCloudPublisher()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(f"PointCloud publisher failed: {error}")

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
