import os
import numpy as np

import rclpy
from rclpy.node import Node


class DatasetReader(Node):

    def __init__(self):
        super().__init__("dataset_reader")

        # ROS parameters
        self.declare_parameter(
            "dataset_path",
            os.path.expanduser("~/KITTI/dataset")
        )
        self.declare_parameter("sequence", "00")
        self.declare_parameter("frame", "000000")

        dataset_path = self.get_parameter("dataset_path").value
        sequence = self.get_parameter("sequence").value
        frame = self.get_parameter("frame").value

        file_path = os.path.join(
            dataset_path,
            sequence,
            "velodyne",
            frame + ".bin"
        )

        self.get_logger().info(f"Loading: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")

        # Read KITTI point cloud
        points = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))
        xyz = points[:, :3]

        self.get_logger().info(f"Loaded {points.shape[0]} points")
        self.get_logger().info(
            f"X: {xyz[:, 0].min():.2f} -> {xyz[:, 0].max():.2f}"
        )
        self.get_logger().info(
            f"Y: {xyz[:, 1].min():.2f} -> {xyz[:, 1].max():.2f}"
        )
        self.get_logger().info(
            f"Z: {xyz[:, 2].min():.2f} -> {xyz[:, 2].max():.2f}"
        )


def main(args=None):
    rclpy.init(args=args)

    try:
        node = DatasetReader()
        rclpy.spin(node)

    except FileNotFoundError as e:
        print(e)

    except KeyboardInterrupt:
        pass

    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()