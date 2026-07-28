import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from sensor_msgs.msg import PointCloud2


class CloudExporterNode(Node):

    def __init__(self):
        super().__init__("cloud_exporter_node")
        
        # Parameters

        self.declare_parameter(
            "dataset_path",
            os.path.expanduser("~/KITTI/dataset")
        )

        self.declare_parameter(
            "output_path",
            os.path.expanduser("~/KITTI_filtered/dataset")
        )

        self.declare_parameter(
            "sequence",
            "00"
        )

        self.declare_parameter(
            "overwrite_existing",
            False
        )

        self.dataset_path = Path(
            self.get_parameter("dataset_path").value
        ).expanduser()

        self.output_path = Path(
            self.get_parameter("output_path").value
        ).expanduser()

        self.sequence = str(
            self.get_parameter("sequence").value
        )

        self.overwrite_existing = bool(
            self.get_parameter("overwrite_existing").value
        )

      
        # Input folders
        

        self.input_velodyne_path = (
            self.dataset_path /
            self.sequence /
            "velodyne"
        )

        self.input_labels_path = (
            self.dataset_path /
            self.sequence /
            "labels"
        )

        
        # Output folders

        self.output_velodyne_path = (
            self.output_path /
            self.sequence /
            "velodyne"
        )

        self.output_labels_path = (
            self.output_path /
            self.sequence /
            "labels"
        )

        self.output_velodyne_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_labels_path.mkdir(
            parents=True,
            exist_ok=True
        )

   
        # Validate source folders
     

        if not self.input_velodyne_path.exists():
            raise RuntimeError(
                f"Input Velodyne folder does not exist: "
                f"{self.input_velodyne_path}"
            )

        if not self.input_labels_path.exists():
            raise RuntimeError(
                f"Input label folder does not exist: "
                f"{self.input_labels_path}"
            )

        # Messages are paired using their exact ROS timestamps.
        self.nonground_cache = {}
        self.indices_cache = {}

        # Patchwork++ publishes using reliable, transient-local QoS.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.nonground_subscription = self.create_subscription(
            PointCloud2,
            "/patchworkpp/nonground",
            self.nonground_callback,
            qos
        )

        self.indices_subscription = self.create_subscription(
            PointCloud2,
            "/patchworkpp/nonground_indices",
            self.indices_callback,
            qos
        )

        self.get_logger().info(
            "Index-based cloud exporter started"
        )

        self.get_logger().info(
            f"Input Velodyne: {self.input_velodyne_path}"
        )

        self.get_logger().info(
            f"Input labels: {self.input_labels_path}"
        )

        self.get_logger().info(
            f"Output Velodyne: {self.output_velodyne_path}"
        )

        self.get_logger().info(
            f"Output labels: {self.output_labels_path}"
        )

    # Message synchronization
 

    @staticmethod
    def stamp_key(msg):
        """
        Create a unique synchronization key from the ROS timestamp.
        """

        return (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec)
        )

    def nonground_callback(self, msg):
        key = self.stamp_key(msg)

        self.nonground_cache[key] = msg

        self.try_process_pair(key)
        self.trim_caches()

    def indices_callback(self, msg):
        key = self.stamp_key(msg)

        self.indices_cache[key] = msg

        self.try_process_pair(key)
        self.trim_caches()

    def try_process_pair(self, key):
      
        #Export only when the non-ground cloud and its original indices have exactly the same timestamp.
       

        if key not in self.nonground_cache:
            return

        if key not in self.indices_cache:
            return

        nonground_msg = self.nonground_cache.pop(key)
        indices_msg = self.indices_cache.pop(key)

        self.export_frame(
            nonground_msg,
            indices_msg
        )

    def trim_caches(self):

        maximum_cached_messages = 20

        while len(self.nonground_cache) > maximum_cached_messages:
            oldest_key = next(iter(self.nonground_cache))
            self.nonground_cache.pop(oldest_key)

            self.get_logger().warning(
                f"Discarded unmatched non-ground message: {oldest_key}"
            )

        while len(self.indices_cache) > maximum_cached_messages:
            oldest_key = next(iter(self.indices_cache))
            self.indices_cache.pop(oldest_key)

            self.get_logger().warning(
                f"Discarded unmatched index message: {oldest_key}"
            )

    # Read index message
   

    @staticmethod
    def read_indices(indices_msg):
        
        #Read the UINT32 index field published by Patchwork++.
        

        expected_field = None

        for field in indices_msg.fields:
            if field.name == "index":
                expected_field = field
                break

        if expected_field is None:
            raise ValueError(
                "The nonground_indices message has no 'index' field."
            )

        if expected_field.datatype != expected_field.UINT32:
            raise ValueError(
                "The nonground_indices 'index' field is not UINT32."
            )

        if indices_msg.point_step != 4:
            raise ValueError(
                "Unexpected nonground_indices point_step: "
                f"{indices_msg.point_step}. Expected 4."
            )

        point_count = (
            int(indices_msg.width) *
            int(indices_msg.height)
        )

        expected_bytes = (
            point_count *
            int(indices_msg.point_step)
        )

        raw_data = bytes(indices_msg.data)

        if len(raw_data) < expected_bytes:
            raise ValueError(
                "Index message contains insufficient data: "
                f"{len(raw_data)} bytes received, "
                f"{expected_bytes} bytes expected."
            )

        # Patchwork++ publishes little-endian UINT32 indices.
        return np.frombuffer(
            raw_data,
            dtype="<u4",
            count=point_count
        ).copy()

   
    # Frame export
    

    def export_frame(
        self,
        nonground_msg,
        indices_msg
    ):

        # Confirm that both messages belong to exactly the same frame.
        nonground_stamp = self.stamp_key(nonground_msg)
        indices_stamp = self.stamp_key(indices_msg)

        if nonground_stamp != indices_stamp:
            self.get_logger().error(
                "Refusing to export messages with different timestamps: "
                f"{nonground_stamp} and {indices_stamp}"
            )
            return

        # The publisher stores the KITTI frame number in nanosec.
        frame_number = int(
            indices_msg.header.stamp.nanosec
        )

        frame_name = f"{frame_number:06d}"

        self.get_logger().info(
            f"Processing frame {frame_name}"
        )

        input_bin_file = (
            self.input_velodyne_path /
            f"{frame_name}.bin"
        )

        input_label_file = (
            self.input_labels_path /
            f"{frame_name}.label"
        )

        output_bin_file = (
            self.output_velodyne_path /
            f"{frame_name}.bin"
        )

        output_label_file = (
            self.output_labels_path /
            f"{frame_name}.label"
        )

     
        # Check source files
       

        if not input_bin_file.exists():
            self.get_logger().error(
                f"Original point-cloud file does not exist: "
                f"{input_bin_file}"
            )
            return

        if not input_label_file.exists():
            self.get_logger().error(
                f"Original label file does not exist: "
                f"{input_label_file}"
            )
            return

        # Skip complete output frames unless overwriting is requested.
        if (
            output_bin_file.exists() and
            output_label_file.exists() and
            not self.overwrite_existing
        ):
            self.get_logger().warning(
                f"Frame {frame_name} already exists; skipping"
            )
            return

        try:
            
            # Load original KITTI points
            

            original_points_flat = np.fromfile(
                input_bin_file,
                dtype=np.float32
            )

            if original_points_flat.size % 4 != 0:
                raise ValueError(
                    f"{input_bin_file} is not a valid KITTI "
                    "x, y, z, intensity file."
                )

            original_points = original_points_flat.reshape(
                (-1, 4)
            )

          
            # Load original labels
            

            original_labels = np.fromfile(
                input_label_file,
                dtype=np.uint32
            )

            if original_labels.shape[0] != original_points.shape[0]:
                raise ValueError(
                    "Original point/label count mismatch: "
                    f"{original_points.shape[0]} points and "
                    f"{original_labels.shape[0]} labels."
                )

            
            # Read exact original point indices
            

            nonground_indices = self.read_indices(
                indices_msg
            )

            published_nonground_count = (
                int(nonground_msg.width) *
                int(nonground_msg.height)
            )

            if (
                nonground_indices.shape[0] !=
                published_nonground_count
            ):
                raise ValueError(
                    "Patchwork++ output count mismatch: "
                    f"{published_nonground_count} non-ground points and "
                    f"{nonground_indices.shape[0]} indices."
                )

           
            # Validate indices
           

            if nonground_indices.size > 0:
                largest_index = int(
                    nonground_indices.max()
                )

                if largest_index >= original_points.shape[0]:
                    raise IndexError(
                        f"Invalid point index {largest_index}. "
                        f"Original frame {frame_name} contains only "
                        f"{original_points.shape[0]} points."
                    )

          
            # Filter points and labels using exact indices
            

            filtered_points = original_points[
                nonground_indices
            ]

            filtered_labels = original_labels[
                nonground_indices
            ]

            if filtered_points.shape[0] != filtered_labels.shape[0]:
                raise RuntimeError(
                    "Filtered point and label counts do not match."
                )

           
            # Save atomically
            

            temporary_bin_file = Path(
                str(output_bin_file) + ".tmp"
            )

            temporary_label_file = Path(
                str(output_label_file) + ".tmp"
            )

            filtered_points.astype(
                np.float32
            ).tofile(
                temporary_bin_file
            )

            filtered_labels.astype(
                np.uint32
            ).tofile(
                temporary_label_file
            )

            os.replace(
                temporary_bin_file,
                output_bin_file
            )

            os.replace(
                temporary_label_file,
                output_label_file
            )

            self.get_logger().info(
                f"Saved frame {frame_name}: "
                f"{original_points.shape[0]} original points -> "
                f"{filtered_points.shape[0]} non-ground points and labels"
            )

        except Exception as error:
            self.get_logger().error(
                f"Failed to export frame {frame_name}: {error}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = CloudExporterNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(f"Cloud exporter failed: {error}")

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
