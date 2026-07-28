import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node


class SemanticFilterNode(Node):

    def __init__(self):
        super().__init__("semantic_filter_node")

        # Nonground input dataset
        self.declare_parameter(
            "input_dataset",
            os.path.expanduser("~/KITTI_filtered/dataset")
        )

        # Semantic-filtered/other objects output dataset
        self.declare_parameter(
            "output_dataset",
            os.path.expanduser("~/KITTI_semantic_filtered/dataset")
        )

        self.declare_parameter(
            "sequence",
            "00"
        )

        # SemanticKITTI class IDs removed by default:
        # 13  = bus
        # 18  = truck
        # 257 = moving-bus
        # 258 = moving-truck
        self.declare_parameter(
            "remove_class_ids",
            [13, 18, 257, 258]
        )

        self.declare_parameter(
            "overwrite_existing",
            False
        )

        self.input_dataset = Path(
            self.get_parameter("input_dataset").value
        ).expanduser()

        self.output_dataset = Path(
            self.get_parameter("output_dataset").value
        ).expanduser()

        self.sequence = str(
            self.get_parameter("sequence").value
        )

        self.remove_class_ids = np.asarray(
            self.get_parameter("remove_class_ids").value,
            dtype=np.uint32
        )

        self.overwrite_existing = bool(
            self.get_parameter("overwrite_existing").value
        )

        # Input folders
        self.input_velodyne = (
            self.input_dataset /
            self.sequence /
            "velodyne"
        )

        self.input_labels = (
            self.input_dataset /
            self.sequence /
            "labels"
        )

        # Output folders
        self.output_velodyne = (
            self.output_dataset /
            self.sequence /
            "velodyne"
        )

        self.output_labels = (
            self.output_dataset /
            self.sequence /
            "labels"
        )

        self.output_velodyne.mkdir(
            parents=True,
            exist_ok=True
        )

        self.output_labels.mkdir(
            parents=True,
            exist_ok=True
        )

    def process_dataset(self):

        if not self.input_velodyne.exists():
            self.get_logger().error(
                f"Input Velodyne folder does not exist: "
                f"{self.input_velodyne}"
            )
            return False

        if not self.input_labels.exists():
            self.get_logger().error(
                f"Input labels folder does not exist: "
                f"{self.input_labels}"
            )
            return False

        # Process actual filenames rather than assuming continuous frames.
        bin_files = sorted(
            self.input_velodyne.glob("*.bin")
        )

        if not bin_files:
            self.get_logger().error(
                f"No .bin files found in {self.input_velodyne}"
            )
            return False

        self.get_logger().info(
            f"Found {len(bin_files)} input frames"
        )

        self.get_logger().info(
            "Removing SemanticKITTI class IDs: "
            f"{self.remove_class_ids.tolist()}"
        )

        processed_frames = 0
        skipped_frames = 0
        failed_frames = 0

        total_input_points = 0
        total_removed_points = 0
        total_saved_points = 0

        for input_bin_file in bin_files:

            frame_name = input_bin_file.stem

            input_label_file = (
                self.input_labels /
                f"{frame_name}.label"
            )

            output_bin_file = (
                self.output_velodyne /
                f"{frame_name}.bin"
            )

            output_label_file = (
                self.output_labels /
                f"{frame_name}.label"
            )

            if not input_label_file.exists():
                self.get_logger().error(
                    f"Missing label file for frame {frame_name}: "
                    f"{input_label_file}"
                )

                failed_frames += 1
                continue

            if (
                output_bin_file.exists() and
                output_label_file.exists() and
                not self.overwrite_existing
            ):
                self.get_logger().warning(
                    f"Frame {frame_name} already exists; skipping"
                )

                skipped_frames += 1
                continue

            try:
                # KITTI points are x, y, z, intensity.
                points_flat = np.fromfile(
                    input_bin_file,
                    dtype=np.float32
                )

                if points_flat.size % 4 != 0:
                    raise ValueError(
                        "Point file does not contain valid "
                        "x, y, z, intensity rows"
                    )

                points = points_flat.reshape(
                    (-1, 4)
                )

                # SemanticKITTI labels are uint32.
                labels = np.fromfile(
                    input_label_file,
                    dtype=np.uint32
                )

                if points.shape[0] != labels.shape[0]:
                    raise ValueError(
                        "Point/label count mismatch: "
                        f"{points.shape[0]} points and "
                        f"{labels.shape[0]} labels"
                    )

                # Lower 16 bits contain the semantic class ID.
                # Upper 16 bits contain the instance information.
                semantic_ids = (
                    labels & np.uint32(0xFFFF)
                )

                remove_mask = np.isin(
                    semantic_ids,
                    self.remove_class_ids
                )

                keep_mask = ~remove_mask

                filtered_points = points[
                    keep_mask
                ]

                # Preserve the original full uint32 label, including
                # instance information, for every retained point.
                filtered_labels = labels[
                    keep_mask
                ]

                removed_count = int(
                    np.count_nonzero(remove_mask)
                )

                if (
                    filtered_points.shape[0] !=
                    filtered_labels.shape[0]
                ):
                    raise RuntimeError(
                        "Filtered point and label counts do not match"
                    )

                # Write temporary files first.
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

                # Replace final files only after both writes succeed.
                os.replace(
                    temporary_bin_file,
                    output_bin_file
                )

                os.replace(
                    temporary_label_file,
                    output_label_file
                )

                total_input_points += points.shape[0]
                total_removed_points += removed_count
                total_saved_points += filtered_points.shape[0]
                processed_frames += 1

                self.get_logger().info(
                    f"Frame {frame_name}: "
                    f"{points.shape[0]} input -> "
                    f"{filtered_points.shape[0]} retained; "
                    f"{removed_count} semantic points removed"
                )

            except Exception as error:
                failed_frames += 1

                self.get_logger().error(
                    f"Failed frame {frame_name}: {error}"
                )

                # Remove incomplete temporary files.
                temporary_bin_file = Path(
                    str(output_bin_file) + ".tmp"
                )

                temporary_label_file = Path(
                    str(output_label_file) + ".tmp"
                )

                if temporary_bin_file.exists():
                    temporary_bin_file.unlink()

                if temporary_label_file.exists():
                    temporary_label_file.unlink()

        self.get_logger().info(
            "Semantic filtering completed"
        )

        self.get_logger().info(
            f"Processed frames: {processed_frames}"
        )

        self.get_logger().info(
            f"Skipped existing frames: {skipped_frames}"
        )

        self.get_logger().info(
            f"Failed frames: {failed_frames}"
        )

        self.get_logger().info(
            f"Total input points: {total_input_points}"
        )

        self.get_logger().info(
            f"Total removed points: {total_removed_points}"
        )

        self.get_logger().info(
            f"Total retained points: {total_saved_points}"
        )

        return failed_frames == 0


def main(args=None):
    rclpy.init(args=args)

    node = SemanticFilterNode()

    try:
        node.process_dataset()

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()