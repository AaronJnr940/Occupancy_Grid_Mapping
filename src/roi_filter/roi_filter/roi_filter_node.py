import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node


class ROIFilterNode(Node):

    def __init__(self):
        super().__init__("roi_filter_node")

        # Dataset parameters
     

        self.declare_parameter(
            "input_dataset",
            os.path.expanduser(
                "~/KITTI_semantic_filtered/dataset"
            )
        )

        self.declare_parameter(
            "output_dataset",
            os.path.expanduser(
                "~/KITTI_roi_filtered/dataset"
            )
        )

        self.declare_parameter("sequence", "00")
        self.declare_parameter("overwrite_existing", False)


        # Main operating ROI
       

        self.declare_parameter("roi_x_min", -10.0)
        self.declare_parameter("roi_x_max", 50.0)

        self.declare_parameter("roi_y_min", -25.0)
        self.declare_parameter("roi_y_max", 25.0)

        self.declare_parameter("roi_z_min", -2.5)
        self.declare_parameter("roi_z_max", 3.0)

        # -------------------------------------------------------------
        # Plane exclusion footprint
        #
        # Disabled initially because the exact plane position relative
        # to the Velodyne sensor must be determined.
        #
        # Example footprint:
        #   25 m in x direction
        #   30 m in y direction
        # -------------------------------------------------------------

        self.declare_parameter(
            "enable_exclusion_mask",
            True
        )

        self.declare_parameter(
            "exclusion_x_min",
            0.0
        )

        self.declare_parameter(
            "exclusion_x_max",
            25.0
        )

        self.declare_parameter(
            "exclusion_y_min",
            -15.0
        )

        self.declare_parameter(
            "exclusion_y_max",
            15.0
        )

        # -------------------------------------------------------------
        # Read parameters
        # -------------------------------------------------------------

        self.input_dataset = Path(
            self.get_parameter("input_dataset").value
        ).expanduser()

        self.output_dataset = Path(
            self.get_parameter("output_dataset").value
        ).expanduser()

        self.sequence = str(
            self.get_parameter("sequence").value
        )

        self.overwrite_existing = bool(
            self.get_parameter("overwrite_existing").value
        )

        self.roi_x_min = float(
            self.get_parameter("roi_x_min").value
        )

        self.roi_x_max = float(
            self.get_parameter("roi_x_max").value
        )

        self.roi_y_min = float(
            self.get_parameter("roi_y_min").value
        )

        self.roi_y_max = float(
            self.get_parameter("roi_y_max").value
        )

        self.roi_z_min = float(
            self.get_parameter("roi_z_min").value
        )

        self.roi_z_max = float(
            self.get_parameter("roi_z_max").value
        )

        self.enable_exclusion_mask = bool(
            self.get_parameter(
                "enable_exclusion_mask"
            ).value
        )

        self.exclusion_x_min = float(
            self.get_parameter(
                "exclusion_x_min"
            ).value
        )

        self.exclusion_x_max = float(
            self.get_parameter(
                "exclusion_x_max"
            ).value
        )

        self.exclusion_y_min = float(
            self.get_parameter(
                "exclusion_y_min"
            ).value
        )

        self.exclusion_y_max = float(
            self.get_parameter(
                "exclusion_y_max"
            ).value
        )

        # -------------------------------------------------------------
        # Input folders
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Output folders
        # -------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Parameter validation
    # -----------------------------------------------------------------

    def validate_parameters(self):

        if self.roi_x_min >= self.roi_x_max:
            raise ValueError(
                "roi_x_min must be smaller than roi_x_max"
            )

        if self.roi_y_min >= self.roi_y_max:
            raise ValueError(
                "roi_y_min must be smaller than roi_y_max"
            )

        if self.roi_z_min >= self.roi_z_max:
            raise ValueError(
                "roi_z_min must be smaller than roi_z_max"
            )

        if self.enable_exclusion_mask:

            if (
                self.exclusion_x_min >=
                self.exclusion_x_max
            ):
                raise ValueError(
                    "exclusion_x_min must be smaller than "
                    "exclusion_x_max"
                )

            if (
                self.exclusion_y_min >=
                self.exclusion_y_max
            ):
                raise ValueError(
                    "exclusion_y_min must be smaller than "
                    "exclusion_y_max"
                )

    # -----------------------------------------------------------------
    # Dataset processing
    # -----------------------------------------------------------------

    def process_dataset(self):

        try:
            self.validate_parameters()

        except ValueError as error:
            self.get_logger().error(str(error))
            return False

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

        bin_files = sorted(
            self.input_velodyne.glob("*.bin")
        )

        if not bin_files:
            self.get_logger().error(
                f"No .bin files found in "
                f"{self.input_velodyne}"
            )
            return False

        self.get_logger().info(
            f"Found {len(bin_files)} input frames"
        )

        self.get_logger().info(
            "ROI limits: "
            f"x=[{self.roi_x_min}, {self.roi_x_max}], "
            f"y=[{self.roi_y_min}, {self.roi_y_max}], "
            f"z=[{self.roi_z_min}, {self.roi_z_max}]"
        )

        if self.enable_exclusion_mask:
            self.get_logger().info(
                "Plane exclusion enabled: "
                f"x=[{self.exclusion_x_min}, "
                f"{self.exclusion_x_max}], "
                f"y=[{self.exclusion_y_min}, "
                f"{self.exclusion_y_max}]"
            )
        else:
            self.get_logger().info(
                "Plane exclusion mask is disabled"
            )

        processed_frames = 0
        skipped_frames = 0
        failed_frames = 0

        total_input_points = 0
        total_roi_removed = 0
        total_plane_removed = 0
        total_retained = 0

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

            temporary_bin_file = Path(
                str(output_bin_file) + ".tmp"
            )

            temporary_label_file = Path(
                str(output_label_file) + ".tmp"
            )

            if not input_label_file.exists():
                self.get_logger().error(
                    f"Missing label for frame {frame_name}"
                )

                failed_frames += 1
                continue

            if (
                output_bin_file.exists() and
                output_label_file.exists() and
                not self.overwrite_existing
            ):
                self.get_logger().warning(
                    f"Frame {frame_name} already exists; "
                    "skipping"
                )

                skipped_frames += 1
                continue

            try:
                # -----------------------------------------------------
                # Load aligned points and labels
                # -----------------------------------------------------

                points_flat = np.fromfile(
                    input_bin_file,
                    dtype=np.float32
                )

                if points_flat.size % 4 != 0:
                    raise ValueError(
                        "Invalid KITTI point-cloud format"
                    )

                points = points_flat.reshape((-1, 4))

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

                input_count = points.shape[0]

                # -----------------------------------------------------
                # Keep points inside the operating ROI
                # -----------------------------------------------------

                roi_mask = (
                    (points[:, 0] >= self.roi_x_min) &
                    (points[:, 0] <= self.roi_x_max) &
                    (points[:, 1] >= self.roi_y_min) &
                    (points[:, 1] <= self.roi_y_max) &
                    (points[:, 2] >= self.roi_z_min) &
                    (points[:, 2] <= self.roi_z_max)
                )

                roi_points = points[roi_mask]
                roi_labels = labels[roi_mask]

                roi_removed_count = (
                    input_count - roi_points.shape[0]
                )

                # -----------------------------------------------------
                # Remove points inside the plane footprint
                # -----------------------------------------------------

                plane_removed_count = 0

                if self.enable_exclusion_mask:

                    inside_plane = (
                        (
                            roi_points[:, 0] >=
                            self.exclusion_x_min
                        ) &
                        (
                            roi_points[:, 0] <=
                            self.exclusion_x_max
                        ) &
                        (
                            roi_points[:, 1] >=
                            self.exclusion_y_min
                        ) &
                        (
                            roi_points[:, 1] <=
                            self.exclusion_y_max
                        )
                    )

                    plane_removed_count = int(
                        np.count_nonzero(inside_plane)
                    )

                    keep_mask = ~inside_plane

                    filtered_points = roi_points[
                        keep_mask
                    ]

                    filtered_labels = roi_labels[
                        keep_mask
                    ]

                else:
                    filtered_points = roi_points
                    filtered_labels = roi_labels

                if (
                    filtered_points.shape[0] !=
                    filtered_labels.shape[0]
                ):
                    raise RuntimeError(
                        "Filtered point and label counts "
                        "do not match"
                    )

                # -----------------------------------------------------
                # Save aligned files atomically
                # -----------------------------------------------------

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

                processed_frames += 1

                total_input_points += input_count
                total_roi_removed += roi_removed_count
                total_plane_removed += plane_removed_count
                total_retained += filtered_points.shape[0]

                self.get_logger().info(
                    f"Frame {frame_name}: "
                    f"{input_count} input -> "
                    f"{filtered_points.shape[0]} retained; "
                    f"{roi_removed_count} outside ROI removed; "
                    f"{plane_removed_count} plane-region "
                    "points removed"
                )

            except Exception as error:

                failed_frames += 1

                self.get_logger().error(
                    f"Failed frame {frame_name}: {error}"
                )

                if temporary_bin_file.exists():
                    temporary_bin_file.unlink()

                if temporary_label_file.exists():
                    temporary_label_file.unlink()

        # -------------------------------------------------------------
        # Final summary
        # -------------------------------------------------------------

        self.get_logger().info(
            "ROI filtering completed"
        )

        self.get_logger().info(
            f"Processed frames: {processed_frames}"
        )

        self.get_logger().info(
            f"Skipped frames: {skipped_frames}"
        )

        self.get_logger().info(
            f"Failed frames: {failed_frames}"
        )

        self.get_logger().info(
            f"Total input points: {total_input_points}"
        )

        self.get_logger().info(
            f"Points removed outside ROI: "
            f"{total_roi_removed}"
        )

        self.get_logger().info(
            f"Points removed by plane mask: "
            f"{total_plane_removed}"
        )

        self.get_logger().info(
            f"Total retained points: {total_retained}"
        )

        return failed_frames == 0


def main(args=None):

    rclpy.init(args=args)

    node = ROIFilterNode()

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