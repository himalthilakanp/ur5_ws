#!/usr/bin/env python3

import rclpy
import time
import math

from rclpy.node import Node
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    CollisionObject,
    PlanningScene,
    ObjectColor,
    AttachedCollisionObject,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy


# ---------------------------------------------------------------------------
# JOINT LIMITS  (from URDF)
# ---------------------------------------------------------------------------
JOINT_LIMITS = {
    "ROT_1":   (-3.14,  3.14),
    "PITCH_1": (-1.57,  1.57),
    "PITCH_2": (-3.14,  3.14),
    "PITCH_3": (-3.14,  3.14),
    "ROT_2":   (-3.14,  3.14),
    "ROT_3":   (-6.28,  6.28),
}

ARM_JOINT_NAMES = [
    "ROT_1",
    "PITCH_1",
    "PITCH_2",
    "PITCH_3",
    "ROT_2",
    "ROT_3",
]

# Threshold: if the absolute value of sin(angle) between consecutive links
# falls below this the config is considered near-singular.
SINGULARITY_THRESHOLD = 0.05

# Small perturbation applied when a near-singular seed is detected.
SINGULARITY_PERTURB = 0.07          # radians


class MoveWithMoveIt(Node):

    def __init__(self):
        super().__init__("move_with_moveit")

        self._last_joints: list[float] | None = None

        # MoveIt client
        self.client = ActionClient(self, MoveGroup, 'move_action')

        self.get_logger().info("Waiting for MoveIt...")
        self.client.wait_for_server()
        self.get_logger().info("MoveIt ready ✔")

        # Gripper action client
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory"
        )

        self.get_logger().info("Waiting for gripper controller...")
        self.gripper_client.wait_for_server()
        self.get_logger().info("Gripper ready ✔")

        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        qos = QoSProfile(depth=10)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/visualization_marker_array",
            qos
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            "/compute_ik"
        )

        self.leaf_markers = MarkerArray()

        self.marker_timer = self.create_timer(
            1.0,
            self.publish_leaf_markers
        )

        self.get_logger().info("Waiting for IK service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service ready ✔")

        self.apply_scene_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene"
        )
        self.get_logger().info("Waiting for /apply_planning_scene service...")
        self.apply_scene_client.wait_for_service()
        self.get_logger().info("ApplyPlanningScene ready ✔")

        self.add_leaves()
        self.add_cylinder()

    # -------------------------------------------------
    # OBSTACLE
    # -------------------------------------------------
    def add_cylinder(self):

        collision = CollisionObject()
        collision.id = "obstacle"
        collision.header.frame_id = "base_footprint"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [1.6, 0.015]

        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"
        pose.pose.position.x = 0.3
        pose.pose.position.y = -0.45
        pose.pose.position.z = 0.8
        pose.pose.orientation.w = 1.0

        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose.pose)
        collision.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(collision)

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            self.get_logger().error("Failed to add cylinder via service ❌")
            return

        self.get_logger().info("Cylinder added via ApplyPlanningScene service ✔")
        time.sleep(1.5)
    
    def publish_leaf_markers(self):

        if hasattr(self, "leaf_markers"):
            self.marker_pub.publish(self.leaf_markers)
    # -------------------------------------------------
    # LEAVES (VISUAL ONLY)
    # -------------------------------------------------
    def add_leaves(self):

        marker_array = MarkerArray()

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS = 0.055
        START_ANGLE = math.pi

        # -------------------------------------------------
        # Petiole geometry
        # -------------------------------------------------
        PETIOLE_LENGTH = 0.06

        BASE_RADIUS = 0.012
        TIP_RADIUS = 0.004

        SIDES = 20

        STEM_X = 0.3
        STEM_Y = -0.45

        # 45° upward inclination
        ELEVATION = math.radians(45)

        for i in range(NUM_LEAVES):

            marker = Marker()

            marker.header.frame_id = "base_footprint"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "leaves"
            marker.id = i

            marker.type = Marker.TRIANGLE_LIST
            marker.action = Marker.ADD
            marker.lifetime.sec = 0

            marker.scale.x = 1.0
            marker.scale.y = 1.0
            marker.scale.z = 1.0

            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.0
            marker.color.a = 1.0

            # -------------------------------------------------
            # Spiral position
            # -------------------------------------------------
            theta = START_ANGLE + i * GOLDEN_ANGLE

            z_center = (
                0.20
                + (i / NUM_LEAVES) * TOTAL_HEIGHT
            )

            # -------------------------------------------------
            # 45° upward petiole axis
            # -------------------------------------------------
            horizontal_mag = math.cos(ELEVATION)

            rx = horizontal_mag * math.cos(theta)
            ry = horizontal_mag * math.sin(theta)
            rz = math.sin(ELEVATION)

            # -------------------------------------------------
            # Stem attachment point (thick end)
            # -------------------------------------------------
            base_x = STEM_X + LEAF_RADIUS * math.cos(theta)
            base_y = STEM_Y + LEAF_RADIUS * math.sin(theta)
            base_z = z_center

            # -------------------------------------------------
            # Tip point (thin end)
            # -------------------------------------------------
            tip_x = base_x + PETIOLE_LENGTH * rx
            tip_y = base_y + PETIOLE_LENGTH * ry
            tip_z = base_z + PETIOLE_LENGTH * rz

            # -------------------------------------------------
            # Local orthonormal basis
            # -------------------------------------------------

            # axis vector
            ax = rx
            ay = ry
            az = rz

            # perpendicular 1
            ux = -math.sin(theta)
            uy = math.cos(theta)
            uz = 0.0

            # perpendicular 2 = axis × u
            vx = ay * uz - az * uy
            vy = az * ux - ax * uz
            vz = ax * uy - ay * ux

            # normalize v
            norm_v = math.sqrt(vx**2 + vy**2 + vz**2)

            vx /= norm_v
            vy /= norm_v
            vz /= norm_v

            # -------------------------------------------------
            # Build frustum
            # -------------------------------------------------
            for s in range(SIDES):

                a0 = 2.0 * math.pi * s / SIDES
                a1 = 2.0 * math.pi * (s + 1) / SIDES

                c0 = math.cos(a0)
                s0 = math.sin(a0)

                c1 = math.cos(a1)
                s1 = math.sin(a1)

                # thick base ring
                p0 = Point(
                    x=base_x + BASE_RADIUS *
                    (ux * c0 + vx * s0),

                    y=base_y + BASE_RADIUS *
                    (uy * c0 + vy * s0),

                    z=base_z + BASE_RADIUS *
                    (uz * c0 + vz * s0)
                )

                p1 = Point(
                    x=base_x + BASE_RADIUS *
                    (ux * c1 + vx * s1),

                    y=base_y + BASE_RADIUS *
                    (uy * c1 + vy * s1),

                    z=base_z + BASE_RADIUS *
                    (uz * c1 + vz * s1)
                )

                # thin tip ring
                p2 = Point(
                    x=tip_x + TIP_RADIUS *
                    (ux * c0 + vx * s0),

                    y=tip_y + TIP_RADIUS *
                    (uy * c0 + vy * s0),

                    z=tip_z + TIP_RADIUS *
                    (uz * c0 + vz * s0)
                )

                p3 = Point(
                    x=tip_x + TIP_RADIUS *
                    (ux * c1 + vx * s1),

                    y=tip_y + TIP_RADIUS *
                    (uy * c1 + vy * s1),

                    z=tip_z + TIP_RADIUS *
                    (uz * c1 + vz * s1)
                )

                marker.points.extend([p0, p1, p2])
                marker.points.extend([p2, p1, p3])

            marker_array.markers.append(marker)

        # -------------------------------------------------
        # Store markers so RViz can receive them later
        # -------------------------------------------------
        self.leaf_markers = marker_array

        # Publish immediately
        self.publish_leaf_markers()

        self.get_logger().info(
            "All 45° upward petioles added ✔"
        )
    # -------------------------------------------------
    # ENABLE SINGLE LEAF COLLISION
    # -------------------------------------------------
    def enable_leaf_collision(self, leaf_id, x, y, z, theta):

        leaf = CollisionObject()
        leaf.id = leaf_id
        leaf.header.frame_id = "base_footprint"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [0.06, 0.006]

        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        q = self.euler_to_quaternion(
            math.pi / 2,
            0,
            theta
        )
        pose.pose.orientation = q

        leaf.primitives.append(primitive)
        leaf.primitive_poses.append(pose.pose)
        leaf.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(leaf)

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        if result is None or not result.success:
            self.get_logger().error(
                f"Failed to add {leaf_id} collision ❌"
            )
            return

        self.get_logger().info(
            f"{leaf_id} collision enabled ✔"
        )

    # -------------------------------------------------
    # ENABLE TARGET LEAF BY INDEX
    # -------------------------------------------------
    def enable_leaf_by_index(self, index):

        GOLDEN_ANGLE = math.radians(137.5)
        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS = 0.055
        START_ANGLE = math.pi

        theta = START_ANGLE + index * GOLDEN_ANGLE
        z = 0.20 + (index / NUM_LEAVES) * TOTAL_HEIGHT
        x = 0.3 + LEAF_RADIUS * math.cos(theta)
        y = -0.45 + LEAF_RADIUS * math.sin(theta)

        self.enable_leaf_collision(
            f"leaf_{index}",
            x, y, z, theta
        )

    # -------------------------------------------------
    def attach_leaf(self, leaf_id):

        scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10
        )

        attached = AttachedCollisionObject()
        attached.object.id = leaf_id
        attached.object.header.frame_id = "gripper_base_link"
        attached.object.operation = CollisionObject.ADD
        attached.link_name = "gripper_base_link"
        attached.object.pose.position.x = 0.0
        attached.object.pose.position.y = 0.0
        attached.object.pose.position.z = 0.10
        attached.object.pose.orientation.w = 1.0

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.attached_collision_objects.append(attached)

        scene_pub.publish(scene)

        self.get_logger().info(f"Leaf {leaf_id} attached ✔")

    # -------------------------------------------------
    # REMOVE SINGLE LEAF MARKER
    # -------------------------------------------------
    def remove_leaf_marker(self, index):

        marker_array = MarkerArray()

        marker = Marker()
        marker.header.frame_id = "base_footprint"
        marker.ns = "leaves"
        marker.id = index
        marker.action = Marker.DELETE

        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

        self.get_logger().info(f"leaf_{index} visual removed ✔")

    # -------------------------------------------------
    # GRIPPER
    # -------------------------------------------------
    def move_gripper(self, position):

        goal = FollowJointTrajectory.Goal()

        traj = JointTrajectory()
        traj.joint_names = ["left_finger_joint"]

        point = JointTrajectoryPoint()
        point.positions = [position]
        point.time_from_start.sec = 1

        traj.points.append(point)
        goal.trajectory = traj

        send_future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Gripper rejected ❌")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info("Gripper done ✔")

    def pluck_motion(self, current_pose):
        j = list(current_pose)

        tilt_pose = j.copy()
        tilt_pose[3] += 0.4
        self.move_to_joints(tilt_pose, "TILT_DOWN")

        twist_pose = tilt_pose.copy()
        twist_pose[5] += math.pi / 2
        self.move_to_joints(twist_pose, "TWIST")

        lift_pose = twist_pose.copy()
        lift_pose[2] += 0.05
        self.move_to_joints(lift_pose, "LIFT_AFTER_PLUCK")

    # -------------------------------------------------
    # MOVE
    # -------------------------------------------------
    def move_to_joints(self, joints, name):

        goal = MoveGroup.Goal()

        goal.request.group_name = "arm"
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id  = "RRTConnectkConfigDefault"
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3
        goal.request.num_planning_attempts = 3

        constraints = Constraints()

        for i, joint_name in enumerate(ARM_JOINT_NAMES):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joints[i])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal.request.goal_constraints.append(constraints)

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{name} rejected ❌")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self.get_logger().info(f"{name} done ✔")

        self._last_joints = list(joints)

    # -------------------------------------------------
    # RUN PICK PIPELINE
    # -------------------------------------------------
    def run(self):

        target_leaf = 18

        self.get_logger().info(
            "Waiting for planning scene to stabilise..."
        )
        time.sleep(1.0)

        #self.remove_leaf_marker(target_leaf)

        p2 = [0.179, 0.534, -1.878, 1.774, -0.085, 0.00]
        self.move_to_joints(p2, "P2")

        self.move_gripper(0.0)

        self.grab_leaf_ik(target_leaf)

        self.get_logger().info("PICK COMPLETE ✔")

    # -------------------------------------------------
    # ALLOWED COLLISION MATRIX helper
    # -------------------------------------------------
    GRASP_LINKS = [
        "gripper_base_link",
        "left_finger",
        "right_finger",
        "J_5",
        "J_6",
    ]

    def _set_leaf_acm(self, leaf_id: str, allow: bool) -> None:
        from moveit_msgs.msg import AllowedCollisionMatrix, AllowedCollisionEntry

        scene = PlanningScene()
        scene.is_diff = True

        acm = AllowedCollisionMatrix()
        all_names = list(self.GRASP_LINKS) + [leaf_id]

        for name in all_names:
            acm.entry_names.append(name)
            entry = AllowedCollisionEntry()
            entry.enabled = [allow] * len(all_names)
            acm.entry_values.append(entry)

        scene.allowed_collision_matrix = acm

        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            self.get_logger().error("ApplyPlanningScene ACM update failed ❌")
            return

        state = "ALLOWED ✔" if allow else "restored ✔"
        self.get_logger().info(
            f"ACM {leaf_id} ↔ gripper links: {state}"
        )
        time.sleep(0.15)

    # -------------------------------------------------
    # QUAT
    # -------------------------------------------------
    def euler_to_quaternion(self, roll, pitch, yaw):

        q = Quaternion()

        q.x = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        q.y = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        q.z = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        q.w = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)

        return q
    
    def wrap_angle(self, angle):

        while angle > math.pi:
            angle -= 2 * math.pi

        while angle < -math.pi:
            angle += 2 * math.pi

        return angle

    # ==========================================================================
    # FEATURE: AVOID SINGULARITIES
    # ==========================================================================
    def _desingularize_seed(self, joints: list[float]) -> list[float]:

        result = list(joints)

        if abs(result[4]) < SINGULARITY_THRESHOLD:
            result[4] += SINGULARITY_PERTURB
            self.get_logger().warn(
                f"ROT_2 near singularity ({joints[4]:.3f} rad) — "
                f"seed perturbed to {result[4]:.3f}"
            )

        if abs(result[2]) < SINGULARITY_THRESHOLD:
            result[2] += SINGULARITY_PERTURB
            self.get_logger().warn(
                f"PITCH_2 near singularity ({joints[2]:.3f} rad) — "
                f"seed perturbed to {result[2]:.3f}"
            )

        return result

    # ==========================================================================
    # FEATURE: RESPECT LIMITS
    # ==========================================================================
    def _check_and_clamp_limits(
        self,
        joints: list[float],
        tolerance: float = 1e-3,
    ) -> list[float] | None:

        result = list(joints)

        for i, name in enumerate(ARM_JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            val = result[i]

            if val < lo - tolerance or val > hi + tolerance:
                self.get_logger().error(
                    f"Joint {name} = {val:.4f} rad violates "
                    f"limits [{lo}, {hi}] — IK solution rejected ❌"
                )
                return None

            result[i] = max(lo, min(hi, val))

        return result

    # ==========================================================================
    # FEATURE: STAY ON SAME BRANCH
    # ==========================================================================
    def _match_branch(
        self,
        solution: list[float],
        seed: list[float],
    ) -> list[float]:

        result = list(solution)

        for i, name in enumerate(ARM_JOINT_NAMES):
            lo, hi = JOINT_LIMITS[name]
            val = result[i]
            ref = seed[i]

            if (val * ref) < 0:
                for delta in (2 * math.pi, -2 * math.pi):
                    candidate = val + delta
                    if lo <= candidate <= hi:
                        self.get_logger().info(
                            f"Branch correction on {name}: "
                            f"{val:.3f} → {candidate:.3f} "
                            f"(seed={ref:.3f})"
                        )
                        result[i] = candidate
                        break

        return result

    # ==========================================================================
    # FEATURE: CHOOSE NEAREST SOLUTION
    # ==========================================================================
    def _joint_distance(self, a: list[float], b: list[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # ==========================================================================
    # ENHANCED solve_ik
    # FIX 1: avoid_collisions=True so the cylinder is respected during IK
    # ==========================================================================
    def solve_ik(
        self,
        x: float,
        y: float,
        z: float,
        orientation_rpy: tuple[float, float, float] = (
            math.pi,
            0.0,
            math.radians(80)
        ),
    ) -> list[float] | None:

        if self._last_joints is not None:
            primary_seed = list(self._last_joints)
            self.get_logger().info(
                "IK seeded from previous state ✔"
            )
        else:
            primary_seed = [0.0] * 6
            self.get_logger().info(
                "IK seeded from neutral home pose"
            )

        clean_seed = self._desingularize_seed(
            primary_seed
        )

        seeds_to_try = [clean_seed]

        for delta in (0.15, -0.15, 0.30):
            perturbed = [
                v + delta
                for v in clean_seed
            ]
            seeds_to_try.append(perturbed)

        target_orientation = self.euler_to_quaternion(
            *orientation_rpy
        )

        candidates = []

        for seed in seeds_to_try:

            request = GetPositionIK.Request()

            request.ik_request.group_name = "arm"

            # -------------------------------------------------
            # FIX: solve IK for TCP frame
            # -------------------------------------------------
            request.ik_request.ik_link_name = "link_tcp"

            request.ik_request.timeout.sec = 2
            request.ik_request.avoid_collisions = True

            seed_state = RobotState()
            seed_state.joint_state.name = list(
                ARM_JOINT_NAMES
            )
            seed_state.joint_state.position = [
                float(v)
                for v in seed
            ]

            request.ik_request.robot_state = (
                seed_state
            )

            pose = PoseStamped()

            # -------------------------------------------------
            # FIX: pose expressed in world frame
            # -------------------------------------------------
            pose.header.frame_id = (
                "base_footprint"
            )

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z

            pose.pose.orientation = (
                target_orientation
            )

            request.ik_request.pose_stamped = (
                pose
            )

            future = self.ik_client.call_async(
                request
            )

            rclpy.spin_until_future_complete(
                self,
                future
            )

            result = future.result()

            if (
                result is None
                or result.error_code.val != 1
            ):
                continue

            joint_state = (
                result.solution.joint_state
            )

            raw_joints = []

            try:
                for name in ARM_JOINT_NAMES:
                    idx = joint_state.name.index(
                        name
                    )

                    raw_joints.append(
                        joint_state.position[idx]
                    )

            except ValueError:
                continue

            branched = self._match_branch(
                raw_joints,
                primary_seed
            )

            clamped = (
                self._check_and_clamp_limits(
                    branched
                )
            )

            if clamped is None:
                continue

            dist = self._joint_distance(
                clamped,
                primary_seed
            )

            candidates.append(
                (dist, clamped)
            )

        if not candidates:
            self.get_logger().error(
                "IK failed for all seeds ❌"
            )
            return None

        candidates.sort(
            key=lambda c: c[0]
        )

        best_dist, best_joints = (
            candidates[0]
        )

        self.get_logger().info(
            f"IK chose nearest solution "
            f"(dist={best_dist:.4f}, "
            f"{len(candidates)} candidate(s)) ✔"
        )

        return best_joints

    # ==========================================================================
    # grab_leaf_ik  —  GRASP FREE TIP (away from stem), collision-safe
    #
    # FIX 2: The approach direction is now OUTWARD from the stem so the gripper
    #         closes on the free tip of the leaf, not the base near the stem.
    #
    # Geometry recap:
    #   stem centre     → (STEM_X, STEM_Y)
    #   leaf attachment → stem + LEAF_RADIUS * (cos θ, sin θ)   ← stem end
    #   leaf free tip   → leaf_attachment + LEAF_LENGTH * (cos θ, sin θ)
    #
    #   The radial unit vector (rx, ry) = (cos θ, sin θ) points OUTWARD from
    #   the stem.  To grab the free tip the gripper must approach from OUTSIDE
    #   (i.e. from beyond the tip) and the TCP target must be placed so the
    #   fingers straddle the tip, not the base.
    # ==========================================================================
    def grab_leaf_ik(
        self,
        leaf_index: int
    ) -> None:

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS = 0.055
        LEAF_LENGTH = 0.06
        START_ANGLE = math.pi

        STEM_X = 0.3
        STEM_Y = -0.45

        # -------------------------------------------------
        # PRE-APPROACH
        # -------------------------------------------------
        PRE_STANDOFF = 0.05
        Z_CLEARANCE = 0.04

        # -------------------------------------------------
        # LEAF GEOMETRY
        # -------------------------------------------------
        theta = (
            START_ANGLE
            + leaf_index * GOLDEN_ANGLE
        )

        z_leaf = (
            0.20
            + (
                leaf_index
                / NUM_LEAVES
            )
            * TOTAL_HEIGHT
        )

        x_base = (
            STEM_X
            + LEAF_RADIUS
            * math.cos(theta)
        )

        y_base = (
            STEM_Y
            + LEAF_RADIUS
            * math.sin(theta)
        )

        rx = math.cos(theta)
        ry = math.sin(theta)

        # -------------------------------------------------
        # LEAF TIP
        # -------------------------------------------------
        x_tip = (
            x_base
            + LEAF_LENGTH * rx
        )

        y_tip = (
            y_base
            + LEAF_LENGTH * ry
        )

        self.get_logger().info(
            f"Leaf {leaf_index}: "
            f"tip=({x_tip:.3f}, "
            f"{y_tip:.3f}, "
            f"{z_leaf:.3f})"
        )

        # -------------------------------------------------
        # FIX: TCP already represents grasp point
        # no fake reach compensation
        # -------------------------------------------------
        grasp_x = x_tip
        grasp_y = y_tip
        grasp_z = z_leaf

        # -------------------------------------------------
        # PRE-APPROACH
        # move from outside inward
        # -------------------------------------------------
        pre_x = (
            grasp_x
            + PRE_STANDOFF * rx
        )

        pre_y = (
            grasp_y
            + PRE_STANDOFF * ry
        )

        pre_z = (
            grasp_z
            + Z_CLEARANCE
        )

        # -------------------------------------------------
        # ORIENTATION
        # -------------------------------------------------
        yaw = self.wrap_angle(
            math.pi - theta
        )

        approach_rpy = (
            math.pi / 2,
            0.0,
            yaw
        )

        self.get_logger().info(
            f"Approach yaw = "
            f"{math.degrees(yaw):.1f}°"
        )

        # -------------------------------------------------
        # PRE-APPROACH IK
        # -------------------------------------------------
        pre_joints = self.solve_ik(
            pre_x,
            pre_y,
            pre_z,
            approach_rpy
        )

        if pre_joints is None:
            self.get_logger().error(
                "Pre-approach IK failed ❌"
            )
            return

        self.move_to_joints(
            pre_joints,
            "PRE_APPROACH"
        )

        # -------------------------------------------------
        # FINAL GRASP IK
        # -------------------------------------------------
        grasp_joints = self.solve_ik(
            grasp_x,
            grasp_y,
            grasp_z,
            approach_rpy
        )

        if grasp_joints is None:
            self.get_logger().error(
                "Final grasp IK failed ❌"
            )
            return

        self.move_to_joints(
            grasp_joints,
            f"GRASP_LEAF_{leaf_index}"
        )

        # -------------------------------------------------
        # CLOSE GRIPPER
        # -------------------------------------------------
        self.move_gripper(0.035)

        # -------------------------------------------------
        # COLLISION
        # -------------------------------------------------
        leaf_id = f"leaf_{leaf_index}"

        self.enable_leaf_by_index(
            leaf_index
        )

        time.sleep(0.3)

        self._set_leaf_acm(
            leaf_id,
            allow=True
        )

        self.get_logger().info(
            "Leaf grasp complete ✔"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MoveWithMoveIt()
    node.run()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
