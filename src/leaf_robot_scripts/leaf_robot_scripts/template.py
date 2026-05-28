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

        # ------------------------------------------------------------------
        # Feature: USE PREVIOUS STATE
        # Stores the last successfully executed joint configuration so every
        # IK call can be seeded from it (nearest-solution + branch-staying).
        # ------------------------------------------------------------------
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

        # Planning scene publisher (kept for visual markers only)
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

        self.get_logger().info("Waiting for IK service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service ready ✔")

        # ApplyPlanningScene: the only reliable way to add/modify collision
        # objects and ACM entries without wiping existing SRDF entries.
        # ALL collision objects (cylinder + leaves) now use this service.
        self.apply_scene_client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene"
        )
        self.get_logger().info("Waiting for /apply_planning_scene service...")
        self.apply_scene_client.wait_for_service()
        self.get_logger().info("ApplyPlanningScene ready ✔")

        # Add visual leaves first (no collision yet)
        self.add_leaves()

        # Add the cylinder via the reliable service path so MoveIt's
        # planning scene monitor is guaranteed to have it before any
        # motion planning call is made.
        self.add_cylinder()

    # -------------------------------------------------
    # OBSTACLE  — added via ApplyPlanningScene service
    # (FIX 1: use service instead of topic publisher so
    #  the scene monitor processes it before planning.)
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

        # Use the service — it merges into the existing scene correctly.
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self.apply_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()

        if result is None or not result.success:
            self.get_logger().error("Failed to add cylinder via service ❌")
            return

        self.get_logger().info("Cylinder added via ApplyPlanningScene service ✔")

        # Give the planning scene monitor a moment to propagate the update
        # before any IK / motion planning call is attempted.
        time.sleep(1.5)

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

        for i in range(NUM_LEAVES):

            marker = Marker()

            marker.header.frame_id = "base_footprint"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "leaves"
            marker.id = i

            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.lifetime.sec = 0

            theta = START_ANGLE + i * GOLDEN_ANGLE

            z = 0.20 + (i / NUM_LEAVES) * TOTAL_HEIGHT

            x = 0.3 + LEAF_RADIUS * math.cos(theta)
            y = -0.45 + LEAF_RADIUS * math.sin(theta)

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z

            yaw = theta + math.pi

            q = self.euler_to_quaternion(0, 0, yaw)

            marker.pose.orientation = q

            marker.scale.x = 0.06
            marker.scale.y = 0.03
            marker.scale.z = 0.01

            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

        self.get_logger().info("All visual leaves added ✔")

    # -------------------------------------------------
    # ENABLE SINGLE LEAF COLLISION
    # (FIX 1: uses ApplyPlanningScene service, not topic)
    # -------------------------------------------------
    def enable_leaf_collision(self, leaf_id, x, y, z, theta):

        leaf = CollisionObject()
        leaf.id = leaf_id
        leaf.header.frame_id = "base_footprint"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.06, 0.015, 0.005]

        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        yaw = theta + math.pi
        q = self.euler_to_quaternion(0, 0, yaw)
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
            self.get_logger().error(f"Failed to add {leaf_id} collision ❌")
            return

        self.get_logger().info(f"{leaf_id} collision enabled via service ✔")

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
        """
        Adds real plucking behavior:
        - slight downward tilt
        - half rotation
        - lift
        """

        j = list(current_pose)

        # 1. small downward tilt (wrist_1_joint)
        tilt_pose = j.copy()
        tilt_pose[3] += 0.4
        self.move_to_joints(tilt_pose, "TILT_DOWN")

        # 2. half rotation (wrist_3_joint)
        twist_pose = tilt_pose.copy()
        twist_pose[5] += math.pi / 2
        self.move_to_joints(twist_pose, "TWIST")

        # 3. slight pull upward
        lift_pose = twist_pose.copy()
        lift_pose[2] += 0.05
        self.move_to_joints(lift_pose, "LIFT_AFTER_PLUCK")

    # -------------------------------------------------
    # MOVE
    # (FIX 2: switched to OMPL + RRTConnect for full
    #  collision-aware path planning around obstacles.
    #  Pilz PTP only interpolates in joint space and
    #  does NOT reliably avoid collision objects.)
    # -------------------------------------------------
    def move_to_joints(self, joints, name):

        goal = MoveGroup.Goal()

        goal.request.group_name = "arm"

        # ── FIX 2: Use OMPL instead of Pilz PTP ──────────────────────────
        # Pilz PTP performs straight-line joint-space interpolation and
        # has no collision-aware replanning.  OMPL (RRTConnect) samples
        # the configuration space and will route around the cylinder.
        goal.request.pipeline_id = "ompl"
        goal.request.planner_id  = "RRTConnectkConfigDefault"

        # Give the planner enough time to find a path around the obstacle.
        goal.request.allowed_planning_time = 5.0

        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        # Retry up to 3 times so transient planning failures don't abort.
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

        # ------------------------------------------------------------------
        # Feature: USE PREVIOUS STATE — persist the executed config
        # ------------------------------------------------------------------
        self._last_joints = list(joints)

    # -------------------------------------------------
    # RUN PICK PIPELINE
    # -------------------------------------------------
    def run(self):

        target_leaf = 18

        # ---------------------------------
        # Extra buffer: confirm the cylinder
        # is fully registered in the planning
        # scene before any motion is planned.
        # (FIX 3: additional safety sleep on
        #  top of the 1.5 s in add_cylinder.)
        # ---------------------------------
        self.get_logger().info(
            "Waiting for planning scene to stabilise..."
        )
        time.sleep(1.0)

        # ---------------------------------
        # REMOVE VISUAL MARKER ONLY
        # Collision box is NOT added here — the leaf must not exist as an
        # obstacle while the arm is approaching it.  enable_leaf_by_index()
        # is called inside grab_leaf_ik(), right before the grasp move,
        # with the ACM already set to permit contact.
        # ---------------------------------
        self.remove_leaf_marker(target_leaf)

        # ---------------------------------
        # SAFE PRE-GRASP POSE
        # ---------------------------------
        p2 = [0.179, 0.534, -1.878, 1.774, -0.085, 0.00]

        self.move_to_joints(p2, "P2")

        # ---------------------------------
        # OPEN GRIPPER BEFORE APPROACH
        # ---------------------------------
        self.move_gripper(0.0)

        # ---------------------------------
        # MOVE TO LEAF USING IK
        # ---------------------------------
        self.grab_leaf_ik(target_leaf)

        self.get_logger().info("PICK COMPLETE ✔")

    # -------------------------------------------------
    # ALLOWED COLLISION MATRIX helper
    # -------------------------------------------------
    # Links that are allowed to touch the leaf during the grasp move.
    GRASP_LINKS = [
        "gripper_base_link",
        "left_finger",
        "right_finger",
        "J_5",
        "J_6",
    ]

    def _set_leaf_acm(self, leaf_id: str, allow: bool) -> None:
        """
        Safely add or remove allowed-collision entries between *leaf_id* and
        every link in GRASP_LINKS using the ApplyPlanningScene service.

        NOTE: "obstacle" (the cylinder) is intentionally NOT included here.
        Only leaf ↔ gripper link pairs are relaxed so the cylinder always
        remains a hard collision constraint for the planner.
        """
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
        time.sleep(0.15)   # let move_group's monitor process the update

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

    # ==========================================================================
    # FEATURE: AVOID SINGULARITIES
    # ==========================================================================
    def _desingularize_seed(self, joints: list[float]) -> list[float]:
        """Return a copy of *joints* with near-singular values perturbed."""

        result = list(joints)

        # Wrist singularity: ROT_2 (index 4) near 0
        if abs(result[4]) < SINGULARITY_THRESHOLD:
            result[4] += SINGULARITY_PERTURB
            self.get_logger().warn(
                f"ROT_2 near singularity ({joints[4]:.3f} rad) — "
                f"seed perturbed to {result[4]:.3f}"
            )

        # Elbow singularity: PITCH_2 (index 2) near 0
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
    # ==========================================================================
    def solve_ik(
        self,
        x: float,
        y: float,
        z: float,
        orientation_rpy: tuple[float, float, float] = (math.pi, 0.0, math.radians(80)),
    ) -> list[float] | None:

        # 1. USE PREVIOUS STATE as the primary seed
        if self._last_joints is not None:
            primary_seed = list(self._last_joints)
            self.get_logger().info("IK seeded from previous state ✔")
        else:
            primary_seed = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.get_logger().info("IK seeded from neutral home pose")

        # 2. AVOID SINGULARITIES — clean the primary seed
        clean_seed = self._desingularize_seed(primary_seed)

        seeds_to_try = [clean_seed]

        for delta in (0.15, -0.15, 0.30):
            perturbed = [v + delta for v in clean_seed]
            seeds_to_try.append(perturbed)

        target_orientation = self.euler_to_quaternion(*orientation_rpy)

        candidates: list[tuple[float, list[float]]] = []

        for seed in seeds_to_try:

            request = GetPositionIK.Request()
            request.ik_request.group_name = "arm"
            request.ik_request.ik_link_name = "gripper_base_link"
            request.ik_request.timeout.sec = 2

            seed_state = RobotState()
            seed_state.joint_state.name = list(ARM_JOINT_NAMES)
            seed_state.joint_state.position = [float(v) for v in seed]
            request.ik_request.robot_state = seed_state

            pose = PoseStamped()
            pose.header.frame_id = "base_footprint"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation = target_orientation

            request.ik_request.pose_stamped = pose

            future = self.ik_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            result = future.result()

            if result.error_code.val != 1:
                continue

            joint_state = result.solution.joint_state

            raw_joints = []
            try:
                for name in ARM_JOINT_NAMES:
                    idx = joint_state.name.index(name)
                    raw_joints.append(joint_state.position[idx])
            except ValueError:
                continue

            # 4. STAY ON SAME BRANCH
            branched = self._match_branch(raw_joints, primary_seed)

            # 5. RESPECT LIMITS
            clamped = self._check_and_clamp_limits(branched)
            if clamped is None:
                continue

            dist = self._joint_distance(clamped, primary_seed)
            candidates.append((dist, clamped))

        if not candidates:
            self.get_logger().error("IK failed for all seeds ❌")
            return None

        # 3. CHOOSE NEAREST SOLUTION
        candidates.sort(key=lambda c: c[0])
        best_dist, best_joints = candidates[0]

        self.get_logger().info(
            f"IK: chose nearest solution "
            f"(dist={best_dist:.4f} rad, "
            f"from {len(candidates)} candidate(s)) ✔"
        )

        return best_joints

    # ==========================================================================
    # grab_leaf_ik  —  HORIZONTAL PARALLEL approach  (bottom-to-top closing)
    # ==========================================================================
     # ==========================================================================
# grab_leaf_ik  —  SOFT STEM AVOIDANCE VERSION
# ==========================================================================
    def grab_leaf_ik(self, leaf_index: int) -> None:

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES   = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS  = 0.055
        START_ANGLE  = math.pi

        STEM_X = 0.3
        STEM_Y = -0.45

        # ---------------------------------------------------------
        # GRIPPER GEOMETRY
        # ---------------------------------------------------------
        FINGER_LENGTH = 0.235

        # IMPORTANT:
        # extra safety from stem
        RADIAL_CLEARANCE = 0.06

        # vertical offset
        Z_CLEARANCE = 0.04

        # final insertion distance
        FINAL_APPROACH = 0.045

        # ---------------------------------------------------------
        # LEAF POSITION
        # ---------------------------------------------------------
        theta = START_ANGLE + leaf_index * GOLDEN_ANGLE

        z_leaf = 0.20 + (leaf_index / NUM_LEAVES) * TOTAL_HEIGHT

        x_leaf = STEM_X + LEAF_RADIUS * math.cos(theta)
        y_leaf = STEM_Y + LEAF_RADIUS * math.sin(theta)

        self.get_logger().info(
            f"Leaf {leaf_index}: "
            f"x={x_leaf:.3f}, y={y_leaf:.3f}, z={z_leaf:.3f}"
        )

        # =========================================================
        # SAFE APPROACH DIRECTION
        # =========================================================

        # radial direction AWAY from stem
        rx = math.cos(theta)
        ry = math.sin(theta)

        # ---------------------------------------------------------
        # FINAL GRASP POSE
        # Put leaf BETWEEN fingers instead of near wrist
        # ---------------------------------------------------------

        # effective TCP → finger-tip offset
        effective_finger_reach = 0.21

        # small safety gap so finger tips don't overshoot
        grasp_margin = 0.01

        reach = effective_finger_reach - grasp_margin

        # TCP target:
        # leaf_position - finger_direction * reach
        grasp_x = x_leaf - reach * rx
        grasp_y = y_leaf - reach * ry
        grasp_z = z_leaf

        # ---------------------------------------------------------
        # PRE-APPROACH POSE
        # farther away along same line
        # ---------------------------------------------------------
        pre_x = grasp_x + RADIAL_CLEARANCE * rx
        pre_y = grasp_y + RADIAL_CLEARANCE * ry
        pre_z = grasp_z + Z_CLEARANCE
        # =========================================================
        # ORIENTATION
        # =========================================================

        # gripper parallel to stem tangent
        yaw = theta + math.pi / 2

        approach_rpy = (
            math.pi / 2,
            0.0,
            yaw
        )

        self.get_logger().info(
            f"Safe approach yaw = {math.degrees(yaw):.1f}"
        )

        # =========================================================
        # SOLVE PRE-APPROACH IK
        # =========================================================
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

        # =========================================================
        # MOVE TO PRE-APPROACH
        # =========================================================
        self.move_to_joints(
            pre_joints,
            "PRE_APPROACH"
        )

        # =========================================================
        # SOLVE FINAL IK
        # =========================================================
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

        # =========================================================
        # FINAL APPROACH
        # =========================================================
        self.move_to_joints(
            grasp_joints,
            f"GRASP_LEAF_{leaf_index}"
        )

        # =========================================================
        # CLOSE GRIPPER
        # =========================================================
        self.move_gripper(0.035)

        # =========================================================
        # ENABLE LEAF COLLISION
        # =========================================================
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