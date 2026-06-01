#!/usr/bin/env python3

import rclpy
import time
import math
from tf2_ros import Buffer, TransformListener

from rclpy.node import Node
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import JointConstraint
from sensor_msgs.msg import JointState


from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    CollisionObject,
    PlanningScene,
    ObjectColor,
    AttachedCollisionObject
)

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import ColorRGBA

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.qos import QoSProfile
from rclpy.qos import DurabilityPolicy


class MoveWithMoveIt(Node):

    def __init__(self):
        super().__init__("move_with_moveit")

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

        # Planning scene publisher
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

        # TF for reading TCP pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        self.current_joint_state = None
        
    

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        self.get_logger().info("Waiting for IK service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service ready ✔")

        self.add_cylinder()
        self.add_leaves()

    # -------------------------------------------------
    # OBSTACLE
    # -------------------------------------------------
    def add_cylinder(self):

        scene = PlanningScene()
        scene.is_diff = True

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

        scene.world.collision_objects.append(collision)

        self.scene_pub.publish(scene)

        self.get_logger().info("Cylinder added ✔")
        time.sleep(2)

    # -------------------------------------------------
    # LEAVES
    # -------------------------------------------------
    # -------------------------------------------------
# LEAVES (VISUAL ONLY)
# -------------------------------------------------
    def add_leaves(self):

        marker_array = MarkerArray()

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS = 0.10

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

            # bigger temporary debug size
            marker.scale.x = 0.10
            marker.scale.y = 0.03
            marker.scale.z = 0.01

            # bright green debug color
            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

        self.get_logger().info("All visual leaves added ✔")
    
    # -------------------------------------------------
# ENABLE SINGLE LEAF COLLISION
# -------------------------------------------------
    def enable_leaf_collision(self, leaf_id, x, y, z, theta):

        scene = PlanningScene()
        scene.is_diff = True

        leaf = CollisionObject()

        leaf.id = leaf_id
        leaf.header.frame_id = "base_footprint"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX

        # same leaf dimensions
        primitive.dimensions = [0.10, 0.015, 0.005]

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

        scene.world.collision_objects.append(leaf)

        self.scene_pub.publish(scene)

        self.get_logger().info(f"{leaf_id} collision enabled ✔")

    # -------------------------------------------------
    # ENABLE TARGET LEAF BY INDEX
    # -------------------------------------------------
    def enable_leaf_by_index(self, index):

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS = 0.10

        START_ANGLE = math.pi

        theta = START_ANGLE + index * GOLDEN_ANGLE

        z = 0.20 + (index / NUM_LEAVES) * TOTAL_HEIGHT

        x = 0.3 + LEAF_RADIUS * math.cos(theta)
        y = -0.45 + LEAF_RADIUS * math.sin(theta)

        self.enable_leaf_collision(
            f"leaf_{index}",
            x,
            y,
            z,
            theta
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

        # move leaf from gripper base to finger tip
        attached.object.pose.position.x = 0.08
        attached.object.pose.position.y = 0.0
        attached.object.pose.position.z = 0.0
        attached.object.pose.orientation.w = 1.0

        scene = PlanningScene()
        scene.is_diff = True

        scene.robot_state.attached_collision_objects.append(
           attached
        )

        scene_pub.publish(scene)

        self.get_logger().info(
        f"Leaf {leaf_id} attached ✔"
        )

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

        # unpack current joints
        j = list(current_pose)

        # 1. small downward tilt (wrist_1_joint)
        tilt_pose = j.copy()
        tilt_pose[3] += 0.4   # pitch down

        self.move_to_joints(tilt_pose, "TILT_DOWN")

        # 2. half rotation (wrist_3_joint)
        twist_pose = tilt_pose.copy()
        twist_pose[5] += math.pi / 2   # 90° twist

        self.move_to_joints(twist_pose, "TWIST")

        # 3. slight pull upward
        lift_pose = twist_pose.copy()
        lift_pose[2] += 0.05  # small lift in joint-space approximation

        self.move_to_joints(lift_pose, "LIFT_AFTER_PLUCK")

    # -------------------------------------------------
    # MOVE
    # -------------------------------------------------
    def move_to_joints(self, joints, name):

        goal = MoveGroup.Goal()

        goal.request.group_name = "arm"
        goal.request.pipeline_id = "pilz_industrial_motion_planner"
        goal.request.planner_id = "PTP"

        goal.request.max_velocity_scaling_factor = 0.3
        goal.request.max_acceleration_scaling_factor = 0.3

        joint_names = [
            
            "ROT_1",
            "PITCH_1",
            "PITCH_2",
            "PITCH_3",
            "ROT_2",
            "ROT_3"
        ]

        constraints = Constraints()

        for i in range(6):
            jc = JointConstraint()
            jc.joint_name = joint_names[i]
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

    # -------------------------------------------------
    # RUN PICK PIPELINE
    # -------------------------------------------------
  
    def run(self):

        target_leaf = 18

        # ---------------------------------
        # leaf height
        # ---------------------------------
        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35

        leaf_z = (
            0.20
            + (target_leaf / NUM_LEAVES)
            * TOTAL_HEIGHT
        )

        # ---------------------------------
        # visual + collision
        # ---------------------------------
        self.remove_leaf_marker(target_leaf)
        #self.enable_leaf_by_index(target_leaf)

        # ---------------------------------
        # MOVE TO P2
        # ---------------------------------
        p2 = [
            0.179,
            0.534,
            -1.878,
            1.774,
            -0.085,
            0.00
        ]

        self.move_to_joints(p2, "P2")

        time.sleep(1.0)

        self.move_gripper(0.0)

        # ---------------------------------
        # TCP POSE
        # ---------------------------------
        tcp = self.get_tcp_pose()

        if tcp is None:
            return

        current_x, current_y, current_z, tcp_yaw = tcp

        self.get_logger().info(
            f"TCP: x={current_x:.3f}, y={current_y:.3f}, "
            f"yaw={math.degrees(tcp_yaw):.2f} deg"
        )
        # ---------------------------------
        # LEAF CENTER (same model)
        # ---------------------------------
        GOLDEN_ANGLE = math.radians(137.5)
        START_ANGLE = math.pi
        LEAF_RADIUS = 0.10

        theta = START_ANGLE + target_leaf * GOLDEN_ANGLE

        leaf_x = 0.3 + LEAF_RADIUS * math.cos(theta)
        leaf_y = -0.45 + LEAF_RADIUS * math.sin(theta)

        # ---------------------------------
        # LEAF CENTER SPHERE (DEBUG)
        # ---------------------------------
        sphere = Marker()
        sphere.header.frame_id = "base_footprint"
        sphere.header.stamp = self.get_clock().now().to_msg()

        sphere.ns = "leaf_center"
        sphere.id = target_leaf + 1000
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        sphere.pose.position.x = leaf_x
        sphere.pose.position.y = leaf_y
        sphere.pose.position.z = leaf_z

        sphere.pose.orientation.w = 1.0

        sphere.scale.x = 0.03
        sphere.scale.y = 0.03
        sphere.scale.z = 0.03

        sphere.color.r = 1.0
        sphere.color.g = 0.0
        sphere.color.b = 0.0
        sphere.color.a = 1.0

        marker_array = MarkerArray()
        marker_array.markers.append(sphere)

        self.marker_pub.publish(marker_array)

        self.get_logger().info(
            f"Leaf center: x={leaf_x:.3f}, y={leaf_y:.3f}"
        )

        # ---------------------------------
        # ANGLE BETWEEN POINTS
        # ---------------------------------
        #dx = leaf_x - current_x
        #dy = leaf_y - current_y

        angle1 = math.atan2(current_y, current_x)
        angle2 = math.atan2(leaf_y, leaf_x)

        angle = angle2 - angle1

        #angle = math.atan2(dy, dx)

        relative_angle = angle - tcp_yaw

        self.get_logger().info(
            f"angle={math.degrees(angle):.2f} deg"
        )

        # ---------------------------------
        # CHECK IF LEAF IS LEFT OR RIGHT
        # ---------------------------------
        stem_x = 0.3

        if leaf_x >= stem_x:
            # RIGHT SIDE LEAF
            target_j5 = relative_angle
            side = "RIGHT"

        else:
            # LEFT SIDE LEAF
            target_j5 = -relative_angle
            side = "LEFT"

        self.get_logger().info(
            f"Leaf side: {side}"
        )

        self.get_logger().info(
            f"target J5={math.degrees(target_j5):.2f} deg"
        )

        # ---------------------------------
        # MOVE TO LEAF HEIGHT
        # ---------------------------------
        dz = leaf_z - current_z

        target_joints = p2.copy()

        target_joints[1] -= dz * 1.2
        target_joints[2] += dz * 1.8

        self.move_to_joints(
            target_joints,
            "MOVE_TO_LEAF_HEIGHT"
        )

        self.get_logger().info("Reached leaf height ✔")

        # ---------------------------------
        # APPLY ONLY J5 (ROT_2 = index 4)
        # ---------------------------------
        stem_x = 0.3

        face_joints = target_joints.copy()

        current_j5 = target_joints[4]

        if leaf_x >= stem_x:
            # RIGHT SIDE
            side = "RIGHT"
            face_joints[4] = current_j5 + abs(relative_angle)

        else:
            # LEFT SIDE
            side = "LEFT"
            face_joints[4] = current_j5 - abs(relative_angle)

        self.get_logger().info(
            f"Leaf side: {side}"
        )

        self.get_logger().info(
            f"Current J5: {math.degrees(current_j5):.2f}"
        )

        self.get_logger().info(
            f"Relative angle: {math.degrees(relative_angle):.2f}"
        )

        self.get_logger().info(
            f"Target J5: {math.degrees(face_joints[4]):.2f}"
        )

        self.move_to_joints(
            face_joints,
            "FACE_LEAF_J5"
        )
        # ---------------------------------
        # APPROACH LEAF USING IK
        # PRE-GRASP FIRST
        # ---------------------------------

        # refresh TCP orientation AFTER J5 move
        tcp = self.get_tcp_pose()

        if tcp is None:
            return

        _, _, _, updated_yaw = tcp

        # pre-grasp point (5 cm away from leaf)
        approach_dist = 0.05

        pre_x = leaf_x - approach_dist * math.cos(theta)
        pre_y = leaf_y - approach_dist * math.sin(theta)

        self.get_logger().info(
            f"Pre-grasp target:"
            f" x={pre_x:.3f}"
            f" y={pre_y:.3f}"
            f" z={leaf_z:.3f}"
        )

        grab_joints = self.solve_ik(
            pre_x,
            pre_y,
            leaf_z,
            updated_yaw,
            # lock_j4=face_joints[3], 
            # lock_j5=face_joints[4]
        )

        if grab_joints is None:
            return

        # move to pre-grasp pose
        self.move_to_joints(
            grab_joints,
            "PRE_GRASP"
        )

        # straight motion to actual leaf
        self.straight_move_to_leaf(
            leaf_x,
            leaf_y,
            leaf_z,
            updated_yaw,
            # lock_j4=face_joints[3], 
            # lock_j5=face_joints[4]
        )

        self.get_logger().info("Reached leaf ✔")

        # close gripper
        self.move_gripper(0.035)

        self.get_logger().info("Leaf grabbed ✔")
        
        # ---------------------------------
        # CURRENT TCP AFTER GRAB
        # ---------------------------------
        tcp = self.get_tcp_pose()

        if tcp is not None:
            grab_x, grab_y, grab_z, grab_yaw = tcp

            self.get_logger().info(
                f"After grab TCP:"
                f" x={grab_x:.3f}"
                f" y={grab_y:.3f}"
                f" z={grab_z:.3f}"
                f" yaw={math.degrees(grab_yaw):.2f} deg"
            )

        # downward semicircle motion
        self.semicircle_down_motion(
            updated_yaw,
            radius=0.02
        )
        
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
    
    def solve_ik(
        self,
        x,
        y,
        z,
        yaw,
        lock_j4=None,
        lock_j5=None
    ):

        request = GetPositionIK.Request()

        request.ik_request.group_name = "arm"
        request.ik_request.ik_link_name = "link_tcp"
        request.ik_request.timeout.sec = 2

        # ---------------------------------
        # current robot state (seed)
        # ---------------------------------
        request.ik_request.robot_state.joint_state.name = [
            "ROT_1",
            "PITCH_1",
            "PITCH_2",
            "PITCH_3",
            "ROT_2",
            "ROT_3"
        ]

        seed = self.get_current_arm_joints()

        if seed is None:
            return None

        request.ik_request.robot_state.joint_state.position = seed

        # ---------------------------------
        # debug info
        # ---------------------------------
        dist = math.sqrt(
            x * x +
            y * y +
            z * z
        )

        self.get_logger().info(
            f"IK target:"
            f" x={x:.3f}"
            f" y={y:.3f}"
            f" z={z:.3f}"
            f" yaw={math.degrees(yaw):.2f}"
            f" dist={dist:.3f}"
        )

        # ---------------------------------
        # target pose
        # ---------------------------------
        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        # Tool-down orientation
        q = self.euler_to_quaternion(
            math.pi/2,   # <-- changed from pi/2
            0.0,
            yaw
        )

        pose.pose.orientation = q

        request.ik_request.pose_stamped = pose

        # ---------------------------------
        # call IK
        # ---------------------------------
        future = self.ik_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        # ---------------------------------
        # failure handling
        # ---------------------------------
        if result is None:
            self.get_logger().error(
                "IK service returned None ❌"
            )
            return None

        if result.error_code.val != 1:
            self.get_logger().error(
                f"IK failed ❌ "
                f"error_code={result.error_code.val}"
            )
            return None

        # ---------------------------------
        # extract arm joints
        # ---------------------------------
        joint_state = result.solution.joint_state

        arm_joint_names = [
            "ROT_1",
            "PITCH_1",
            "PITCH_2",
            "PITCH_3",
            "ROT_2",
            "ROT_3"
        ]

        joints = []

        for name in arm_joint_names:
            idx = joint_state.name.index(name)
            joints.append(
                joint_state.position[idx]
            )

        self.get_logger().info(
            "IK solved ✔ "
            + str(
                [round(j, 3) for j in joints]
            )
        )

        return joints
    def grab_leaf_ik(self, leaf_index):

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        START_ANGLE = math.pi
        LEAF_RADIUS = 0.10

        STEM_X = 0.3
        STEM_Y = -0.45

        # -----------------------------------
        # TARGET LEAF POSITION HEIGHT
        # -----------------------------------
        theta = START_ANGLE + leaf_index * GOLDEN_ANGLE

        leaf_z = (
            0.20
            + (leaf_index / NUM_LEAVES)
            * TOTAL_HEIGHT
        )

        leaf_x = STEM_X + LEAF_RADIUS * math.cos(theta)
        leaf_y = STEM_Y + LEAF_RADIUS * math.sin(theta)

        self.get_logger().info(
            f"Leaf {leaf_index}: "
            f"x={leaf_x:.3f}, "
            f"y={leaf_y:.3f}, "
            f"z={leaf_z:.3f}"
        )

        # -----------------------------------
        # ONLY MOVE TO SAME HEIGHT
        # -----------------------------------
        #
        # Keep arm in front of stem
        # Keep same orientation
        # Only change Z
        #

        target_x = 0.18
        target_y = -0.28
        target_z = leaf_z

        orientation_rpy = (
            math.pi,   # tool down
            0.0,
            0.0
        )

        joints = self.solve_ik(
            target_x,
            target_y,
            target_z,
            orientation_rpy
        )

        if joints is None:
            self.get_logger().error(
                "Height IK failed ❌"
            )
            return

        self.move_to_joints(
            joints,
            f"MOVE_TO_LEAF_HEIGHT_{leaf_index}"
        )

        self.get_logger().info(
            "Reached leaf height ✔"
        )
    def get_tcp_pose(self):

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_footprint",
                "link_tcp",
                rclpy.time.Time()
            )

            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w

            # quaternion → yaw
            siny_cosp = 2 * (
                qw * qz + qx * qy
            )

            cosy_cosp = 1 - 2 * (
                qy * qy + qz * qz
            )

            yaw = math.atan2(
                siny_cosp,
                cosy_cosp
            )

            return x, y, z, yaw

        except Exception as e:

            self.get_logger().error(
                f"Failed to read link_tcp pose: {e}"
            )

            return None

    def straight_move_to_leaf(
        self,
        target_x,
        target_y,
        target_z,
        yaw,
        lock_j4=None,
        lock_j5=None,
        steps=20
    ):

        tcp = self.get_tcp_pose()

        if tcp is None:
            return

        start_x, start_y, start_z, _ = tcp

        for i in range(1, steps + 1):

            alpha = i / steps

            x = (
                start_x
                + alpha
                * (target_x - start_x)
            )

            y = (
                start_y
                + alpha
                * (target_y - start_y)
            )

            z = (
                start_z
                + alpha
                * (target_z - start_z)
            )

            # CHANGED HERE
            joints = self.solve_ik(
                x,
                y,
                z,
                yaw,
                lock_j4=lock_j4,
                lock_j5=lock_j5
            )

            if joints is None:
                self.get_logger().error(
                    "Straight approach failed ❌"
                )
                return

            self.move_to_joints(
                joints,
                f"APPROACH_{i}"
            )
    def semicircle_down_motion(
        self,
        yaw,
        radius=0.02,
        steps=10
    ):

        tcp = self.get_tcp_pose()

        if tcp is None:
            return

        start_x, start_y, start_z, _ = tcp

        center_y = start_y
        center_z = start_z - radius

        for i in range(steps + 1):

            alpha = i / steps

            theta = (
                math.pi / 2
                - alpha * math.pi
            )

            y = (
                center_y
                - radius * math.cos(theta)
            )

            z = (
                center_z
                + radius * math.sin(theta)
            )

            x = start_x

            self.move_to_pose_lin(
                x,
                y,
                z,
                yaw,
                f"SEMICIRCLE_{i}"
            )

        # ---------------------------------
        # CALCULATED END POINT
        # ---------------------------------
        self.get_logger().info(
            f"Semicircle target end:"
            f" x={x:.3f}"
            f" y={y:.3f}"
            f" z={z:.3f}"
        )

        # ---------------------------------
        # REAL TCP AFTER MOTION
        # ---------------------------------
        tcp = self.get_tcp_pose()

        if tcp is not None:
            real_x, real_y, real_z, real_yaw = tcp

            self.get_logger().info(
                f"Real TCP end:"
                f" x={real_x:.3f}"
                f" y={real_y:.3f}"
                f" z={real_z:.3f}"
                f" yaw={math.degrees(real_yaw):.2f} deg"
            )

        self.get_logger().info(
            "Semicircle complete ✔"
        )
    def move_to_pose_lin(
        self,
        x,
        y,
        z,
        yaw,
        name
    ):

        goal = MoveGroup.Goal()

        goal.request.group_name = "arm"
        goal.request.pipeline_id = (
            "pilz_industrial_motion_planner"
        )
        goal.request.planner_id = "LIN"

        goal.request.max_velocity_scaling_factor = 0.1
        goal.request.max_acceleration_scaling_factor = 0.1

        # -----------------------------
        # Cartesian target pose
        # -----------------------------
        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        q = self.euler_to_quaternion(
            math.pi / 2,
            0.0,
            yaw
        )

        pose.pose.orientation = q

        # -----------------------------
        # Pose constraint (important)
        # -----------------------------
        constraint = Constraints()

        # MoveIt pose target
        from moveit_msgs.msg import PositionConstraint
        from moveit_msgs.msg import OrientationConstraint
        from shape_msgs.msg import SolidPrimitive

        pc = PositionConstraint()
        pc.header.frame_id = "base_footprint"
        pc.link_name = "link_tcp"

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.001, 0.001, 0.001]

        pc.constraint_region.primitives.append(box)
        pc.constraint_region.primitive_poses.append(
            pose.pose
        )

        oc = OrientationConstraint()
        oc.header.frame_id = "base_footprint"
        oc.link_name = "link_tcp"

        oc.orientation = pose.pose.orientation

        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0

        constraint.position_constraints.append(pc)
        constraint.orientation_constraints.append(oc)

        goal.request.goal_constraints.append(
            constraint
        )

        send_future = self.client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            send_future
        )

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                f"{name} rejected ❌"
            )
            return

        result_future = (
            goal_handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future
        )

        self.get_logger().info(
            f"{name} done ✔"
        )

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def get_current_arm_joints(self):

        if self.current_joint_state is None:
            self.get_logger().error(
                "No joint state received ❌"
            )
            return None

        wanted = [
            "ROT_1",
            "PITCH_1",
            "PITCH_2",
            "PITCH_3",
            "ROT_2",
            "ROT_3"
        ]

        joints = []

        for name in wanted:
            idx = self.current_joint_state.name.index(name)
            joints.append(
                self.current_joint_state.position[idx]
            )

        return joints
def main(args=None):
    rclpy.init(args=args)
    node = MoveWithMoveIt()
    node.run()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()