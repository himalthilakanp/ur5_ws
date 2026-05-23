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

        self.get_logger().info("Waiting for IK service...")
        self.ik_client.wait_for_service()
        self.get_logger().info("IK service ready ✔")

        #self.add_cylinder()
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

            # bigger temporary debug size
            marker.scale.x = 0.06
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
        LEAF_RADIUS = 0.055

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

        target_leaf = 14

        # ---------------------------------
        # REMOVE VISUAL + ENABLE COLLISION
        # ---------------------------------
        self.remove_leaf_marker(target_leaf)

        self.enable_leaf_by_index(
            target_leaf
        )

        # ---------------------------------
        # SAFE PRE-GRASP POSE
        # ---------------------------------
        p2 = [
            0.122,
            0.368,
           -2.312,
            1.194,
            -0.090,
            0.00
        ]

        self.move_to_joints(
            p2,
            "P2"
        )

        # ---------------------------------
        # OPEN GRIPPER BEFORE APPROACH
        # ---------------------------------
        self.move_gripper(0.0)

        # ---------------------------------
        # MOVE TO LEAF USING IK
        # ---------------------------------
        self.grab_leaf_ik(
            target_leaf
        )

        self.get_logger().info(
            "PICK COMPLETE ✔"
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
    
    def solve_ik(self, x, y, z, yaw=0.0):

        request = GetPositionIK.Request()

        request.ik_request.group_name = "arm"
        request.ik_request.ik_link_name = "gripper_base_link"
        request.ik_request.timeout.sec = 2

        pose = PoseStamped()
        pose.header.frame_id = "base_footprint"

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        q = self.euler_to_quaternion(
            math.pi,   # tool facing downward
            0.0,
            yaw
        )

        pose.pose.orientation = q

        request.ik_request.pose_stamped = pose

        future = self.ik_client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future
        )

        result = future.result()

        if result.error_code.val != 1:
            self.get_logger().error(
                "IK failed ❌"
            )
            return None

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

        for joint_name in arm_joint_names:

            idx = joint_state.name.index(
                joint_name
            )

            joints.append(
                joint_state.position[idx]
            )

        return joints
    def grab_leaf_ik(self, leaf_index):

        GOLDEN_ANGLE = math.radians(137.5)

        NUM_LEAVES = 24
        TOTAL_HEIGHT = 1.35
        LEAF_RADIUS = 0.055

        START_ANGLE = math.pi

        theta = START_ANGLE + leaf_index * GOLDEN_ANGLE

        z = 0.20 + (
            leaf_index / NUM_LEAVES
        ) * TOTAL_HEIGHT

        x = 0.3 + LEAF_RADIUS * math.cos(theta)
        y = -0.45 + LEAF_RADIUS * math.sin(theta)

        self.get_logger().info(
            f"Leaf {leaf_index} at "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}"
        )

        # remove visual leaf
        self.remove_leaf_marker(
            leaf_index
        )

        # enable collision
        self.enable_leaf_collision(
            f"leaf_{leaf_index}",
            x,
            y,
            z,
            theta
        )

        # approach from outside of stem
        approach_distance = 0.08

        dx = math.cos(theta)
        dy = math.sin(theta)

        target_x = x + dx * approach_distance
        target_y = y + dy * approach_distance
        target_z = z

        joints = self.solve_ik(
            target_x,
            target_y,
            target_z,
            yaw=theta+ math.pi
        )

        if joints is None:
            return

        self.move_gripper(0.0)

        self.move_to_joints(
            joints,
            f"LEAF_{leaf_index}_APPROACH"
        )

        self.move_gripper(0.035)

        self.attach_leaf(
            f"leaf_{leaf_index}"
        )

        self.get_logger().info(
            f"Leaf {leaf_index} grabbed ✔"
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