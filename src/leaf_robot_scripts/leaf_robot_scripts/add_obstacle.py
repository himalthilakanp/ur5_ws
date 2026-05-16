#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class MoveItWithObjectUR5(Node):

    def __init__(self):
        super().__init__("add_obstacles")

        # MoveIt planning scene publisher
        self.scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10
        )

        # publish once after startup
        self.timer = self.create_timer(2.0, self.add_cylinders_once)

        self.sent = False

        self.get_logger().info(
            "Node started — waiting to publish cylinders..."
        )

    # -------------------------------------------------
    # CREATE CYLINDER OBJECT
    # -------------------------------------------------
    def create_cylinder(
        self,
        object_id,
        x,
        y,
        radius,
        height
    ):

        obj = CollisionObject()

        obj.id = object_id

        # IMPORTANT:
        # Must match MoveIt planning frame
        obj.header.frame_id = "base_footprint"

        # ---------------- SHAPE ----------------
        cylinder = SolidPrimitive()

        cylinder.type = SolidPrimitive.CYLINDER

        # [height, radius]
        cylinder.dimensions = [height, radius]

        # ---------------- POSE ----------------
        pose = Pose()

        pose.position.x = x
        pose.position.y = y

        # center of cylinder
        pose.position.z = height / 2.0

        pose.orientation.w = 1.0

        # ---------------- ATTACH ----------------
        obj.primitives.append(cylinder)
        obj.primitive_poses.append(pose)

        obj.operation = CollisionObject.ADD

        return obj

    # -------------------------------------------------
    # ADD CYLINDERS
    # -------------------------------------------------
    def add_cylinders_once(self):

        if self.sent:
            return

        # ---------------- PLANNING SCENE ----------------
        scene = PlanningScene()
        scene.is_diff = True

        # =================================================
        # BIG CYLINDERS
        # =================================================
        big_cylinder_positions = [

            # back-left
            #(-0.30, -0.45),

            # front-right
            #(0.30, 0.45),

            # front-left
            #(-0.30, 0.45),

            # back-right
            (0.30, -0.45),
        ]

        # =================================================
        # SMALL CYLINDERS
        # positioned INSIDE the 4 big cylinders
        # =================================================
        small_cylinder_positions = [

            # inside back-left
            (-0.30, -0.45),

            # inside front-right
            (0.30, 0.45),

            # inside front-left
            (-0.30, 0.45),

            # inside back-right
            (0.30, -0.45),
        ]

        # =================================================
        # CREATE BIG CYLINDERS
        # =================================================
        for i, (x, y) in enumerate(big_cylinder_positions):

            obj = self.create_cylinder(
                object_id=f"big_cylinder_{i}",
                x=x,
                y=y,
                radius=0.30,
                height=1.6
            )

            scene.world.collision_objects.append(obj)

        # =================================================
        # CREATE SMALL CYLINDERS
        # =================================================
        for i, (x, y) in enumerate(small_cylinder_positions):

            obj = self.create_cylinder(
                object_id=f"small_cylinder_{i}",
                x=x,
                y=y,
                radius=0.015,
                height=1.6
            )

            scene.world.collision_objects.append(obj)

        # =================================================
        # PUBLISH
        # =================================================
        self.scene_pub.publish(scene)

        self.get_logger().info(
            "4 big cylinders + 4 small cylinders added ✔"
        )

        self.sent = True


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    rclpy.init()

    node = MoveItWithObjectUR5()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()