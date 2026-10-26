import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


class ObstacleAI(Node):
    def __init__(self):
        super().__init__('obstacle_ai')

        self.pub = self.create_publisher(
            Twist, '/model/vehicle/cmd_vel', 10)
        self.create_subscription(
            LaserScan, '/lidar', self.on_scan, 10)
        self.create_subscription(
            Odometry, '/model/vehicle/odometry', self.on_odom, 10)

        self.x = self.y = self.yaw = None
        self.initial_yaw = None
        self.lane_y = None
        self.state = 'forward'
        self.waypoints = []
        self.target_index = 0
        self.side = 1
        self.passed = 0
        self.last_scan = None
        self.scan_count = 0

        self.get_logger().info('Obstacle AI ready; waiting for sensors.')

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y = p.x, p.y
        self.yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z))

        if self.initial_yaw is None:
            self.initial_yaw = self.yaw
            self.lane_y = self.y
            self.get_logger().info(
                f'Start heading recorded: {self.yaw:.3f} rad')

    def stop(self):
        self.pub.publish(Twist())

    def scan_values(self, scan, low_deg, high_deg):
        vals = []
        for i, d in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            if math.radians(low_deg) <= a <= math.radians(high_deg):
                if math.isfinite(d) and scan.range_min <= d <= scan.range_max:
                    vals.append((a, d))
                elif math.isinf(d):
                    vals.append((a, scan.range_max))
        return vals

    def estimate_obstacle(self, scan, yaw):
        # Use the closest cluster in the forward field of view.
        hits = []
        for i, d in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) > math.radians(48):
                continue
            if not math.isfinite(d) or d < scan.range_min or d > 3.0:
                continue
            hits.append((a, d))

        if not hits:
            return None

        nearest = min(d for _, d in hits)
        cluster = [(a, d) for a, d in hits if d <= nearest + 0.35]
        c, s = math.cos(yaw), math.sin(yaw)

        # LiDAR is mounted 0.3 m ahead of the robot's centre.
        world_points = []
        for a, d in cluster:
            lx = 0.3 + d * math.cos(a)
            ly = d * math.sin(a)
            wx = self.x + lx*c - ly*s
            wy = self.y + lx*s + ly*c
            world_points.append((wx, wy))

        hit_x = sum(p[0] for p in world_points) / len(world_points)
        hit_y = sum(p[1] for p in world_points) / len(world_points)

        # The box is 0.5 m long along X; estimate its centre from its front face.
        cx = hit_x + 0.25*c
        cy = hit_y + 0.25*s
        return cx, cy, nearest

    def begin_avoidance(self, scan):
        obstacle = self.estimate_obstacle(scan, self.yaw)
        if obstacle is None:
            return False

        cx, cy, distance = obstacle

        left = self.scan_values(scan, 25, 75)
        right = self.scan_values(scan, -75, -25)

        left_clear = sorted(d for _, d in left)[len(left)//2] if left else scan.range_max
        right_clear = sorted(d for _, d in right)[len(right)//2] if right else scan.range_max

        # Alternate sides unless the opposite side has substantially more space.
        chosen = self.side
        preferred_space = left_clear if chosen > 0 else right_clear
        other_space = right_clear if chosen > 0 else left_clear
        if other_space > preferred_space + 0.5:
            chosen *= -1

        # The lateral clearance includes obstacle half-width, robot half-width,
        # and an additional margin.
        offset = 0.6 + 0.2 + 0.25
        side_y = cy + chosen * offset

        # Approach beside the obstacle, pass it, then return to the initial lane.
        self.waypoints = [
            (cx - 0.70, side_y),
            (cx + 0.75, side_y),
            (cx + 1.45, self.lane_y)
        ]
        self.target_index = 0
        self.side = chosen
        self.state = 'navigate'

        self.get_logger().info(
            f'Obstacle detected at estimated ({cx:.2f}, {cy:.2f}), '
            f'distance={distance:.2f} m; '
            f'passing {"left" if chosen > 0 else "right"}')
        return True

    def navigate_to_waypoint(self, target):
        tx, ty = target
        dx, dy = tx - self.x, ty - self.y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        error = wrap_angle(desired_yaw - self.yaw)

        cmd = Twist()

        if distance < 0.22:
            self.target_index += 1
            if self.target_index >= len(self.waypoints):
                self.passed += 1
                self.get_logger().info(
                    f'Obstacle passed and lane restored: {self.passed}/7')

                if self.passed >= 7:
                    self.state = 'finished'
                    self.get_logger().info('Seven obstacles completed. Stopping.')
                else:
                    self.state = 'forward'
                    self.side *= -1
                return cmd

        # Rotate in place for large heading errors instead of driving in circles.
        if abs(error) > 0.45:
            cmd.linear.x = 0.0
            cmd.angular.z = clamp(1.1 * error, -0.35, 0.35)
        else:
            cmd.linear.x = 0.11 if distance > 0.5 else 0.06
            cmd.angular.z = clamp(1.3 * error, -0.28, 0.28)

        return cmd

    def on_scan(self, scan):
        self.last_scan = scan
        self.scan_count += 1

        if self.x is None or self.yaw is None or self.initial_yaw is None:
            self.stop()
            return

        front = self.scan_values(scan, -18, 18)
        front_min = min((d for _, d in front), default=scan.range_max)

        # Emergency stop only for a critically close obstacle.
        if front_min < 0.30:
            self.stop()
            self.get_logger().error(
                f'Emergency stop: obstacle at {front_min:.2f} m')
            return

        if self.state == 'finished':
            self.stop()
            return

        cmd = Twist()

        if self.state == 'forward':
            heading_error = wrap_angle(self.initial_yaw - self.yaw)

            # Detect early, while there is room to move toward a side waypoint.
            if front_min < 2.65:
                if self.begin_avoidance(scan):
                    cmd = self.navigate_to_waypoint(
                        self.waypoints[self.target_index])
                else:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.14
                cmd.angular.z = clamp(1.0 * heading_error, -0.18, 0.18)

        elif self.state == 'navigate':
            cmd = self.navigate_to_waypoint(
                self.waypoints[self.target_index])

        self.pub.publish(cmd)


def main():
    rclpy.init()
    node = ObstacleAI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
