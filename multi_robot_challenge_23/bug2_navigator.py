#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from .wall_follower import WallFollower
from .goal_navigator import GoalNavigator # Antas å eksistere

class Bug2Navigator:
    """
    BUG2-koordinator (Controller) - Administrerer Bug2-tilstander og DELEGERER bevegelse.
    """
    
    # --- INNSTILLINGER ---
    FRONT_THRESHOLD = 0.8       # Avstand for å detektere hindring og bytte til WALL_FOLLOWING
    GOAL_THRESHOLD = 0.45       # Avstand for å betrakte målet som nådd (økt for Big Fire)
    ANGLE_TOLERANCE = 20        # Vinkeltoleranse for M-linje (grader)
    MIN_DIST_CHECK = 1.0        # Minimum avstand for å sjekke M-linje
    
    # BUG2 States
    GO_TO_GOAL = "GO_TO_GOAL"
    WALL_FOLLOWING = "WALL_FOLLOWING"
    
    def __init__(self, node_ref: Node, wall_follower: WallFollower, goal_navigator: GoalNavigator):
        self.node = node_ref
        
        # Delegerte Navigatører
        self.wall_follower = wall_follower
        self.goal_navigator = goal_navigator
        
        self.state = self.GO_TO_GOAL
        self.target_position = None
        self.robot_position = (0.0, 0.0)
        self.robot_orientation = 0.0
        
        # BUG2 spesifikke variabler
        self.start_x = 0.0          # Startposisjon X
        self.start_y = 0.0          # Startposisjon Y
        self.target_x = 0.0         # Målposisjon X
        self.target_y = 0.0         # Målposisjon Y
        self.M_start_x = 0.0        # M-linje start X
        self.M_start_y = 0.0        # M-linje start Y
        self.M_start_dist_to_goal = float('inf')  # Avstand til mål ved M-linje start
        self.M_line_angle = 0.0     # Vinkel til M-linjen
        self.regions = {}           # Laser scan regions

        # Wall-follow tracking
        self.wall_follow_start_time = None
        self.wall_follow_start_dist = float('inf')
        self.MIN_WALL_FOLLOW_TIME = 2.0
        self.LEAVE_DISTANCE_IMPROVEMENT = 1.2
        self.LEAVE_ANGLE_TOLERANCE = math.radians(15.0)
        self.MIN_FRONT_CLEARANCE = 1.5
        
        self.node.get_logger().info('🐛 Delegert Bug2Navigator initialisert')
        self.goal_aborted = False

    def set_goal(self, position: tuple):
        """Setter nytt mål og oppdaterer GoalNavigator."""
        if self.target_position == position and self.target_position is not None:
            return

        self.target_position = position
        self.target_x = position[0]
        self.target_y = position[1]
        self.start_x = self.robot_position[0]
        self.start_y = self.robot_position[1]
        self.state = self.GO_TO_GOAL
        self.wall_follow_start_time = None
        self.wall_follow_start_dist = float('inf')
        self.goal_aborted = False
        
        # Reset M-line variables
        self.M_start_dist_to_goal = float('inf')
        self.M_start_x = 0.0
        self.M_start_y = 0.0
        
        # Calculate M-line angle
        self.M_line_angle = math.atan2(self.target_y - self.start_y, self.target_x - self.start_x)
        
        self.goal_navigator.set_goal(position)
        
        self.node.get_logger().info(f'🐛 BUG2: Nytt mål satt: {position}. Starter GO_TO_GOAL.')

    def clear_goal(self):
        """Fjern aktivt mål og stopp bevegelse."""
        self.target_position = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.state = self.GO_TO_GOAL
        self.M_start_dist_to_goal = float('inf')
        self.M_start_x = 0.0
        self.M_start_y = 0.0
        self.goal_navigator.clear_goal()
        self.stop_robot()
        self.goal_aborted = False

    def update_robot_pose(self, position: tuple, orientation: float):
        """Oppdater robot posisjon og orientering."""
        self.robot_position = position
        self.robot_orientation = orientation
        self.goal_navigator.update_robot_pose(position, orientation)

    def navigate(self, msg: LaserScan) -> bool:
        """
        Hovedfunksjon: Utfører tilstandsoverganger og delegerer bevegelse.
        """
        if not self.target_position:
            self.stop_robot()
            return False

        # Process laser scan for obstacle detection
        self.process_laser_scan(msg)

        # 1. Sjekk om mål er nådd
        current_dist_to_goal = self.get_current_distance_to_goal()
        if current_dist_to_goal < self.GOAL_THRESHOLD:
            self.node.get_logger().info(f"🎯 Målet nådd! Avstand: {current_dist_to_goal:.2f}")
            self.stop_robot()
            return True

        # 2. State Machine
        if self.state == self.GO_TO_GOAL:
            self._go_to_goal_state(msg)
        
        elif self.state == self.WALL_FOLLOWING:
            self._wall_following_state(msg)
            
        return False

    # --- BUG2 STATE LOGIKK OG DELEGERING ---

    def _go_to_goal_state(self, msg: LaserScan):
        """Tilstand: Gå direkte mot målet. Bytter til WALL_FOLLOWING ved hindring."""
        
        front_distance = self.regions.get('front', 10.0)
        
        # Only log Go-To-Goal state every 20th time
        if not hasattr(self, '_go_to_goal_log_counter'):
            self._go_to_goal_log_counter = 0
        self._go_to_goal_log_counter += 1
        
        if self._go_to_goal_log_counter % 50 == 0:
            self.node.get_logger().info(
                f'🎯 GO_TO_GOAL: front_dist={front_distance:.2f}m, threshold={self.FRONT_THRESHOLD}m, '
                f'target={self.target_position}, robot={self.robot_position}'
            )
        
        if front_distance < self.FRONT_THRESHOLD:
            # TRANSISJON: GO_TO_GOAL -> WALL_FOLLOWING
            self.node.get_logger().warn(f"🛑 BUG2: Hinder funnet ({front_distance:.2f} m). Bytt til Wall Follow.")
            
            # Set M-line start point
            self.M_start_x = self.robot_position[0]
            self.M_start_y = self.robot_position[1]
            self.M_start_dist_to_goal = self.get_current_distance_to_goal()
            self.wall_follow_start_time = time.time()
            self.wall_follow_start_dist = self.M_start_dist_to_goal
            
            # Change state
            self.state = self.WALL_FOLLOWING
            return
        
        # DELEGER til GoalNavigator
        self.goal_navigator.navigate_to_goal(msg) 

    def _wall_following_state(self, msg: LaserScan):
        """Tilstand: Følg veggen til exit-kriteriene er møtt."""
        
        current_dist_to_goal = self.get_current_distance_to_goal()
        
        # Sjekk for leave point
        is_on_line = self.is_on_M_line()
        is_closer_to_goal = current_dist_to_goal < self.M_start_dist_to_goal

        if self.wall_follow_start_time is None:
            self.wall_follow_start_time = time.time()
        elapsed = time.time() - self.wall_follow_start_time

        # Only log wall following state every 15th time
        if not hasattr(self, '_wall_follow_log_counter'):
            self._wall_follow_log_counter = 0
        self._wall_follow_log_counter += 1
        
        if self._wall_follow_log_counter % 30 == 0:
            self.node.get_logger().info(
                f'🐛 WALL_FOLLOWING: on_m_line={is_on_line}, closer_to_goal={is_closer_to_goal}, '
                f'current_dist={current_dist_to_goal:.2f}, M_start_dist={self.M_start_dist_to_goal:.2f}'
            )

        # Alternative exit: if front is clear and we're closer to goal, exit wall following
        front_distance = self.regions.get('front', 10.0)
        
        progress_ratio = (self.M_start_dist_to_goal - current_dist_to_goal) if self.M_start_dist_to_goal != float('inf') else 0.0
        progress_ratio = max(progress_ratio, 0.0)
        max_allowable = max(0.5, 4.0 - progress_ratio)

        if elapsed < self.MIN_WALL_FOLLOW_TIME:
            self.wall_follower.follow_wall(msg)
            return

        if is_on_line and is_closer_to_goal:
            self.node.get_logger().warn(
                f"✅ BUG2: Kan forlate vegg: På M-linjen OG nærmere målet. "
                f"Nåværende avstand: {current_dist_to_goal:.2f}, "
                f"M_start avstand: {self.M_start_dist_to_goal:.2f}"
            )
            self.state = self.GO_TO_GOAL
            return
        elif front_distance > self.FRONT_THRESHOLD * 1.5 and is_closer_to_goal:
            # Alternative exit: front is clear and we're closer to goal
            self.node.get_logger().warn(
                f"✅ BUG2: Front fri og nærmere målet. Front: {front_distance:.2f}m, "
                f"Nåværende avstand: {current_dist_to_goal:.2f}, M_start: {self.M_start_dist_to_goal:.2f}"
            )
            self.state = self.GO_TO_GOAL
            return
        
        if current_dist_to_goal > (self.M_start_dist_to_goal + max_allowable): 
            self.node.get_logger().error(
                f"❌ BUG2: Gikk for langt vekk fra målet ({current_dist_to_goal:.2f}m, grense {self.M_start_dist_to_goal + max_allowable:.2f}m). Stopper navigasjon."
            )
            self.abort_goal()
            return

        improvement = (self.wall_follow_start_dist - current_dist_to_goal)
        angle_to_goal = math.atan2(self.target_y - self.robot_position[1], self.target_x - self.robot_position[0])
        angle_diff = abs(self.normalize_angle(angle_to_goal - self.M_line_angle))

        can_leave = (
            improvement > self.LEAVE_DISTANCE_IMPROVEMENT and
            angle_diff < self.LEAVE_ANGLE_TOLERANCE and
            is_closer_to_goal and
            front_distance > self.MIN_FRONT_CLEARANCE
        )

        if can_leave:
            self.node.get_logger().info(
                f"✅ BUG2: Forlater vegg – forbedring {improvement:.2f}m, front {front_distance:.2f}m, "
                f"vinkel {math.degrees(angle_diff):.1f}°, tid {elapsed:.1f}s"
            )
            self.state = self.GO_TO_GOAL
            self.wall_follow_start_time = None
            self.wall_follow_start_dist = float('inf')
            return

        # Fortsett wall following - DELEGER til WallFollower
        self.wall_follower.follow_wall(msg)

    # --- BUG2 HELPER METHODS ---
    
    def process_laser_scan(self, msg: LaserScan):
        """Process laser scan and update regions"""
        import numpy as np
        ranges = np.array(msg.ranges)
        n = len(ranges)
        front_ranges = np.concatenate((ranges[0:int(n*0.05)], ranges[int(n*0.95):]))
        front_ranges[np.isinf(front_ranges)] = 10.0 
        self.regions['front'] = np.min(front_ranges) if len(front_ranges) > 0 else 10.0

    def get_current_distance_to_goal(self) -> float:
        """Get current distance to goal"""
        if not self.target_position:
            return float('inf')
        return math.hypot(self.target_x - self.robot_position[0], self.target_y - self.robot_position[1])

    def is_on_M_line(self) -> bool:
        """Check if robot is on M-line - simplified version"""
        if not self.target_position:
            return False

        current_x = self.robot_position[0]
        current_y = self.robot_position[1]
        
        # Vinkel fra robotens nåværende posisjon til målet
        angle_to_target = math.atan2(self.target_y - current_y, self.target_x - current_x)
        
        # Vinkel mellom M-linjen og robotens siktlinje til målet
        angle_diff = abs(self.normalize_angle(angle_to_target - self.M_line_angle))
        
        # Much more lenient angle tolerance (60 degrees instead of 20)
        ANGLE_TOLERANCE_RAD = math.radians(60)  # Increased from 20 to 60 degrees
        dist_to_goal = self.get_current_distance_to_goal()
        
        # Simplified Bug2 Leave-kriterium - just check if we're closer and roughly on the right path
        if angle_diff < ANGLE_TOLERANCE_RAD and dist_to_goal < self.M_start_dist_to_goal:
            self.node.get_logger().info(f"🐛 Nær M-linjen: Vinkelavvik: {math.degrees(angle_diff):.1f} deg, Nåværende dist: {dist_to_goal:.2f} m.")
            return True
        return False
        
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        angle = math.fmod(angle + math.pi, 2 * math.pi)
        if angle < 0:
            angle += 2 * math.pi
        return angle - math.pi
    
    # --- KONTROLL ---
    
    def stop_robot(self):
        """Stopper robot bevegelse via de delegerte navigatørene."""
        self.goal_navigator.stop_robot()
        self.wall_follower.stop_robot()

    def abort_goal(self):
        """Avslutt aktivt mål fordi det virker utilgjengelig."""
        if self.target_position is None:
            return
        self.goal_aborted = True
        self.target_position = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.state = self.GO_TO_GOAL
        self.M_start_dist_to_goal = float('inf')
        self.M_start_x = 0.0
        self.M_start_y = 0.0
        self.goal_navigator.clear_goal()
        self.stop_robot()

    def was_goal_aborted(self) -> bool:
        """Returner True én gang dersom siste mål ble avbrutt."""
        if self.goal_aborted:
            self.goal_aborted = False
            return True
        return False