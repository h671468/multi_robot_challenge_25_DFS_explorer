#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from geometry_msgs.msg import Twist

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from std_msgs.msg import String

# Importer alle komponenter
from .wall_follower import WallFollower
from .goal_navigator import GoalNavigator
from .bug2_navigator import Bug2Navigator
from .big_fire_coordinator import BigFireCoordinator
from .aruco_detector import ArUcoDetector
from .robot_memory import RobotMemory
from .sensor_manager import SensorManager
from .dfs_explorer import DfsExplorer


class SearchRescueCoordinator:
    """
    Search & Rescue koordinator - koordinerer alle komponenter
    """

    # Kollisjons- / møtekonstanter
    AVOID_DISTANCE_TRIGGER = 0.75
    AVOID_DISTANCE_RELEASE = 1.40
    AVOID_FRONT_THRESHOLD = 0.70
    AVOID_FRONT_RELEASE = 1.10
    AVOID_TIMEOUT = 2.0
    AVOID_STALE_TIME = 3.0
    AVOID_HEADING_DIFFERENCE = 0.6  # rad (~35°)
    AVOID_BACK_DURATION = 0.8
    AVOID_HOLD_MIN = 1.0
    AVOID_ADVANCE_DURATION = 0.7

    def __init__(self, node_ref: Node):
        self.node = node_ref
        self.robot_id = self.node.get_namespace().strip('/')

        self.robot_position = (0.0, 0.0)
        self.robot_orientation = 0.0

        self.sensor_manager = SensorManager(node_ref)
        self.robot_memory = RobotMemory()
        self.big_fire_coordinator = BigFireCoordinator(node_ref, self.robot_memory)
        self.aruco_detector = ArUcoDetector(node_ref, self.handle_aruco_detection)

        # Koble sensorens ArUco callback direkte til koordinatorens handler
        self.sensor_manager.aruco_callback = self.handle_aruco_detection

        try:
            robot_number = int(self.robot_id.rsplit('_', 1)[-1])
        except ValueError:
            robot_number = 0
        self.robot_number = robot_number

        follow_left = (robot_number % 2 == 1)
        self.wall_follower = WallFollower(node_ref, self.sensor_manager, follow_left=follow_left)
        side = 'venstre' if follow_left else 'høyre'
        self.node.get_logger().info(f'🧱 WallFollower konfig: følger {side} vegg')

        self.goal_navigator = GoalNavigator(node_ref, self.sensor_manager)
        self.bug2_navigator = Bug2Navigator(node_ref, self.wall_follower, self.goal_navigator)
        self.dfs_explorer = DfsExplorer()
        if hasattr(self.dfs_explorer, 'prefer_left'):
            self.dfs_explorer.prefer_left = follow_left

        # Del posisjon med andre roboter for trafikkregler
        self.robot_presence_pub = self.node.create_publisher(String, '/robot_presence', 10)
        self.robot_presence_sub = self.node.create_subscription(
            String, '/robot_presence', self.handle_robot_presence, 10
        )

        self.last_presence_pub_time = 0.0
        self.last_other_robot_update = None
        self.other_robot_id = None
        self.other_robot_number = None
        self.other_robot_yaw = None
        self.avoidance_active = False
        self.avoidance_release_time = 0.0
        self._avoidance_logged = False
        self.avoidance_mode = None
        self.avoidance_cooldown_until = 0.0
        self.avoidance_phase = None
        self.avoidance_phase_start = 0.0
        self.handled_big_fires = set()
        self.active_big_fire_key = None

        # --- ArUco sweep / scanning (for å oppdage vegg-merkene) ---
        self.last_aruco_check_position = (0.0, 0.0)
        self.last_aruco_check_time = 0.0
        self.aruco_check_interval = 30.0  # sekunder mellom sjekker
        self.aruco_check_distance = 4.0  # meter som må være flyttet før ny sjekk
        self.aruco_scan_angular_speed = 1.0  # rad/s for rotasjon
        self.aruco_half_turn_duration = (math.pi / 2.0) / self.aruco_scan_angular_speed
        self.aruco_scan_duration = self.aruco_half_turn_duration * 2.0 + 0.4
        self.is_doing_aruco_scan = False
        self.aruco_scan_start = 0.0
        self.pending_aruco_scan = False

        self.node.get_logger().info(f'🤖 SearchRescueCoordinator ({self.robot_id}) initialisert')

    # ---------------- ArUco sweep/hjelpefunksjoner ----------------

    def should_perform_aruco_scan(self) -> bool:
        """Return True dersom vi bør starte en ArUco-sveip."""
        if self.is_doing_aruco_scan:
            return False

        if self.pending_aruco_scan:
            if self.big_fire_coordinator.should_handle_big_fire() or self.avoidance_active:
                return False
            self.pending_aruco_scan = False
            return True

        if self.big_fire_coordinator.should_handle_big_fire() or self.avoidance_active:
            return False

        now = time.time()
        x, y = self.robot_position
        lx, ly = self.last_aruco_check_position
        dist = math.hypot(x - lx, y - ly)
        tdiff = now - self.last_aruco_check_time
        # Utfør sveip hvis roboten har flyttet seg mer enn 2 m ELLER det har gått 5 s
        if dist < self.aruco_check_distance and tdiff < self.aruco_check_interval:
            return False


        if self.big_fire_coordinator.should_handle_big_fire() or self.avoidance_active:
            self.pending_aruco_scan = True
            return False

        return True

    def start_aruco_scan(self):
        """Start en periode hvor roboten roterer for å se etter ArUco-markører."""
        self.node.get_logger().info(f'👀 Starter ArUco-sveip (pos={self.robot_position})')
        self.is_doing_aruco_scan = True
        self.aruco_scan_start = time.time()
        self.pending_aruco_scan = False
        try:
            self.bug2_navigator.stop_robot()
        except Exception:
            pass
        try:
            self.wall_follower.stop_robot()
        except Exception:
            pass

    def perform_aruco_rotation(self):
        """Publiser rotasjonskommando mens vi skanner 90° høyre og 90° venstre."""
        elapsed = time.time() - self.aruco_scan_start
        t = Twist()
        t.linear.x = 0.0
        rotation_duration = self.aruco_half_turn_duration

        # Roter først mot høyre, så mot venstre, deretter stopp
        if elapsed < rotation_duration:
            # 0–90°: høyre rotasjon
            t.angular.z = -self.aruco_scan_angular_speed
        elif elapsed < 2 * rotation_duration:
            # 90–180°: venstre rotasjon
            t.angular.z = self.aruco_scan_angular_speed
        else:
            # Ferdig med begge sider
            self.finish_aruco_scan()
            return

    # Publiser rotasjonskommando
        try:
            self.bug2_navigator.cmd_vel_pub.publish(t)
        except Exception:
            if not hasattr(self, '_tmp_cmd_pub'):
                self._tmp_cmd_pub = self.node.create_publisher(Twist, f'/{self.robot_id}/cmd_vel', 1)
            self._tmp_cmd_pub.publish(t)

    def finish_aruco_scan(self):
        """Avslutt sveip og oppdater tid/posisjon."""
        self.is_doing_aruco_scan = False
        self.last_aruco_check_position = self.robot_position
        self.last_aruco_check_time = time.time()
        self.pending_aruco_scan = False
        try:
            self.bug2_navigator.stop_robot()
        except Exception:
            pass
        self.node.get_logger().info('👁️ Ferdig ArUco-sveip — fortsetter normal navigasjon')

    # -------------------------------------------------------------

    def process_scan(self, msg: LaserScan):
        """Hovedfunksjon - koordinerer navigasjon. Kaller KUN ÉN navigasjonskontroller per syklus."""
        # Oppdater Big Fire state tidlig
        self.big_fire_coordinator.update_state(self.robot_position, self.robot_orientation)
        big_fire_active = self.big_fire_coordinator.should_handle_big_fire()

        if big_fire_active:
            # Pause DFS mens storbrann håndteres
            self.dfs_explorer.clear_current_goal()

            # Hent målet for Big Fire navigasjon
            target = self.big_fire_coordinator.get_target_position()

            # KORRIGERT FEIL: Bruker den nye is_moving_to_fire metoden
            if target and self.robot_memory.is_moving_to_fire():
                robot_pos = self.robot_memory.robot_position
                distance_to_target = math.hypot(
                    target[0] - robot_pos[0],
                    target[1] - robot_pos[1]
                )

                threshold = self.bug2_navigator.GOAL_THRESHOLD + 0.15
                if distance_to_target <= threshold:
                    self.node.get_logger().info(
                        f'🔥 Allerede ved brannposisjon (d={distance_to_target:.2f} m). Stopper og oppdaterer tilstand.'
                    )
                    self.bug2_navigator.stop_robot()
                    if self.robot_memory.my_role == self.robot_memory.LEDER:
                        self.robot_memory.transition_to_leder_waiting()
                    else:
                        if not self.robot_memory.i_am_at_fire:
                            self.big_fire_coordinator.publish_robot_at_fire()
                            self.node.get_logger().info('🔥 SUPPORTER: Bekrefter ankomst ved brannen.')
                        self.robot_memory.transition_to_extinguishing()
                    self.handle_big_fire_state_logic()
                    return

                self.node.get_logger().info(f'🔥 BUG2: Target={target}. State={self.robot_memory.big_fire_state}')
                self.bug2_navigator.set_goal(target)
                goal_reached = self.bug2_navigator.navigate(msg)
                if goal_reached:
                    self.bug2_navigator.stop_robot()
                    self.node.get_logger().info('🔥 BUG2: Mål nådd! Oppdaterer Big Fire state.')
                    if self.robot_memory.my_role == self.robot_memory.LEDER:
                        self.robot_memory.transition_to_leder_waiting()
                    else:
                        if not self.robot_memory.i_am_at_fire:
                            self.big_fire_coordinator.publish_robot_at_fire()
                            self.node.get_logger().info('🔥 SUPPORTER: Bekrefter ankomst ved brannen.')
                        self.robot_memory.transition_to_extinguishing()
            else:
                # Ingen bevegelse (Venter, slukker, eller nylig detektert)
                self.bug2_navigator.stop_robot()
                self.handle_big_fire_state_logic()

        else:
            # Normal utforskning / veggfølging / DFS

            # Oppdater pose til DFS-modulen
            self.dfs_explorer.update_pose(self.robot_position, self.robot_orientation)

            # Oppdater LIDAR-data i wall follower før vi leser åpninger
            self.wall_follower.process_laser_scan(msg)

            openings = self.wall_follower.get_openings()
            self.dfs_explorer.register_openings(openings)
            if self.dfs_explorer.has_pending_goals():
                self.node.get_logger().debug(
                    f'🧭 DFS: stakkstørrelse={self.dfs_explorer.pending_goal_count()} (aktivt mål={self.dfs_explorer.has_active_goal()})'
                )

            front_distance = self.wall_follower.regions.get('front', self.wall_follower.MAX_RANGE)
            if self.handle_robot_avoidance(front_distance):
                return

            # --- ArUco-sveip: hvis kriterier møtes, start / utfør / avslutt sveip ---
            if self.is_doing_aruco_scan:
                if time.time() - self.aruco_scan_start >= self.aruco_scan_duration:
                    self.finish_aruco_scan()
                else:
                    self.perform_aruco_rotation()
                return

            if self.should_perform_aruco_scan():
                self.start_aruco_scan()
                # publish initial rotation immediately
                self.perform_aruco_rotation()
                return

            # Ny Logikk i SearchRescueCoordinator.process_scan


            # 1. Prioriter AKTIVT DFS-MÅL (Bug2 skal navigere)
            if self.dfs_explorer.has_active_goal():
                goal_reached = self.bug2_navigator.navigate(msg)
                
                if goal_reached or self.bug2_navigator.was_goal_aborted():
                    self.bug2_navigator.stop_robot()
                    self.bug2_navigator.clear_goal()
                    self.dfs_explorer.goal_reached()
                    if goal_reached:
                         self.node.get_logger().info('🧭 DFS: delmål nådd')
                    else:
                         self.node.get_logger().warn('🧭 DFS: delmål utilgjengelig. Hopper videre.')
                return # Fortsett å navigere eller prosesser neste syklus

            # 2. Hvis ikke aktivt mål, men HAR VENTENDE MÅL (hent det neste)
            if self.dfs_explorer.has_pending_goals():
                next_goal = self.dfs_explorer.next_goal()
                if next_goal:
                    self.node.get_logger().info(f'🧭 DFS: nytt delmål {next_goal}')
                    self.bug2_navigator.set_goal(next_goal)
                    # Gå til toppen av process_scan i neste syklus for å starte Bug2-navigasjon
                    return
            
            # 3. Hvis ingen aktive eller ventende mål, fortsett veggfølging (WallFollower)
            self.bug2_navigator.clear_goal()
            self.wall_follower.follow_wall(msg)

    # -------------------------------------------------------------

    def process_odom(self, msg: Odometry):
        """Oppdater robot posisjon og orientering"""
        self.robot_position = self.sensor_manager.get_robot_position()
        self.robot_orientation = self.sensor_manager.get_robot_orientation()

        self.robot_memory.update_robot_pose(self.robot_position, self.robot_orientation)
        self.bug2_navigator.update_robot_pose(self.robot_position, self.robot_orientation)
        self.publish_robot_presence()

    def handle_big_fire_state_logic(self):
        """Håndterer KUN tilstandsoverganger og publisering."""
        coordinator = self.big_fire_coordinator
        current_state = coordinator.memory.big_fire_state

        # Merk: Siden koordinatoren og minneobjektet deler konstanter, kan vi bruke memory.KONSTANT

        if current_state == coordinator.memory.LEDER_WAITING:
            self.node.get_logger().info('🔥 LEDER: In LEDER_WAITING state!')
            if not coordinator.memory.i_am_at_fire:
                coordinator.publish_robot_at_fire()
            if coordinator.memory.other_robot_at_fire:
                coordinator.memory.transition_to_extinguishing()
                self.node.get_logger().info('🔥 LEDER: Supporter ankommet - begynner slukking!')

        elif current_state == coordinator.memory.EXTINGUISHING:
            self.node.get_logger().info('🔥 SLUKKING PÅGÅR!')
            if not coordinator.memory.fire_extinguished:
                if coordinator.memory.big_fire_position is not None:
                    key = self._big_fire_key(coordinator.memory.big_fire_position)
                    self.handled_big_fires.add(key)
                    self.active_big_fire_key = None
                coordinator.publish_fire_extinguished()
                self.node.get_logger().info('🔥 Brannen slukket! Roboter returnerer til normal utforskning.')
                coordinator.memory.transition_to_normal()
                self.dfs_explorer.reset()
                self.bug2_navigator.clear_goal()

        elif current_state == coordinator.memory.NORMAL:
            if coordinator.memory.big_fire_detected_by_me:
                self.node.get_logger().info('🔥 LEDER: Jeg oppdaget Big Fire - starter navigasjon!')
                coordinator.memory.transition_to_leder_going_to_fire()
            elif coordinator.memory.big_fire_detected_by_other:
                self.node.get_logger().info('🔥 SUPPORTER: Mottok Big Fire melding - starter navigasjon!')
                coordinator.memory.transition_to_supporter_going_to_fire()

    def handle_aruco_detection(self, marker_id: int, position: tuple):
        """Håndterer ArUco marker detection"""
        # Only log once per marker to avoid spam
        if not hasattr(self, '_processed_aruco_markers'):
            self._processed_aruco_markers = set()

        marker_key = f"{marker_id}_{position[0]:.1f}_{position[1]:.1f}"
        if marker_key in self._processed_aruco_markers:
            return  # Already processed this marker at this position

        self._processed_aruco_markers.add(marker_key)

        if marker_id == 4:  # Big Fire
            big_fire_key = self._big_fire_key(position)
            if self.big_fire_coordinator.should_handle_big_fire():
                # Allerede i gang med denne hendelsen
                return
            if big_fire_key in self.handled_big_fires:
                self.node.get_logger().info('🔥 Big Fire på denne posisjonen er allerede slukket. Ignorerer.')
                return
            self.active_big_fire_key = big_fire_key
            self.bug2_navigator.stop_robot()
            self.wall_follower.stop_robot()
            self.node.get_logger().info(f'🛑 ROBOT STOPPED! ArUco ID {marker_id} oppdaget på {position}')
            self.node.get_logger().info(f'🔥 BIG FIRE DETECTED! Calling detect_big_fire({position})')
            self.big_fire_coordinator.detect_big_fire(position)
            # Kaller update_state umiddelbart for å sette i gang navigasjonen i neste process_scan
            self.big_fire_coordinator.update_state(self.robot_position, self.robot_orientation)
        else:
            self.node.get_logger().info(f'📊 ArUco ID {marker_id} på {position} - registrert for scoring, fortsetter.')

    # --- Trafikkregler mellom roboter ---
    def publish_robot_presence(self):
        """Publiser egen posisjon periodisk slik at andre roboter kan planlegge."""
        now = self.get_time_seconds()
        if now - self.last_presence_pub_time < 0.4:
            return

        msg = String()
        msg.data = (
            f"POSE:{self.robot_id}:"
            f"{self.robot_position[0]:.3f}:{self.robot_position[1]:.3f}:"
            f"{self.robot_orientation:.3f}:{self.robot_number}"
        )
        self.robot_presence_pub.publish(msg)
        self.last_presence_pub_time = now

    def handle_robot_presence(self, msg: String):
        """Motta posisjonsinfo fra andre roboter."""
        try:
            parts = msg.data.split(':')
            if len(parts) != 6 or parts[0] != "POSE":
                return
            sender_id = parts[1]
            if sender_id == self.robot_id:
                return
            x = float(parts[2])
            y = float(parts[3])
            yaw = float(parts[4])
            number = int(parts[5])
        except (ValueError, IndexError):
            return

        self.robot_memory.other_robot_position = (x, y)
        self.other_robot_id = sender_id
        self.other_robot_number = number
        self.other_robot_yaw = yaw
        self.last_other_robot_update = self.get_time_seconds()

    def handle_robot_avoidance(self, front_distance: float) -> bool:
        """
        Enkle trafikkregler: lavest robot-ID får forkjørsrett.
        Returnerer True når vi skal pause annen logikk i denne syklusen.
        """
        now = self.get_time_seconds()

        if now < self.avoidance_cooldown_until:
            return False

        if (
            self.robot_memory.other_robot_position is None
            or self.last_other_robot_update is None
            or (now - self.last_other_robot_update) > self.AVOID_STALE_TIME
        ):
            if self.avoidance_active:
                self._reset_avoidance_state()
                self.node.get_logger().info('🤝 Ingen annen robot i nærheten. Fortsetter normal drift.')
            return False

        other_x, other_y = self.robot_memory.other_robot_position
        distance = math.hypot(other_x - self.robot_position[0], other_y - self.robot_position[1])

        if self.avoidance_active:
            if self.avoidance_mode == 'yield':
                if self._handle_yield_phase(now, distance, front_distance):
                    return True
                if not self.avoidance_active:
                    return False

            if (
                distance > self.AVOID_DISTANCE_RELEASE
                and self._heading_difference_ok(self.avoidance_mode)
            ) or (
                distance > (self.AVOID_DISTANCE_RELEASE * 1.6)
            ) or (
                front_distance > self.AVOID_FRONT_RELEASE
            ) or (
                now >= self.avoidance_release_time
            ):
                self._reset_avoidance_state(cooldown=1.0)
                self.node.get_logger().info('🤝 Konflikt løst, fortsetter veggfølging.')
            else:
                self.bug2_navigator.stop_robot()
                self.wall_follower.hold_position_against_wall()
                return True

        if distance > self.AVOID_DISTANCE_TRIGGER or front_distance > self.AVOID_FRONT_THRESHOLD:
            self._avoidance_logged = False
            return False

        my_number = self.robot_number if self.robot_number is not None else 0
        other_number = self.other_robot_number if self.other_robot_number is not None else 999

        if my_number <= other_number:
            if not self._avoidance_logged:
                target = self.other_robot_id or 'ukjent'
                self.node.get_logger().info(f'🤝 Møter {target}, beholder forkjørsrett.')
                self._avoidance_logged = True
            self.avoidance_mode = 'priority'
            return False

        self.avoidance_active = True
        self.avoidance_release_time = now + self.AVOID_TIMEOUT
        self.avoidance_mode = 'yield'
        self.avoidance_phase = 'backing'
        self.avoidance_phase_start = now
        other_label = self.other_robot_id or 'ukjent'
        self.node.get_logger().info(f'🤝 Gir forkjørsrett til {other_label}. Kjører til side.')
        self.bug2_navigator.stop_robot()
        self.wall_follower.backup_along_wall()
        return True

    def get_time_seconds(self) -> float:
        """Hent ROS-klokkens tid i sekunder."""
        return self.node.get_clock().now().nanoseconds / 1e9

    def _heading_difference_ok(self, mode: str) -> bool:
        """Sjekk om forskjellen i heading tilsier at vi kan slippe videre."""
        if self.other_robot_yaw is None:
            return True
        diff = abs(self._normalize_angle(self.robot_orientation - self.other_robot_yaw))
        if mode == 'yield':
            return diff > self.AVOID_HEADING_DIFFERENCE
        return True

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliser vinkel til [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _handle_yield_phase(self, now: float, distance: float, front_distance: float) -> bool:
        """Utfør trinnvis vike-manøver for roboten som må stoppe."""
        elapsed = now - self.avoidance_phase_start

        if self.avoidance_phase == 'backing':
            if elapsed < self.AVOID_BACK_DURATION:
                self.wall_follower.backup_along_wall()
                self.bug2_navigator.stop_robot()
                return True
            self.avoidance_phase = 'holding'
            self.avoidance_phase_start = now
            elapsed = 0.0

        if self.avoidance_phase == 'holding':
            self.wall_follower.hold_position_against_wall()
            self.bug2_navigator.stop_robot()
            if (
                elapsed >= self.AVOID_HOLD_MIN
                and distance > self.AVOID_DISTANCE_RELEASE
                and self._heading_difference_ok('yield')
            ):
                self.avoidance_phase = 'returning'
                self.avoidance_phase_start = now
            return True

        if self.avoidance_phase == 'returning':
            if elapsed < self.AVOID_ADVANCE_DURATION:
                self.wall_follower.advance_along_wall()
                self.bug2_navigator.stop_robot()
                return True
            self._reset_avoidance_state(cooldown=1.0)
            self.node.get_logger().info('🤝 Vike-manøver ferdig. Fortsetter veggfølging.')
            return False

        return False

    def _reset_avoidance_state(self, cooldown: float = 0.0):
        """Nullstill tilstand for møte med annen robot."""
        self.avoidance_active = False
        self.avoidance_mode = None
        self.avoidance_phase = None
        self._avoidance_logged = False
        self.robot_memory.other_robot_position = None
        self.other_robot_id = None
        self.other_robot_number = None
        self.other_robot_yaw = None
        self.avoidance_phase_start = 0.0
        self.avoidance_release_time = 0.0
        self.avoidance_cooldown_until = self.get_time_seconds() + cooldown

    def _big_fire_key(self, position: tuple, precision: float = 0.5) -> tuple:
        """Generer en avrundet nøkkel for en Big Fire-posisjon."""
        return (round(position[0] / precision) * precision,
                round(position[1] / precision) * precision)