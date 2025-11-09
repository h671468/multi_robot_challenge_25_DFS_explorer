#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class OccupancyGridManager:
    """
    Occupancy Grid Manager - EN ansvar: Håndtering av occupancy grid data
    
    Single Responsibility: Kun occupancy grid prosessering og koordinattransformasjoner
    """
    
    def __init__(self, node_ref: Node):
        self.node = node_ref
        
        # Occupancy grid data
        self.map_msg = None
        self.map_available = False
        
        # Setup QoS profile for map subscription
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            depth=5,
        )
        
        # Setup subscribers and publishers
        self.setup_map_subscription(qos_profile)
        self.setup_marker_publisher()
        
        self.node.get_logger().info('🗺️ OccupancyGridManager initialisert')

    def setup_map_subscription(self, qos_profile):
        """Sett opp map subscription med riktig QoS"""
        self.map_sub = self.node.create_subscription(
            OccupancyGrid, '/map', self.map_callback, qos_profile
        )

    def setup_marker_publisher(self):
        """Sett opp marker publisher for visualisering"""
        self.marker_pub = self.node.create_publisher(Marker, 'map_markers', 10)

    def map_callback(self, msg: OccupancyGrid):
        """Håndterer occupancy grid data"""
        self.map_msg = msg
        self.map_available = True
        self.node.get_logger().info(f'🗺️ Map mottatt: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}')

    def is_map_available(self) -> bool:
        """Sjekk om map er tilgjengelig"""
        return self.map_available and self.map_msg is not None

    def get_map_pos(self, map_iter: int) -> list:
        """
        Translate map list integer into x and y map coordinates
        
        Args:
            map_iter: Index i map array
            
        Returns:
            [x, y] map coordinates
        """
        if not self.is_map_available():
            return [0, 0]
        
        x = int(map_iter / self.map_msg.info.width)
        y = int(map_iter - x * self.map_msg.info.width)
        return [x, y]

    def get_world_pos(self, x: int, y: int) -> Point:
        """
        Translate map coordinates into world position coordinates in meters
        
        Args:
            x, y: Map coordinates
            
        Returns:
            Point: World position in meters
        """
        if not self.is_map_available():
            return Point()
        
        map_position = Point()
        map_position.x = self.map_msg.info.origin.position.x + y * self.map_msg.info.resolution
        map_position.y = self.map_msg.info.origin.position.y + x * self.map_msg.info.resolution
        map_position.z = 0.0
        
        return map_position

    def world_to_map(self, world_x: float, world_y: float) -> tuple:
        """Konverter verdenskoordinater til map-indekser."""
        if not self.is_map_available():
            return None

        origin_x = self.map_msg.info.origin.position.x
        origin_y = self.map_msg.info.origin.position.y
        resolution = self.map_msg.info.resolution

        map_y = int((world_x - origin_x) / resolution)
        map_x = int((world_y - origin_y) / resolution)

        if map_x < 0 or map_y < 0 or map_x >= self.map_msg.info.height or map_y >= self.map_msg.info.width:
            return None

        return (map_x, map_y)

    def get_map_iter(self, x: int, y: int) -> int:
        """
        Translate x and y map coordinates into corresponding map list integer value
        
        Args:
            x, y: Map coordinates
            
        Returns:
            int: Map array index
        """
        if not self.is_map_available():
            return 0
        
        map_iter = x * self.map_msg.info.width + y
        return map_iter

    def get_occupancy_value(self, x: int, y: int) -> int:
        """
        Hent occupancy verdi for gitt map koordinater
        
        Args:
            x, y: Map coordinates
            
        Returns:
            int: Occupancy value (-1=unknown, 0-100=probability)
        """
        if not self.is_map_available():
            return -1
        
        if x < 0 or x >= self.map_msg.info.height or y < 0 or y >= self.map_msg.info.width:
            return -1
        
        map_iter = self.get_map_iter(x, y)
        if map_iter < 0 or map_iter >= len(self.map_msg.data):
            return -1
        
        return self.map_msg.data[map_iter]

    def is_obstacle(self, x: int, y: int, threshold: int = 50) -> bool:
        """
        Sjekk om det er en hindring på gitt posisjon
        
        Args:
            x, y: Map coordinates
            threshold: Obstacle threshold (default 50)
            
        Returns:
            bool: True hvis hindring
        """
        occupancy = self.get_occupancy_value(x, y)
        return occupancy > threshold

    def is_free_space(self, x: int, y: int) -> bool:
        """
        Sjekk om det er fritt område på gitt posisjon
        
        Args:
            x, y: Map coordinates
            
        Returns:
            bool: True hvis fritt område
        """
        occupancy = self.get_occupancy_value(x, y)
        return occupancy == 0

    def is_unknown(self, x: int, y: int) -> bool:
        """
        Sjekk om området er ukjent
        
        Args:
            x, y: Map coordinates
            
        Returns:
            bool: True hvis ukjent
        """
        occupancy = self.get_occupancy_value(x, y)
        return occupancy == -1

    def has_line_of_sight(self, start: Point, end: Point, step: float = 0.05, obstacle_threshold: int = 50) -> bool:
        """Sjekk om det finnes fri sikt mellom to punkter i kartet."""
        if not self.is_map_available():
            return True

        distance = math.hypot(end.x - start.x, end.y - start.y)
        steps = max(int(distance / step), 1)

        for i in range(steps + 1):
            t = i / steps
            sample_x = start.x + (end.x - start.x) * t
            sample_y = start.y + (end.y - start.y) * t
            map_indices = self.world_to_map(sample_x, sample_y)
            if map_indices is None:
                return False
            if self.get_occupancy_value(map_indices[0], map_indices[1]) > obstacle_threshold:
                return False

        return True

    def find_bounding_box(self) -> dict:
        """
        Finn bounding box rundt ytre vegger av kartet
        
        Returns:
            dict: {'x_min', 'x_max', 'y_min', 'y_max', 'corners'}
        """
        if not self.is_map_available():
            return {}
        
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')
        
        # Scan gjennom hele kartet for å finne vegger
        for x in range(self.map_msg.info.height):
            for y in range(self.map_msg.info.width):
                if self.is_obstacle(x, y):
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)
        
        # Konverter til world koordinater
        corners = {
            'bottom_left': self.get_world_pos(x_min, y_min),
            'top_left': self.get_world_pos(x_min, y_max),
            'bottom_right': self.get_world_pos(x_max, y_min),
            'top_right': self.get_world_pos(x_max, y_max)
        }
        
        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'corners': corners
        }

    def visualize_bounding_box(self):
        """Visualiser bounding box i RViz"""
        if not self.is_map_available():
            return
        
        bounding_box = self.find_bounding_box()
        if not bounding_box:
            return
        
        # Opprett marker
        marker_msg = Marker()
        marker_msg.header.frame_id = "/map"
        marker_msg.header.stamp = self.node.get_clock().now().to_msg()
        marker_msg.id = 0
        marker_msg.type = Marker.POINTS
        marker_msg.action = Marker.ADD
        marker_msg.pose.orientation.w = 1.0
        marker_msg.scale.x = 0.2
        marker_msg.scale.y = 0.2
        marker_msg.color.r = 0.0
        marker_msg.color.g = 1.0
        marker_msg.color.b = 0.0
        marker_msg.color.a = 1.0
        marker_msg.lifetime = rclpy.duration.Duration().to_msg()
        
        # Legg til hjørnepunkter
        corners = bounding_box['corners']
        marker_msg.points = [
            corners['bottom_left'],
            corners['top_left'],
            corners['bottom_right'],
            corners['top_right']
        ]
        
        # Publiser marker
        self.marker_pub.publish(marker_msg)
        self.node.get_logger().info('🗺️ Bounding box visualisert i RViz')

    def get_map_info(self) -> dict:
        """Hent map informasjon"""
        if not self.is_map_available():
            return {}
        
        return {
            'width': self.map_msg.info.width,
            'height': self.map_msg.info.height,
            'resolution': self.map_msg.info.resolution,
            'origin_x': self.map_msg.info.origin.position.x,
            'origin_y': self.map_msg.info.origin.position.y,
            'data_size': len(self.map_msg.data)
        }

    def world_to_map(self, world_x: float, world_y: float) -> tuple:
        """
        Konverter world koordinater til map koordinater
        
        Args:
            world_x, world_y: World koordinater i meter
            
        Returns:
            tuple: (map_x, map_y)
        """
        if not self.is_map_available():
            return (0, 0)
        
        map_x = int((world_x - self.map_msg.info.origin.position.x) / self.map_msg.info.resolution)
        map_y = int((world_y - self.map_msg.info.origin.position.y) / self.map_msg.info.resolution)
        
        return (map_x, map_y)

    def map_to_world(self, map_x: int, map_y: int) -> tuple:
        """
        Konverter map koordinater til world koordinater
        
        Args:
            map_x, map_y: Map koordinater
            
        Returns:
            tuple: (world_x, world_y)
        """
        if not self.is_map_available():
            return (0.0, 0.0)
        
        world_x = self.map_msg.info.origin.position.x + map_y * self.map_msg.info.resolution
        world_y = self.map_msg.info.origin.position.y + map_x * self.map_msg.info.resolution
        
        return (world_x, world_y)
