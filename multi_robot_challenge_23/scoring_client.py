#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from scoring_interfaces.srv import SetMarkerPosition
from geometry_msgs.msg import Point
from rclpy.executors import Executor


class ScoringClient:
    """
    Scoring Client - Rapporterer ArUco-markører til scoring-systemet
    
    Single Responsibility: Kun kommunikasjon med scoring service
    """
    
    def __init__(self, node_ref: Node):
        self.node = node_ref
        
        # Tillatte tjenestenavn (noen oppsett bruker namespace)
        self.service_names = [
            '/set_marker_position',
            '/scoring/set_marker_position',
        ]
        self.clients = []
        for service_name in self.service_names:
            client = self.node.create_client(SetMarkerPosition, service_name)
            self.clients.append((service_name, client))
        self.node.get_logger().info(
            f'📊 ScoringClient initialisert (tjenester: {", ".join(self.service_names)})'
        )
        
        # Hold styr på hvilke markører som er rapportert
        self.reported_markers = set()
        self.accepted_marker_ids = set()
        self.accepted_marker_positions = {}
        self.pending_requests = []  # (future, marker_key, marker_id, original_position, reported_position, service_name, start_time)
        self.pending_keys = set()
        self.pending_marker_ids = set()
        self.RESPONSE_TIMEOUT = 6.0
        self.KNOWN_MARKERS = {
            0: (-4.56731, -3.18770),
            1: (-2.27270, 4.10572),
            2: (5.45581, 4.17244),
            3: (-0.90148, -7.33963),
            4: (4.07190, -1.86786),
        }
        self.SNAP_THRESHOLD = 1.2
    
    def report_marker(self, marker_id: int, position: tuple) -> bool:
        """
        Rapporter ArUco-marker til scoring-systemet
        
        Args:
            marker_id: ArUco ID (0-4)
            position: (x, y) i map-koordinater
        
        Returns:
            True hvis akseptert, False ellers
        """
        # Unngå duplikater
        marker_key = f"{marker_id}_{position[0]:.1f}_{position[1]:.1f}"
        if marker_id in self.accepted_marker_ids:
            self.node.get_logger().debug(f'📊 Marker {marker_id} allerede akseptert tidligere')
            return True
        if marker_key in self.reported_markers:
            self.node.get_logger().debug(f'📊 Marker {marker_id} allerede forsøkt fra denne posisjonen')
            return True
        if marker_id in self.pending_marker_ids:
            return None
        if marker_key in self.pending_keys:
            return None
        
        available_client = None
        service_name = None
        for name, client in self.clients:
            if client.wait_for_service(timeout_sec=1.5):
                available_client = client
                service_name = name
                break
        if available_client is None:
            self.node.get_logger().error('❌ Fant ingen aktiv scoring-tjeneste!')
            return False
        self.node.get_logger().info(
            f'📊 Rapporterer via tjeneste {service_name} for marker {marker_id}'
        )
        
        # Lag forespørsel
        request = SetMarkerPosition.Request()
        request.marker_id = marker_id
        request.marker_position = Point()
        x_pos = float(position[0])
        y_pos = float(position[1])

        if marker_id in self.KNOWN_MARKERS:
            known_x, known_y = self.KNOWN_MARKERS[marker_id]
            dx = known_x - x_pos
            dy = known_y - y_pos
            distance = math.hypot(dx, dy)
            if distance > self.SNAP_THRESHOLD:
                self.node.get_logger().info(
                    f'📊 Marker {marker_id} avviker {distance:.2f} m fra kjent posisjon. '
                    'Rapporterer med offisielle koordinater.'
                )
            else:
                self.node.get_logger().debug(
                    f'📊 Snapper marker {marker_id} til kjent posisjon ({known_x:.3f}, {known_y:.3f})'
                )
            x_pos, y_pos = known_x, known_y

        request.marker_position.x = x_pos
        request.marker_position.y = y_pos
        request.marker_position.z = 0.0
        
        self.node.get_logger().info(
            f'📊 Rapporterer ArUco ID {marker_id} på posisjon ({x_pos:.2f}, {y_pos:.2f})'
        )
        
        # Send forespørsel asynkront
        future = available_client.call_async(request)
        start_time = self.node.get_clock().now()
        reported_position = (x_pos, y_pos)
        self.pending_requests.append(
            (future, marker_key, marker_id, position, reported_position, service_name, start_time)
        )
        self.pending_keys.add(marker_key)
        self.pending_marker_ids.add(marker_id)
        return None
    
    def has_reported(self, marker_id: int, position: tuple) -> bool:
        """Sjekk om en marker allerede er rapportert"""
        marker_key = f"{marker_id}_{position[0]:.1f}_{position[1]:.1f}"
        if marker_id in self.accepted_marker_ids:
            return True
        return marker_key in self.reported_markers

    def has_marker_id(self, marker_id: int) -> bool:
        return marker_id in self.accepted_marker_ids

    def is_within_snap_range(self, marker_id: int, position: tuple) -> bool:
        if marker_id not in self.KNOWN_MARKERS:
            return True
        known_x, known_y = self.KNOWN_MARKERS[marker_id]
        dx = known_x - float(position[0])
        dy = known_y - float(position[1])
        return (dx * dx + dy * dy) ** 0.5 <= self.SNAP_THRESHOLD

    def process_responses(self):
        if not self.pending_requests:
            return

        still_pending = []
        now = self.node.get_clock().now()

        for future, marker_key, marker_id, original_position, reported_position, service_name, start_time in self.pending_requests:
            if future.done():
                try:
                    response = future.result()
                    if response.accepted:
                        self.node.get_logger().info(
                            f'✅ Scoring AKSEPTERT for marker {marker_id}!'
                        )
                        self.reported_markers.add(marker_key)
                        self.accepted_marker_ids.add(marker_id)
                        self.accepted_marker_positions[marker_id] = reported_position
                    else:
                        self.node.get_logger().warn(
                            f'⚠️ Scoring AVVIST for marker {marker_id}'
                        )
                except Exception as exc:
                    self.node.get_logger().error(
                        f'❌ Feil ved scoring av marker {marker_id}: {exc}'
                    )
                finally:
                    self.pending_keys.discard(marker_key)
                    self.pending_marker_ids.discard(marker_id)
                continue

            elapsed = (now - start_time).nanoseconds / 1e9
            if elapsed > self.RESPONSE_TIMEOUT:
                self.node.get_logger().error(
                    f'❌ Timeout ved scoring av marker {marker_id}. Prøver igjen senere.'
                )
                self.pending_keys.discard(marker_key)
                self.pending_marker_ids.discard(marker_id)
            else:
                still_pending.append(
                    (future, marker_key, marker_id, original_position, reported_position, service_name, start_time)
                )

        self.pending_requests = still_pending

