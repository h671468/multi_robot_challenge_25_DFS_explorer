#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from scoring_interfaces.srv import SetMarkerPosition
from geometry_msgs.msg import Point
from rclpy.executors import Executor
import numpy as np


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
        self.world_file = self.node.declare_parameter('world_file', '').value
        self.KNOWN_MARKERS = self._load_marker_positions(self.world_file)
        self.calibration_samples = {}
        self.transform_ready = False
        self.rotation = np.eye(2)
        self.translation = np.zeros(2)
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
        marker_key = str(marker_id)
        if marker_id in self.accepted_marker_ids:
            self.node.get_logger().debug(f'📊 Marker {marker_id} allerede akseptert tidligere')
            return True
        if marker_key in self.reported_markers:
            self.node.get_logger().debug(f'📊 Marker {marker_id} allerede forsøkt')
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
        adjusted_position = self._map_to_world(position, marker_id)
        x_pos = float(adjusted_position[0])
        y_pos = float(adjusted_position[1])

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
        return marker_id in self.accepted_marker_ids

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

    def _load_marker_positions(self, world_file: str) -> dict:
        markers = {}
        if not world_file:
            self.node.get_logger().warn('📊 world_file-parameter ikke satt. Rapporterer rå posisjoner.')
            return markers

        if not os.path.isfile(world_file):
            self.node.get_logger().warn(f'📊 Fant ikke world-fil: {world_file}. Rapporterer rå posisjoner.')
            return markers

        try:
            tree = ET.parse(world_file)
            root = tree.getroot()
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'

            for model in root.findall(f'.//{namespace}model'):
                name = model.get('name', '')
                if not name.lower().startswith('marker'):
                    continue
                try:
                    marker_id = int(''.join(filter(str.isdigit, name)))
                except ValueError:
                    continue
                pose_elem = model.find(f'{namespace}pose')
                if pose_elem is None or not pose_elem.text:
                    continue
                try:
                    pose_vals = [float(val) for val in pose_elem.text.strip().split()]
                except ValueError:
                    continue
                markers[marker_id] = (pose_vals[0], pose_vals[1])

            if not markers:
                self.node.get_logger().warn(f'📊 Fant ingen Marker-modeller i {world_file}.')
        except Exception as exc:
            self.node.get_logger().warn(f'📊 Klarte ikke lese markerposisjoner fra {world_file}: {exc}')

        return markers

    def _update_calibration(self, marker_id: int, map_position: tuple):
        if marker_id not in self.KNOWN_MARKERS:
            return

        self.calibration_samples[marker_id] = np.array(map_position[:2], dtype=float)
        if len(self.calibration_samples) < 2:
            return

        map_points = np.stack(list(self.calibration_samples.values()))
        world_points = np.stack([np.array(self.KNOWN_MARKERS[mid], dtype=float) for mid in self.calibration_samples.keys()])

        map_centroid = map_points.mean(axis=0)
        world_centroid = world_points.mean(axis=0)

        H = (map_points - map_centroid).T @ (world_points - world_centroid)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = world_centroid - R @ map_centroid

        self.rotation = R
        self.translation = t
        self.transform_ready = True

    def _map_to_world(self, map_position: tuple, marker_id: int) -> tuple:
        map_xy = np.array(map_position[:2], dtype=float)
        self._update_calibration(marker_id, map_xy)

        if self.transform_ready:
            world_xy = self.rotation @ map_xy + self.translation
            return (float(world_xy[0]), float(world_xy[1]))

        if marker_id in self.KNOWN_MARKERS:
            # Bruk kjent posisjon inntil kalibrering er klar
            return self.KNOWN_MARKERS[marker_id]

        # Ingen kalibrering tilgjengelig. Returner rå kartposisjon.
        return (float(map_xy[0]), float(map_xy[1]))

