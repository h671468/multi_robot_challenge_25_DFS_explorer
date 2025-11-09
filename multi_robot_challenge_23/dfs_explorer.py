#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math


class DfsExplorer:
    """Enkel DFS-basert utforsker som bruker en LIFO-stakk av delmål."""

    OFFSET = 1.0  # meter fra krysset før vi går inn i sidegang

    def __init__(self, prefer_left: bool = False):
        self.stack = []  # LIFO med (x, y)
        self.visited_targets = set()
        self.current_goal = None
        self.pose = None  # (position tuple, yaw)
        self.prefer_left = prefer_left

    def reset(self):
        """Nullstill all DFS-status."""
        self.stack.clear()
        self.visited_targets.clear()
        self.current_goal = None
        self.pose = None

    def clear_current_goal(self):
        """Fjern aktivt mål uten å tømme stacken."""
        self.current_goal = None

    def update_pose(self, position: tuple, yaw: float) -> None:
        """Oppdater robotens posisjon og retning (yaw)."""
        self.pose = (position, yaw)

    def register_openings(self, openings: list) -> None:
        """
        Registrer sideåpninger i et kryss.

        Vi registrerer kun venstre/høyre åpninger slik at DFS følger korridoren
        videre og lagrer sidegrener for senere besøk.
        """
        if self.pose is None:
            return

        position, yaw = self.pose
        side_openings = [d for d in openings if d in ('LEFT', 'RIGHT')]
        if self.prefer_left:
            side_openings.sort(key=lambda d: 0 if d == 'LEFT' else 1)
        else:
            side_openings.sort(key=lambda d: 0 if d == 'RIGHT' else 1)
        if not side_openings:
            return

        direction_vectors = {
            'LEFT': (math.cos(yaw + math.pi / 2.0), math.sin(yaw + math.pi / 2.0)),
            'RIGHT': (math.cos(yaw - math.pi / 2.0), math.sin(yaw - math.pi / 2.0)),
        }

        for tag in side_openings:
            vec = direction_vectors[tag]
            target = (position[0] + vec[0] * self.OFFSET,
                      position[1] + vec[1] * self.OFFSET)
            key = self._round_tuple(target)
            if key not in self.visited_targets:
                self.stack.append(target)
                self.visited_targets.add(key)

    def has_active_goal(self) -> bool:
        return self.current_goal is not None

    def has_pending_goals(self) -> bool:
        return bool(self.stack)

    def pending_goal_count(self) -> int:
        return len(self.stack)

    def should_request_goal(self, openings: list) -> bool:
        """Når korridoren stopper (ingen åpninger) henter vi neste delmål."""
        return len(openings) == 0 and self.has_pending_goals()

    def next_goal(self):
        """Hent neste delmål fra DFS-stakken."""
        if not self.stack:
            self.current_goal = None
            return None
        self.current_goal = self.stack.pop()
        return self.current_goal

    def goal_reached(self):
        """Kall når Bug2 har nådd et DFS-delmål."""
        self.current_goal = None

    @staticmethod
    def _round_tuple(target: tuple) -> tuple:
        return (round(target[0], 1), round(target[1], 1))

