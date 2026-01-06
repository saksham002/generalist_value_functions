import collections
import dataclasses
import heapq

import numpy as np

# LARGE_MAZE layout from gymnasium_robotics (9x12)
# 1: Wall, 0: Empty
LARGE_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


@dataclasses.dataclass
class PointMazeOracle:
    """Ground truth Q-value oracle using Visibility Graph for precise geodesic distances."""

    # Physics parameters
    mass: float = 4.18
    damping: float = 1.0
    gear: float = 100.0
    dt: float = 0.01
    discount: float = 0.99

    # Maze parameters
    maze_map: list[list[int]] = dataclasses.field(default_factory=lambda: LARGE_MAZE)
    scaling: float = 1.0

    # Visibility Graph Cache
    _static_nodes: list[np.ndarray] = dataclasses.field(init=False, default_factory=list)
    _static_edges: dict[int, list[tuple[int, float]]] = dataclasses.field(init=False, default_factory=dict)

    def __post_init__(self):
        self.height = len(self.maze_map)
        self.width = len(self.maze_map[0])
        self.center_x = self.width / 2 * self.scaling
        self.center_y = self.height / 2 * self.scaling

        # Build Static Visibility Graph
        self._build_static_graph()

    def _xy_to_rowcol(self, xy: np.ndarray) -> np.ndarray:
        x, y = xy[0], xy[1]
        c = int(np.floor((x + self.center_x) / self.scaling))
        r = int(np.floor((self.center_y - y) / self.scaling))
        return np.array([r, c])

    def _rowcol_to_xy_center(self, rowcol: np.ndarray) -> np.ndarray:
        r, c = rowcol[0], rowcol[1]
        x = (c + 0.5) * self.scaling - self.center_x
        y = self.center_y - (r + 0.5) * self.scaling
        return np.array([x, y])

    def _is_blocked(self, r: int, c: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width):
            return True
        return self.maze_map[r][c] == 1

    def _line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Exact line of sight check against grid walls."""
        x0, y0 = p1
        x1, y1 = p2
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return True

        # Step size smaller than cell size to query blocking
        step_size = 0.1 * self.scaling
        steps = int(np.ceil(dist / step_size))

        # Robust check:
        # Instead of endpoints (which might be exactly on corners), verify segment interior.
        # Nudging corners in _build_static_graph helps, but we also skip endpoints here.

        for i in range(1, steps):  # Skip 0 and steps (endpoints)
            t = i / steps
            px = x0 + t * dx
            py = y0 + t * dy

            rc = self._xy_to_rowcol(np.array([px, py]))
            if self._is_blocked(rc[0], rc[1]):
                return False
        return True

    def _build_static_graph(self):
        """Identify convex corners of free space and build visibility graph."""
        self._static_nodes = []

        # Nudge factor to push corner nodes slightly into free space to avoid grazing failures
        epsilon = 1e-3

        # Scan internal vertices
        for r in range(1, self.height):
            for c in range(1, self.width):
                cells = [
                    self.maze_map[r - 1][c - 1],
                    self.maze_map[r - 1][c],
                    self.maze_map[r][c - 1],
                    self.maze_map[r][c],
                ]

                n_walls = sum(cells)
                is_node = False
                nudge_x, nudge_y = 0.0, 0.0

                # Determine "outer corner" type and nudge direction
                # Vertex (r, c) is TL of cell(r,c).
                # Nudge should be towards the diagonal FREE space.

                if n_walls == 1:
                    is_node = True
                    # If walls are:
                    # [0, 0]
                    # [0, 1] (BR is wall) -> Nudge Top-Left (-x, +y relative to vertex)
                    if cells[3] == 1:  # BR is Wall
                        nudge_x, nudge_y = -1, 1
                    elif cells[2] == 1:  # BL is Wall
                        nudge_x, nudge_y = 1, 1
                    elif cells[1] == 1:  # TR is Wall
                        nudge_x, nudge_y = -1, -1
                    elif cells[0] == 1:  # TL is Wall
                        nudge_x, nudge_y = 1, -1

                elif n_walls == 2:
                    # Diagonal case
                    if cells[0] == cells[3] and cells[0] != cells[1]:
                        # Checkerboard. Pivot point.
                        is_node = True
                        # No clear nudge direction suitable for ALL paths.
                        # Usually agent can pass through vertex? No, diagonal blocked?
                        # If checkerboard [1,0] / [0,1], path is blocked unless squeezing.
                        # Assuming robust "no squeeze", we might skip these nodes?
                        # Large Maze usually doesn't have checkerboards accessible.

                if is_node:
                    # Vertex cartesian position
                    vx = -self.center_x + c * self.scaling
                    vy = self.center_y - r * self.scaling

                    # Apply nudge
                    vx += nudge_x * epsilon
                    vy += nudge_y * epsilon

                    self._static_nodes.append(np.array([vx, vy]))

        # Build Edges
        self._static_edges = collections.defaultdict(list)
        n = len(self._static_nodes)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self._static_nodes[i]
                p2 = self._static_nodes[j]
                if self._line_of_sight(p1, p2):
                    dist = float(np.linalg.norm(p1 - p2))
                    self._static_edges[i].append((j, dist))
                    self._static_edges[j].append((i, dist))

    def compute_geodesic_distance(self, start: np.ndarray, goal: np.ndarray) -> float:
        """Compute Euclidean shortest path distance using Visibility Graph."""
        if self._line_of_sight(start, goal):
            return float(np.linalg.norm(start - goal))

        # Add Start and Goal to graph dynamically
        nodes = [*self._static_nodes, start, goal]
        start_idx = len(nodes) - 2
        goal_idx = len(nodes) - 1

        adj = collections.defaultdict(list)

        # 1. Static connections
        for u, neighbors in self._static_edges.items():
            for v, dist in neighbors:
                adj[u].append((v, dist))

        # 2. Dynamic connections
        for i in [start_idx, goal_idx]:
            p = nodes[i]
            # Try connecting to all static nodes
            for j in range(len(self._static_nodes)):
                target = nodes[j]
                if self._line_of_sight(p, target):
                    dist = float(np.linalg.norm(p - target))
                    adj[i].append((j, dist))
                    adj[j].append((i, dist))

        # Dijkstra (Dynamic) - only need to search if start/goal connected
        pq = [(0.0, start_idx)]
        dists = {start_idx: 0.0}

        while pq:
            d, u = heapq.heappop(pq)

            if d > dists.get(u, float("inf")):
                continue

            if u == goal_idx:
                return d

            for v, weight in adj[u]:
                if dists.get(v, float("inf")) > d + weight:
                    dists[v] = d + weight
                    heapq.heappush(pq, (dists[v], v))

        return float("inf")

    def compute_dense_distance(self, state: np.ndarray, goal: np.ndarray, action: np.ndarray) -> float:
        """Q(s, a) = -1 + gamma * V(s')."""
        x, y, vx, vy = state
        ax, ay = action

        # Physics
        fx = self.gear * np.clip(ax, -1, 1) - self.damping * np.clip(vx, -5, 5)
        fy = self.gear * np.clip(ay, -1, 1) - self.damping * np.clip(vy, -5, 5)

        vx_next = np.clip(vx + (fx / self.mass) * self.dt, -5, 5)
        vy_next = np.clip(vy + (fy / self.mass) * self.dt, -5, 5)

        nx = x + vx_next * self.dt
        ny = y + vy_next * self.dt
        next_pos = np.array([nx, ny])

        # Bound check
        rc = self._xy_to_rowcol(next_pos)
        if self._is_blocked(rc[0], rc[1]):
            return -100.0

        # Check if already at goal (within radius 0.45)
        # We explicitly check this to match the environment termination condition.
        if np.linalg.norm(next_pos - goal) <= 0.45:
            return 0.0

        try:
            geo_dist = self.compute_geodesic_distance(next_pos, goal)
        except Exception:
            return -100.0

        if geo_dist == float("inf"):
            return -100.0

        # Steps = T / dt
        # Subtract goal radius (0.45) from distance, plus a tolerance (0.05) to cover boundary edge cases
        # where agent stops slightly 'outside' due to discrete steps.
        effective_dist = max(0.0, geo_dist - 0.45 - 0.05)

        if effective_dist == 0.0:
            return 0.0

        # Use V_max = 6.0 (slightly > 5.0) to account for observed empirical speeds > 5.0 in dataset
        # This ensures the Oracle step count is an optimistic lower bound, producing an upper bound for Value.
        steps = (effective_dist / 6.0) / self.dt

        # Geometric Series
        v_next = -steps if self.discount >= 0.99999 else -(1.0 - self.discount**steps) / (1.0 - self.discount)

        return -1.0 + self.discount * v_next
