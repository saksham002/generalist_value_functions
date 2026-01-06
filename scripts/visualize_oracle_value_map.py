import os

import matplotlib.pyplot as plt
import numpy as np

from openpi.point_maze_utils.point_maze_oracle import PointMazeOracle


def main():
    # Setup Oracle
    gamma = 0.99
    oracle = PointMazeOracle(discount=gamma)

    # Define Goal: Center (0,0) is known free
    goal = np.array([0.0, 0.0])

    # Create Grid
    resolution = 100
    xs = np.linspace(-5, 5, resolution)
    ys = np.linspace(-5, 5, resolution)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)

    print(f"Computing Value Map for Goal {goal} with gamma={gamma}...")

    action = np.zeros(2)  # Zero action for Q(s, 0)

    for i in range(resolution):
        if i % 10 == 0:
            print(f"Row {i}/{resolution}")
        for j in range(resolution):
            x, y = xx[i, j], yy[i, j]
            # State: x, y, vx, vy
            state = np.array([x, y, 0.0, 0.0])

            # Compute Q-value using Oracle
            # This handles pathfinding, discounting, and unreachable points (-100.0)
            q = oracle.compute_dense_distance(state, goal, action)

            zz[i, j] = q

    # Plotting
    plt.figure(figsize=(10, 8))

    # Countour plot
    # Q-values range from approx -100 (far/unreachable) to 0 (goal)
    cnt = plt.contourf(xx, yy, zz, levels=100, cmap="viridis")
    plt.colorbar(cnt, label="Q(s, 0)")

    # Plot Goal
    plt.scatter([goal[0]], [goal[1]], c="red", marker="*", s=200, label="Goal", edgecolor="black")

    plt.title(f"Oracle Q-Value Map (Gamma={gamma})\nGoal at {goal}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()

    output_path = "oracle_value_map_gamma99.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
