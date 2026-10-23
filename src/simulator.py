
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def simulate_trajectory(
    speed=50.0,
    angle_deg=60.0,
    mass=1.0,
    area=0.01,
    drag_coefficient=0.5,
    air_density=1.225,
    show_plot=True,
):
    """Simulate idealized 2D motion under gravity and quadratic drag."""

    if speed <= 0:
        raise ValueError("Speed must be positive.")
    if mass <= 0 or area < 0 or drag_coefficient < 0 or air_density < 0:
        raise ValueError("Mass must be positive; drag parameters cannot be negative.")
    if not 0 < angle_deg < 90:
        raise ValueError("Angle must be between 0 and 90 degrees.")

    gravity = 9.81
    angle = np.radians(angle_deg)

    vx0 = speed * np.cos(angle)
    vy0 = speed * np.sin(angle)

    # State: x, y, vx, vy
    initial_state = [0.0, 0.0, vx0, vy0]

    def equations(t, state):
        x, y, vx, vy = state
        velocity = np.hypot(vx, vy)

        drag_factor = (
            0.5 * air_density * drag_coefficient
            * area * velocity / mass
        )

        ax = -drag_factor * vx
        ay = -gravity - drag_factor * vy

        return [vx, vy, ax, ay]

    def ground_event(t, state):
        return state[1]

    ground_event.terminal = True
    ground_event.direction = -1

    result = solve_ivp(
        equations,
        (0.0, 30.0),
        initial_state,
        events=ground_event,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.02,
    )

    if not result.success:
        raise RuntimeError(result.message)

    x, y, vx, vy = result.y

    print(f"Flight time: {result.t[-1]:.2f} seconds")
    print(f"Horizontal distance: {x[-1]:.2f} metres")
    print(f"Maximum height: {np.max(y):.2f} metres")

    
    if show_plot:
        plt.figure(figsize=(9, 5))
        plt.plot(x, y, label="Simulated trajectory")
        plt.xlabel("Horizontal distance (m)")
        plt.ylabel("Height (m)")
        plt.title("ASTRAL — 2D Trajectory Simulator")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result


if __name__ == "__main__":
    simulate_trajectory()