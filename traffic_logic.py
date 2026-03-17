from __future__ import annotations

from dataclasses import dataclass
import time

import config
from utils import clamp


_DIRECTIONS = ("north", "south", "east", "west")


def calculate_density(region_counts: dict[str, int]) -> dict[str, int]:
    """
    Calculate traffic density per region.

    For now, density == number of vehicles (per region). This function is a
    dedicated hook so future versions can incorporate speed, occupancy, flow,
    time windows, or normalization by region area.
    """
    out = {"north": 0, "south": 0, "east": 0, "west": 0}
    if not region_counts:
        return out

    for key in out.keys():
        try:
            out[key] = max(0, int(region_counts.get(key, 0)))
        except Exception:
            out[key] = 0
    return out


def calculate_green_time(vehicle_count: int) -> int:
    """
    Calculate green time from a vehicle count.

    Formula:
        green_time = BASE_GREEN_TIME + (vehicle_count * TIME_PER_VEHICLE)

    Constraints:
        min = BASE_GREEN_TIME
        max = MAX_GREEN_TIME
    """
    vehicle_count = max(0, int(vehicle_count))
    proposed = config.BASE_GREEN_TIME + (vehicle_count * config.TIME_PER_VEHICLE)
    return clamp(int(proposed), config.BASE_GREEN_TIME, config.MAX_GREEN_TIME)


@dataclass(frozen=True)
class SignalTiming:
    green_seconds: int
    yellow_seconds: int


@dataclass
class SignalState:
    color: str  # "GREEN" | "YELLOW" | "RED"
    green_seconds: int = 0


class TrafficController:
    """
    Manages 4 traffic signals (north/south/east/west).

    - Only one direction is GREEN at a time
    - Chooses next GREEN based on highest density
    - Computes green time from selected direction's density
    """

    def __init__(
        self,
        *,
        yellow_time: int = config.YELLOW_TIME,
        initial_green: str = "north",
        max_wait_seconds: int = 60,
    ) -> None:
        self._yellow_time = int(yellow_time)
        self._max_wait_seconds = int(max_wait_seconds)
        self._current_green = initial_green if initial_green in _DIRECTIONS else "north"
        self._signals: dict[str, SignalState] = {
            d: SignalState(color="RED", green_seconds=0) for d in _DIRECTIONS
        }
        self._signals[self._current_green].color = "GREEN"

        self._last_density: dict[str, int] = {d: 0 for d in _DIRECTIONS}
        self._current_green_time: int = calculate_green_time(0)
        self._phase: str = "GREEN"  # "GREEN" | "YELLOW"
        self._pending_green: str | None = None
        self._yellow_started_at: float | None = None
        now = time.monotonic()
        self._last_served_at: dict[str, float] = {d: now for d in _DIRECTIONS}
        self._last_served_at[self._current_green] = now

    @property
    def signals(self) -> dict[str, SignalState]:
        return self._signals

    @property
    def current_green(self) -> str:
        return self._current_green

    def get_wait_times(self) -> dict[str, int]:
        """
        Current waiting time per direction in whole seconds.
        """
        now = time.monotonic()
        return {d: int(max(0.0, now - self._last_served_at.get(d, now))) for d in _DIRECTIONS}

    def compute_timing(self, vehicle_count: int) -> SignalTiming:
        """
        Backwards-compatible timing helper.
        """
        green = calculate_green_time(vehicle_count)
        return SignalTiming(green_seconds=green, yellow_seconds=self._yellow_time)

    def update(self, density_data: dict[str, int]) -> None:
        """
        Update controller state given current density data.

        Smooth transition behavior:
            GREEN -> (if change needed) YELLOW for YELLOW_TIME -> RED -> next GREEN

        Call `update()` repeatedly (e.g., once per frame) so the controller can
        complete the YELLOW phase when the timer elapses.
        """
        self._last_density = calculate_density(density_data)

        # If we're in a YELLOW transition, finalize switch when timer expires.
        if self._phase == "YELLOW":
            if self._yellow_started_at is None or self._pending_green is None:
                self._phase = "GREEN"
            else:
                elapsed = time.monotonic() - self._yellow_started_at
                if elapsed >= self._yellow_time:
                    self._finalize_switch(self._pending_green)
            return

        next_dir = self.get_next_signal()
        if next_dir != self._current_green:
            self.switch_signal(next_dir)
        else:
            self._assign_green_time(next_dir)

    def get_next_signal(self) -> str:
        """
        Select direction with the highest density.
        Ties are broken by keeping the current green.
        """
        # Fairness: if any lane has waited too long, force it next.
        now = time.monotonic()
        wait_times = {d: now - self._last_served_at.get(d, now) for d in _DIRECTIONS}
        overdue = [d for d, w in wait_times.items() if w >= self._max_wait_seconds]
        if overdue:
            # Pick the most-starved lane; tie-breaker: keep current green if it's among them.
            overdue.sort(key=lambda d: wait_times[d], reverse=True)
            if self._current_green in overdue:
                return self._current_green
            return overdue[0]

        densities = self._last_density or {d: 0 for d in _DIRECTIONS}
        best = self._current_green
        best_val = densities.get(best, 0)
        for d in _DIRECTIONS:
            val = densities.get(d, 0)
            if val > best_val:
                best, best_val = d, val
        return best

    def switch_signal(self, direction: str | None = None) -> str:
        """
        Begin switching the GREEN signal to `direction` (or to the computed next signal).

        Transition is smooth:
            current GREEN -> YELLOW for YELLOW_TIME -> RED -> next GREEN

        Returns the direction that is currently GREEN (may be unchanged while in YELLOW).
        """
        new_dir = direction or self.get_next_signal()
        if new_dir not in _DIRECTIONS:
            new_dir = self.get_next_signal()

        if self._phase == "YELLOW":
            # Already transitioning; allow updating the pending target.
            self._pending_green = new_dir
            return self._current_green

        if new_dir == self._current_green:
            self._assign_green_time(new_dir)
            return self._current_green

        # Start YELLOW on current direction, schedule pending direction.
        self._begin_yellow(new_dir)
        return self._current_green

    def force_green(self, direction: str) -> str:
        """
        Emergency override: immediately force a direction to GREEN.

        This bypasses the normal GREEN->YELLOW->RED transition to prioritize safety.
        """
        if direction not in _DIRECTIONS:
            return self._current_green

        for d in _DIRECTIONS:
            self._signals[d].color = "RED"
            self._signals[d].green_seconds = 0

        self._current_green = direction
        self._signals[direction].color = "GREEN"

        # Reset transition state.
        self._phase = "GREEN"
        self._pending_green = None
        self._yellow_started_at = None

        self._last_served_at[direction] = time.monotonic()
        self._assign_green_time(direction)
        return self._current_green

    def _assign_green_time(self, direction: str) -> None:
        density = max(0, int(self._last_density.get(direction, 0)))
        self._current_green_time = calculate_green_time(density)
        self._signals[direction].green_seconds = self._current_green_time

    def _begin_yellow(self, pending_direction: str) -> None:
        for d in _DIRECTIONS:
            self._signals[d].color = "RED"
            self._signals[d].green_seconds = 0

        self._signals[self._current_green].color = "YELLOW"
        self._phase = "YELLOW"
        self._pending_green = pending_direction
        self._yellow_started_at = time.monotonic()

    def _finalize_switch(self, new_green: str) -> None:
        for d in _DIRECTIONS:
            self._signals[d].color = "RED"
            self._signals[d].green_seconds = 0

        self._current_green = new_green
        self._signals[new_green].color = "GREEN"
        self._phase = "GREEN"
        self._pending_green = None
        self._yellow_started_at = None
        self._last_served_at[new_green] = time.monotonic()
        self._assign_green_time(new_green)

