"""CabledDrone: MuJoCo simulation of a quadrotor (Skydio X2) linked by cables.

The MJCF model (x2.xml + scene.xml) comes from mujoco_menagerie/skydio_x2.
The script runs a real-time simulation in the MuJoCo viewer, driven by the
control law selected with `--controller`.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(BASE_DIR, "scene.xml")

logger = logging.getLogger(__name__)


@dataclass
class PayloadParams:
    """Load hanging under the drone: a chain of rigid rods ending in a sphere.

    Each rod is linked to the previous one (the first one to the drone) by a
    ball joint (3 free rotations, no damping). With one rod, it is a
    spherical pendulum; with two, a double spherical pendulum, whose motion
    is chaotic. The first ball joint is at the center of the drone's
    underside, ~5 cm below its center of mass.
    """

    rod_count: int = 1
    rod_length: float = 0.5  # m, length of each rod
    rod_radius: float = 0.005  # m
    rod_mass: float = 0.02  # kg, mass of each rod
    sphere_radius: float = 0.05  # m
    sphere_mass: float = 0.3  # kg
    ground_clearance: float = 0.1  # m, height of the sphere above the ground at start
    # Simulation timestep forced when a payload is present (x2.xml: 0.01 s). A thin rod has almost no
    # inertia around its own axis (~2.5e-7 kg·m²): with two rods, the intermediate rod, with no sphere
    # to weigh it down, made the simulation blow up at 0.01 s (NaN, drone flipped over from 0.8 kg).
    timestep: float = 0.002  # s

    @property
    def total_length(self) -> float:
        """Vertical distance from the drone's underside to the bottom of the sphere, chain at rest."""
        return self.rod_count * self.rod_length + self.sphere_radius


def add_payload(spec: mujoco.MjSpec, payload: PayloadParams) -> None:
    """Add the chain of rods and the sphere to the "x2" drone, and adapt the "hover" keyframe."""
    parent, attach_pos = spec.body("x2"), [0, 0, 0]
    for i in range(1, payload.rod_count + 1):
        # each rod hangs below its parent, with the ball joint at its upper end
        link = parent.add_body(name=f"payload_link{i}", pos=attach_pos)
        link.add_joint(name=f"payload_ball{i}", type=mujoco.mjtJoint.mjJNT_BALL)
        rod = link.add_geom(
            name=f"payload_rod{i}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[payload.rod_radius, 0, 0],
            mass=payload.rod_mass,
            rgba=[0.6, 0.6, 0.6, 1],
        )
        rod.fromto = [0, 0, 0, 0, 0, -payload.rod_length]
        parent, attach_pos = link, [0, 0, -payload.rod_length]

    parent.add_geom(
        name="payload_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[payload.sphere_radius, 0, 0],
        pos=attach_pos,
        mass=payload.sphere_mass,
        rgba=[0.9, 0.6, 0.1, 1],
    )

    spec.option.timestep = min(spec.option.timestep, payload.timestep)

    # keyframe: add the orientation of each ball joint (identity quaternion = vertical rod)
    # and raise the drone so the sphere doesn't touch the ground at start
    hover = spec.key("hover")
    qpos = list(hover.qpos)
    qpos[2] = max(qpos[2], payload.total_length + payload.ground_clearance)
    hover.qpos = qpos + [1, 0, 0, 0] * payload.rod_count


@dataclass
class RopeParams:
    """Flexible rope attached under the drone, with its mass spread evenly along its length.

    Modeled by a 1D `flexcomp`: a chain of point masses (with no
    orientation) linked by segments of fixed length. The rope therefore has
    no bending or twisting stiffness at all, like a real rope.

    It is attached to the drone by a point attachment (`connect`: 3
    translations blocked, no rotation): like a knot, it only transmits
    tension. A ball joint would add the rotation of the rope around its own
    axis, which has no physical meaning for a rope and almost zero inertia
    (the cause of the rigid double pendulum blowing up).

    With `anchored`, the other end is pinned to the ground (where it lies at
    start): the drone then can't move further from the anchor than the rope
    length.
    """

    length: float = 1.0  # m
    anchored: bool = False
    mass: float = 0.2  # kg, spread evenly over the points
    point_count: int = 21  # point masses, i.e. point_count - 1 segments
    radius: float = 0.006  # m, collision and display radius
    attach_offset: float = -0.01  # m, attachment below the drone's origin (avoids constant contact with its hull)
    # Air drag, in N·s/m per meter of rope, applied to each point: without it, the rope keeps
    # swinging forever. With 0.1, a swing dies out in a few seconds.
    drag: float = 0.1
    # Segment length and attachment constraints. MuJoCo constraints are "soft", and their stiffness
    # depends on the masses involved: with the default values (solref 0.02), a 10 g point carrying the
    # weight of the whole rope let the first segment stretch by 8 %. With these values, stretch stays
    # under 0.3 %. solref must stay above 2 * timestep.
    solref: str = "0.004 1"
    solimp: str = "0.99 0.999 0.001"
    timestep: float = 0.002  # s


# Tethered rope: longer by default, to give the drone room around the anchor.
ANCHORED_ROPE_LENGTH = 2.0  # m


def rope_points(rope: RopeParams, top: np.ndarray) -> np.ndarray:
    """Initial rope shape: vertical below the attachment point `top` down to the ground, the rest lying on it.

    The rest length of each segment is taken from the initial shape, so all
    segments must have the same length in it. The segment at the corner goes
    down diagonally to the ground to keep that length.
    """
    segment = rope.length / (rope.point_count - 1)
    vertical_count = min(int((top[2] - rope.radius) // segment), rope.point_count - 1)
    corner = top + [0, 0, -vertical_count * segment]
    corner_drop = corner[2] - rope.radius  # < segment
    corner_run = np.sqrt(segment**2 - corner_drop**2)
    points = []
    for i in range(rope.point_count):
        if i <= vertical_count:
            points.append(top + [0, 0, -i * segment])
        else:
            run = corner_run + (i - vertical_count - 1) * segment
            points.append([corner[0] + run, corner[1], rope.radius])
    return np.array(points)


def rope_mjcf(rope: RopeParams, top: np.ndarray) -> str:
    """MJCF that includes scene.xml and adds the rope to it, attached to the drone.

    A `flexcomp` only exists in the XML format: MjSpec can't create one, and
    grafting it from another MjSpec (`attach`) loses its constraints. This
    MJCF is therefore generated as text.
    """
    points = rope_points(rope, top)
    point = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    element = " ".join(f"{i} {i + 1}" for i in range(rope.point_count - 1))
    pin_xml = anchor_xml = ""
    if rope.anchored:
        # last point pinned: flexcomp creates no body for it, it is fixed to the world.
        # The dark post (no collision) only shows the anchor in the viewer.
        anchor_x, anchor_y, _ = points[-1]
        pin_xml = f'''
      <pin id="{rope.point_count - 1}"/>'''
        anchor_xml = f'''
    <geom name="rope_anchor" type="cylinder" size=".03 .01" pos="{anchor_x:.6f} {anchor_y:.6f} .01"
          rgba=".2 .2 .2 1" contype="0" conaffinity="0"/>'''
    return f"""<mujoco>
  <!-- absolute paths: without an original file, MuJoCo can't resolve relative directories -->
  <compiler assetdir="{os.path.join(BASE_DIR, "assets")}"/>
  <include file="{SCENE_PATH}"/>
  <worldbody>
    <flexcomp name="rope" type="direct" dim="1" point="{point}" element="{element}"
              mass="{rope.mass}" radius="{rope.radius}" rgba=".85 .75 .55 1">
      <edge equality="true" solref="{rope.solref}" solimp="{rope.solimp}"/>{pin_xml}
    </flexcomp>{anchor_xml}
  </worldbody>
  <equality>
    <connect name="rope_attach" body1="rope_0" body2="x2" anchor="0 0 0" solref="{rope.solref}" solimp="{rope.solimp}"/>
  </equality>
</mujoco>"""


def build_model(payload: PayloadParams | None = None, rope: RopeParams | None = None) -> mujoco.MjModel:
    """Compile scene.xml, with the rigid payload (`payload`) or the rope (`rope`) if given."""
    if rope is None:
        spec = mujoco.MjSpec.from_file(SCENE_PATH)
        if payload is not None:
            add_payload(spec, payload)
        return spec.compile()

    # The `connect` equality links the two bodies in their reference configuration (qpos0), where the
    # drone sits at its body position from x2.xml. The drone is moved there to its start position
    # (keyframe "hover"), so that the rope, built below that start point, is already attached to it.
    scene_spec = mujoco.MjSpec.from_file(SCENE_PATH)  # keep a reference: the vectors read point into its memory
    start = np.array(scene_spec.key("hover").qpos)[:3]
    spec = mujoco.MjSpec.from_string(rope_mjcf(rope, start + [0, 0, rope.attach_offset]))
    spec.body("x2").pos = start
    spec.option.timestep = min(spec.option.timestep, rope.timestep)
    damping = rope.drag * rope.length / rope.point_count
    for joint in spec.joints:  # the 3 slide joints of each rope point (unnamed, created by flexcomp)
        if joint.parent.name.startswith("rope_"):
            joint.damping = [damping, 0, 0]
    return spec.compile()


# A control law returns the thrusts of the 4 rotors (N), in the order thrust1..thrust4.
Controller = Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray]


def hover_control(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Constant thrust from the "hover" keyframe: exactly balances the weight, open loop.

    No state feedback: the slightest disturbance (e.g. a force applied in
    the viewer) makes the drone drift away.
    """
    return model.key("hover").ctrl.copy()


def off_control(model: mujoco.MjModel, _data: mujoco.MjData) -> np.ndarray:
    """Motors off: the drone falls."""
    return np.zeros(model.nu)


@dataclass
class AltitudePIDParams:
    """Setpoint and gains of the altitude PID.

    Gains are expressed as accelerations (m/s² per m of error, etc.): they
    are multiplied by the drone's mass, so they don't depend on it.
    """

    target_altitude: float = 1.0  # m
    kp: float = 9.0
    ki: float = 3.0
    kd: float = 6.0
    integral_limit: float = 2.0  # m·s, bounds the integral (anti-windup)
    max_vertical_speed: float = 0.5  # m/s, speed at which the setpoint moves toward the target


class AltitudePID:
    """Holds the drone at a target altitude, with a PID on the total thrust.

    The total thrust is split equally across the 4 rotors: the PID only
    corrects altitude, not attitude. If the drone is tilted (e.g. a
    rotational disturbance in the viewer), it stays tilted and drifts
    sideways; that is the job of an attitude controller.

    Total thrust = m * (g + kp * e + ki * ∫e + kd * (vz_ref - vz)), with e = z_ref - z:
    - the m * g term (feedforward) balances the weight, so the PID only has
      to correct the error;
    - the derivative term acts on the velocity error (setpoint velocity -
      measured velocity): no kick when the target changes, and it doesn't
      slow down the climb during the ramp;
    - the setpoint z_ref doesn't jump straight to `target_altitude`: it
      moves toward it at `max_vertical_speed`. Without this ramp, a long
      climb fills the integral and the drone clearly overshoots the target
      (~13 % for 0.3 → 1 m).
    """

    def __init__(self, model: mujoco.MjModel, params: AltitudePIDParams) -> None:
        self.params = params
        self.mass = model.body_mass[model.body("x2").id]  # drone alone: any payload is unknown
        self.gravity = -model.opt.gravity[2]
        self.dt = model.opt.timestep
        self.integral = 0.0
        self.reference: float | None = None  # current setpoint, initialized to the starting altitude

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        p = self.params
        # the freejoint is the first joint: qpos[2] is the altitude, qvel[2] the vertical speed (world frame)
        altitude, vertical_speed = data.qpos[2], data.qvel[2]

        if self.reference is None:
            self.reference = altitude
        max_step = p.max_vertical_speed * self.dt
        reference_step = np.clip(p.target_altitude - self.reference, -max_step, max_step)
        self.reference += reference_step
        reference_speed = reference_step / self.dt

        error = self.reference - altitude
        self.integral = np.clip(self.integral + error * self.dt, -p.integral_limit, p.integral_limit)

        acceleration = self.gravity + p.kp * error + p.ki * self.integral + p.kd * (reference_speed - vertical_speed)
        total_thrust = self.mass * acceleration
        return np.full(model.nu, total_thrust / model.nu)


@dataclass
class PositionControllerParams:
    """Gains of the position controller (outer loop) and attitude controller (inner loop).

    As in `AltitudePIDParams`, gains are expressed as accelerations (linear
    or angular): they are multiplied by the mass or the inertia.
    """

    kp_xy: float = 6.0
    ki_xy: float = 0.5
    kd_xy: float = 5.0
    kp_z: float = 9.0
    ki_z: float = 3.0
    kd_z: float = 6.0
    # m·s, per axis (anti-windup). Also bounds the unknown load that can be compensated:
    # ki_z * limit * drone mass = 24 N, i.e. ~2.4 kg (with 2.0, the drone couldn't carry more than ~0.8 kg)
    integral_limit: float = 6.0
    # The integral empties `integral_unwind_gain` times faster than it fills (when the error changes sign).
    # After pulling on a tethered rope toward an unreachable target, it is full; without fast unwinding,
    # the drone was still ~60 cm off the target 8 s after it became reachable again.
    integral_unwind_gain: float = 10.0
    max_speed: float = 1.0  # m/s, speed at which the setpoint moves toward the target
    max_acceleration: float = 1.0  # m/s², acceleration/braking of the setpoint
    max_tilt: float = np.deg2rad(30)  # maximum tilt requested from the drone
    kp_attitude: float = 100.0  # rad/s² per rad of orientation error
    kd_attitude: float = 20.0  # rad/s² per rad/s of angular velocity


class PositionController:
    """Brings the drone to the position and heading (yaw) of the "target" mocap body.

    Cascaded control, as on real drone autopilots:
    1. outer loop (position): the setpoint follows a smooth trajectory
       toward the target (`update_reference`), a PID per axis gives the
       desired acceleration, to which g is added to balance the weight;
    2. this acceleration sets both the total thrust and the desired
       attitude: the drone must tilt its z axis along the acceleration (it
       moves sideways by tilting), then rotate around that axis to reach
       the target heading;
    3. inner loop (attitude): a PD on the orientation error gives the roll,
       pitch and yaw torques (geometric controller of Lee et al., 2010,
       which stays valid at large angles);
    4. mixing: the total thrust and the 3 torques are converted into the 4
       rotor thrusts by inverting the allocation matrix, computed from the
       rotor positions in x2.xml.

    The inner loop must be much faster than the outer loop (here ~10 rad/s
    against ~2.5 rad/s): the outer loop assumes that the drone reaches the
    requested attitude almost instantly.
    """

    def __init__(self, model: mujoco.MjModel, params: PositionControllerParams) -> None:
        self.params = params
        self.body_id = model.body("x2").id
        self.target_mocap_id = model.body("target").mocapid[0]
        self.mass = model.body_mass[self.body_id]  # drone alone: any payload is unknown
        self.gravity = -model.opt.gravity[2]
        self.dt = model.opt.timestep

        # inertia tensor in the drone frame: MuJoCo stores it diagonalized
        # (body_inertia) in a principal frame rotated by body_iquat
        principal_axes = np.zeros(9)
        mujoco.mju_quat2Mat(principal_axes, model.body_iquat[self.body_id])
        principal_axes = principal_axes.reshape(3, 3)
        self.inertia = principal_axes @ np.diag(model.body_inertia[self.body_id]) @ principal_axes.T

        self.allocation_inv = np.linalg.inv(self.allocation_matrix(model, self.body_id))

        self.reference: np.ndarray | None = None  # current setpoint, initialized to the starting position
        self.reference_velocity = np.zeros(3)
        self.integral = np.zeros(3)
        self.thrust_saturated = False  # motors saturated at the previous step (see `mix`), freezes the integral

    @staticmethod
    def allocation_matrix(model: mujoco.MjModel, body_id: int) -> np.ndarray:
        """4x4 matrix: rotor thrusts -> [total thrust, torque x, torque y, torque z] (drone frame).

        Torques are taken about the center of mass. Each rotor pushes along
        z and creates a yaw reaction torque (6th component of `gear`), whose
        direction alternates from one rotor to the next.
        """
        center_of_mass = model.body_ipos[body_id]
        columns = []
        for i in range(model.nu):
            site_id = model.actuator_trnid[i, 0]
            lever_arm = model.site_pos[site_id] - center_of_mass
            force, reaction_torque = model.actuator_gear[i, :3], model.actuator_gear[i, 3:]
            torque = np.cross(lever_arm, force) + reaction_torque
            columns.append([force[2], *torque])
        return np.array(columns).T

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        p = self.params
        # freejoint (first joint): qpos[:3] position (world), qvel[:3] linear velocity (world),
        # qvel[3:] angular velocity (drone frame)
        position, velocity, angular_velocity = data.qpos[:3], data.qvel[:3], data.qvel[3:6]
        rotation = data.xmat[self.body_id].reshape(3, 3)  # columns = drone x, y, z axes in the world frame

        target = data.mocap_pos[self.target_mocap_id]
        w, x, y, z = data.mocap_quat[self.target_mocap_id]
        target_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        # 1. setpoint trajectory toward the target, then position PID
        reference_acceleration = self.update_reference(position, target)

        error = self.reference - position
        previous_integral = self.integral
        integral_step = error * self.dt
        unwinding = integral_step * self.integral < 0  # per axis: the error opposes the accumulated integral
        integral_step = np.where(unwinding, p.integral_unwind_gain * integral_step, integral_step)
        self.integral = np.clip(self.integral + integral_step, -p.integral_limit, p.integral_limit)

        kp = np.array([p.kp_xy, p.kp_xy, p.kp_z])
        ki = np.array([p.ki_xy, p.ki_xy, p.ki_z])
        kd = np.array([p.kd_xy, p.kd_xy, p.kd_z])
        acceleration = (
            reference_acceleration + kp * error + ki * self.integral + kd * (self.reference_velocity - velocity)
        )
        acceleration[2] += self.gravity

        # the required tilt is atan(a_horizontal / a_vertical): the horizontal part is bounded
        acceleration[2] = max(acceleration[2], 0.2 * self.gravity)  # always push upward a little
        horizontal = acceleration[:2]
        max_horizontal = acceleration[2] * np.tan(p.max_tilt)
        tilt_limited = np.linalg.norm(horizontal) > max_horizontal
        if tilt_limited:
            acceleration[:2] = horizontal * (max_horizontal / np.linalg.norm(horizontal))

        # anti-windup: when the command is saturated (tilt bounded or motors at maximum), the error
        # can't be reduced (e.g. target out of reach of a tethered rope) and the integral would grow up
        # to its limit; when the target became reachable again, the drone took more than 8 s to settle
        if tilt_limited or self.thrust_saturated:
            self.integral = previous_integral

        # 2. total thrust (projected on the drone's current z axis) and desired attitude
        thrust = self.mass * acceleration @ rotation[:, 2]
        z_desired = acceleration / np.linalg.norm(acceleration)
        heading = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        y_desired = np.cross(z_desired, heading)
        y_desired /= np.linalg.norm(y_desired)
        x_desired = np.cross(y_desired, z_desired)
        rotation_desired = np.column_stack([x_desired, y_desired, z_desired])

        # 3. attitude PD: orientation error e_R = 1/2 vee(R_d^T R - R^T R_d)
        error_matrix = rotation_desired.T @ rotation - rotation.T @ rotation_desired
        attitude_error = 0.5 * np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        angular_acceleration = -p.kp_attitude * attitude_error - p.kd_attitude * angular_velocity
        torque = self.inertia @ angular_acceleration + np.cross(angular_velocity, self.inertia @ angular_velocity)

        # 4. mixing to the 4 rotors
        return self.mix(model, thrust, torque)

    def update_reference(self, position: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Move the setpoint toward the target (bounded speed and acceleration), return its acceleration.

        The target can jump (dragged in the viewer): following that jump
        directly would violently flip the drone. The setpoint therefore
        moves toward it at `max_speed` at most, and accelerates and brakes
        at `max_acceleration` (wanted speed v = sqrt(2 a d) near the target).
        The returned acceleration is used as a feedforward term: without
        it, the drone lags behind the setpoint's speed changes and
        overshoots the target by ~20 cm.
        """
        p = self.params
        if self.reference is None:
            self.reference = position.copy()
            self.reference_velocity = np.zeros(3)

        to_target = target - self.reference
        distance = np.linalg.norm(to_target)
        if distance < 1e-6:
            desired_velocity = np.zeros(3)
        else:
            desired_speed = min(p.max_speed, np.sqrt(2 * p.max_acceleration * distance))
            desired_velocity = to_target / distance * desired_speed

        velocity_change = desired_velocity - self.reference_velocity
        max_change = p.max_acceleration * self.dt
        change_norm = np.linalg.norm(velocity_change)
        if change_norm > max_change:
            velocity_change *= max_change / change_norm

        self.reference_velocity += velocity_change
        step = self.reference_velocity * self.dt
        if distance < 1e-6 or np.linalg.norm(step) >= distance:
            # arrival: snap exactly onto the target
            self.reference = target.copy()
            self.reference_velocity = np.zeros(3)
            return np.zeros(3)
        self.reference += step
        return velocity_change / self.dt

    def mix(self, model: mujoco.MjModel, thrust: float, torque: np.ndarray) -> np.ndarray:
        """Convert total thrust + torques into rotor thrusts, with priorities when saturated.

        Priority 1, roll/pitch before total thrust: if a rotor exceeds its
        maximum thrust, all rotors are lowered by the same amount, which
        keeps the differences between rotors, hence the torques. Clipping
        each rotor on its own would destroy those differences: pulling with
        all its strength on a tethered rope, the drone stayed stuck at a 43°
        tilt.

        Priority 2, thrust and roll/pitch before yaw. Yaw is only produced
        by the propellers' reaction torque (~0.02 N·m per N of thrust): it
        needs very large thrust differences between rotors. Without care, a
        large heading change requires negative thrusts, clipped to 0 by
        saturation, which corrupts the total thrust and roll/pitch (the
        drone flew up to 7 m during a 90° turn). As on real autopilots, the
        yaw torque is therefore reduced just enough to keep the thrusts
        within the motor limits: thrust and roll/pitch take priority, the
        turn is simply slower.
        """
        low, high = model.actuator_ctrlrange.T
        without_yaw = self.allocation_inv @ np.array([thrust, torque[0], torque[1], 0.0])

        # priority 1: lower the total thrust just enough for the most loaded rotor to fit
        collective = self.allocation_inv[:, 0]  # share of each rotor in 1 N of total thrust
        excess = np.max((without_yaw - high) / collective)
        self.thrust_saturated = excess > 0
        if self.thrust_saturated:
            without_yaw -= excess * collective

        yaw_only = self.allocation_inv @ np.array([0.0, 0.0, 0.0, torque[2]])

        # largest yaw fraction s such that low <= without_yaw + s * yaw_only <= high
        scale = 1.0
        for base, delta, lo, hi in zip(without_yaw, yaw_only, low, high):
            if delta > 0:
                scale = min(scale, (hi - base) / delta)
            elif delta < 0:
                scale = min(scale, (lo - base) / delta)
        return without_yaw + max(scale, 0.0) * yaw_only


# Each entry builds the control law from the model and the command-line arguments.
CONTROLLERS: dict[str, Callable[[mujoco.MjModel, argparse.Namespace], Controller]] = {
    "hover": lambda _model, _args: hover_control,
    "off": lambda _model, _args: off_control,
    "altitude": lambda model, args: AltitudePID(model, AltitudePIDParams(target_altitude=args.target_altitude)),
    "position": lambda model, _args: PositionController(model, PositionControllerParams()),
}


def run_simulation(model: mujoco.MjModel, data: mujoco.MjData, control_fn: Controller) -> None:
    ctrl_low, ctrl_high = model.actuator_ctrlrange.T

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # camera that follows the drone while staying mouse-controllable (zoom, rotation)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("x2").id
        viewer.cam.distance = 1.5  # zoom on the drone
        viewer.cam.azimuth = 180  # view from behind the drone
        viewer.cam.elevation = -20.0  # viewing angle (negative = from above)

        while viewer.is_running():
            step_start = time.time()

            data.ctrl[:] = np.clip(control_fn(model, data), ctrl_low, ctrl_high)
            logger.debug("t=%.2f pos=%s ctrl=%s", data.time, np.round(data.qpos[:3], 3), np.round(data.ctrl, 3))

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        choices=sorted(CONTROLLERS),
        default="hover",
        help="control law used during the simulation",
    )
    parser.add_argument(
        "--target-altitude",
        type=float,
        default=AltitudePIDParams.target_altitude,
        help="target altitude in meters (altitude controller)",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="initial target position in meters (position controller), can then be moved in the viewer",
    )
    parser.add_argument(
        "--payload",
        action="store_true",
        help="hang a load under the drone (rod + sphere, ball joint), unknown to the controllers",
    )
    parser.add_argument(
        "--rod-count",
        type=int,
        default=PayloadParams.rod_count,
        help="number of rods linked by ball joints (1 = pendulum, 2 = double pendulum; with --payload)",
    )
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=PayloadParams.sphere_mass,
        help="mass of the sphere in kg (with --payload)",
    )
    parser.add_argument(
        "--rod-length",
        type=float,
        default=PayloadParams.rod_length,
        help="length of each rod in meters (with --payload)",
    )
    parser.add_argument(
        "--rope",
        action="store_true",
        help="attach a flexible rope under the drone (mass spread evenly, partly lying on the ground at start)",
    )
    parser.add_argument(
        "--rope-anchor",
        action="store_true",
        help="pin the other end of the rope to the ground (with --rope)",
    )
    parser.add_argument(
        "--rope-length",
        type=float,
        default=None,
        help=f"rope length in meters (with --rope; default {RopeParams.length} m, "
        f"{ANCHORED_ROPE_LENGTH} m with --rope-anchor)",
    )
    parser.add_argument(
        "--rope-mass", type=float, default=RopeParams.mass, help="total rope mass in kg (with --rope)"
    )
    parser.add_argument(
        "--rope-points",
        type=int,
        default=RopeParams.point_count,
        help="number of point masses in the rope (with --rope)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print position and rotor thrusts at each step"
    )
    args = parser.parse_args()
    if args.payload and args.rope:
        parser.error("--payload and --rope are mutually exclusive: choose one")
    if args.rope_anchor and not args.rope:
        parser.error("--rope-anchor requires --rope")
    if args.rope_length is None:
        args.rope_length = ANCHORED_ROPE_LENGTH if args.rope_anchor else RopeParams.length
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    payload = (
        PayloadParams(rod_count=args.rod_count, rod_length=args.rod_length, sphere_mass=args.payload_mass)
        if args.payload
        else None
    )
    rope = (
        RopeParams(
            length=args.rope_length, mass=args.rope_mass, point_count=args.rope_points, anchored=args.rope_anchor
        )
        if args.rope
        else None
    )
    model = build_model(payload, rope)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)  # drone 30 cm above the ground (higher with a load)
    target_mocap_id = model.body("target").mocapid[0]
    if args.target_position is not None:
        data.mocap_pos[target_mocap_id] = args.target_position
    elif rope is not None and rope.anchored:
        # tethered rope: a target straight above the start would be out of reach (rope taut from the
        # start), so it is placed halfway between the drone and the anchor, 1 m high
        anchor = model.geom("rope_anchor").pos
        data.mocap_pos[target_mocap_id] = [(data.qpos[0] + anchor[0]) / 2, (data.qpos[1] + anchor[1]) / 2, 1.0]
    else:
        # default target (scene.xml) at least 40 cm above the load length: otherwise the load would
        # touch the ground (the drone sags at takeoff, since the load is unknown to it)
        load_length = payload.total_length if payload else rope.length if rope else 0.0
        data.mocap_pos[target_mocap_id, 2] = max(data.mocap_pos[target_mocap_id, 2], load_length + 0.4)

    control_fn = CONTROLLERS[args.controller](model, args)
    run_simulation(model, data, control_fn)


if __name__ == "__main__":
    main()
