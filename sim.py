"""CabledDrone : simulation MuJoCo d'un drone quadrirotor (Skydio X2) relié par des câbles.

Le modèle MJCF (x2.xml + scene.xml) est une copie de celui de
mujoco_menagerie/skydio_x2. Le script lance une simulation temps réel avec
le viewer MuJoCo, pilotée par la loi de commande choisie via `--controller`.
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
    """Charge suspendue sous le drone : une chaîne de tiges rigides terminée par une sphère.

    Chaque tige est reliée à la précédente (la première au drone) par une
    liaison rotule (3 rotations libres, sans amortissement). Avec une tige,
    c'est un pendule sphérique ; avec deux, un double pendule sphérique,
    dont le mouvement est chaotique. La première rotule est au centre du
    dessous du drone, ~5 cm sous son centre de masse.
    """

    rod_count: int = 1
    rod_length: float = 0.5  # m, longueur de chaque tige
    rod_radius: float = 0.005  # m
    rod_mass: float = 0.02  # kg, masse de chaque tige
    sphere_radius: float = 0.05  # m
    sphere_mass: float = 0.3  # kg
    ground_clearance: float = 0.1  # m, hauteur de la sphère au-dessus du sol au départ
    # Pas de simulation imposé avec une charge (x2.xml : 0.01 s). Une tige fine n'a quasiment pas
    # d'inertie autour de son axe (~2.5e-7 kg·m²) : avec deux tiges, la tige intermédiaire, sans sphère
    # pour l'alourdir, fait diverger la simulation à 0.01 s (NaN, drone retourné dès 0.8 kg).
    timestep: float = 0.002  # s

    @property
    def total_length(self) -> float:
        """Distance verticale entre le dessous du drone et le bas de la sphère, chaîne au repos."""
        return self.rod_count * self.rod_length + self.sphere_radius


def add_payload(spec: mujoco.MjSpec, payload: PayloadParams) -> None:
    """Ajoute la chaîne de tiges et la sphère au drone "x2", et adapte le keyframe "hover"."""
    parent, attach_pos = spec.body("x2"), [0, 0, 0]
    for i in range(1, payload.rod_count + 1):
        # chaque tige pend sous son parent, rotule à son extrémité haute
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

    # keyframe : ajoute l'orientation de chaque rotule (quaternion identité = tige verticale)
    # et monte le drone pour que la sphère ne touche pas le sol au départ
    hover = spec.key("hover")
    qpos = list(hover.qpos)
    qpos[2] = max(qpos[2], payload.total_length + payload.ground_clearance)
    hover.qpos = qpos + [1, 0, 0, 0] * payload.rod_count


@dataclass
class RopeParams:
    """Corde souple accrochée sous le drone, masse répartie uniformément sur sa longueur.

    Modélisée par un `flexcomp` 1D : une chaîne de points matériels (sans
    orientation) reliés par des segments de longueur imposée. La corde n'a
    donc aucune raideur en flexion ni en torsion, comme une vraie corde.

    Elle est accrochée au drone par une liaison ponctuelle (`connect` :
    3 translations bloquées, aucune rotation) : comme un nœud, elle ne
    transmet que la traction. Une rotule ajouterait la rotation de la corde
    autour de son propre axe, sans signification physique pour une corde et
    d'inertie quasi nulle (cause de la divergence du double pendule rigide).

    Avec `anchored`, l'autre extrémité est épinglée au sol (là où elle repose
    au départ) : le drone ne peut alors pas s'éloigner de l'ancre de plus
    de la longueur de la corde.
    """

    length: float = 1.0  # m
    anchored: bool = False
    mass: float = 0.2  # kg, répartie uniformément sur les points
    point_count: int = 21  # points matériels, soit point_count - 1 segments
    radius: float = 0.006  # m, rayon de collision et d'affichage
    attach_offset: float = -0.01  # m, accroche sous l'origine du drone (évite le contact permanent avec sa coque)
    # Résistance de l'air, en N·s/m par mètre de corde, appliquée sur chaque point : sans elle, la corde
    # oscille indéfiniment. Avec 0.1, une oscillation s'éteint en quelques secondes.
    drag: float = 0.1
    # Contraintes de longueur des segments et d'accroche. Les contraintes MuJoCo sont « souples », et leur
    # raideur dépend de la masse en jeu : avec les valeurs par défaut (solref 0.02), un point de 10 g qui
    # porte tout le poids de la corde laissait le premier segment s'allonger de 8 %. Avec ces valeurs,
    # l'allongement reste sous 0.3 %. solref doit rester supérieur à 2 * timestep.
    solref: str = "0.004 1"
    solimp: str = "0.99 0.999 0.001"
    timestep: float = 0.002  # s


# Corde ancrée : plus longue par défaut, pour laisser au drone de la marge autour de l'ancre.
ANCHORED_ROPE_LENGTH = 2.0  # m


def rope_points(rope: RopeParams, top: np.ndarray) -> np.ndarray:
    """Forme initiale de la corde : verticale sous le point d'accroche `top` jusqu'au sol, le reste posé au sol.

    La longueur au repos de chaque segment est celle de la forme initiale : tous
    les segments doivent donc y avoir la même longueur. Le segment du coude
    descend en diagonale jusqu'au sol pour respecter cette longueur.
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
    """MJCF qui inclut scene.xml et y ajoute la corde, accrochée au drone.

    Un `flexcomp` n'existe que dans le format XML : MjSpec ne sait pas en
    créer, et le greffer depuis un autre MjSpec (`attach`) perd ses
    contraintes. On génère donc ce MJCF sous forme de texte.
    """
    points = rope_points(rope, top)
    point = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    element = " ".join(f"{i} {i + 1}" for i in range(rope.point_count - 1))
    pin_xml = anchor_xml = ""
    if rope.anchored:
        # dernier point épinglé : flexcomp ne lui crée pas de corps, il est fixé au monde.
        # Le plot sombre (sans collision) ne sert qu'à voir l'ancre dans le viewer.
        anchor_x, anchor_y, _ = points[-1]
        pin_xml = f'''
      <pin id="{rope.point_count - 1}"/>'''
        anchor_xml = f'''
    <geom name="rope_anchor" type="cylinder" size=".03 .01" pos="{anchor_x:.6f} {anchor_y:.6f} .01"
          rgba=".2 .2 .2 1" contype="0" conaffinity="0"/>'''
    return f"""<mujoco>
  <!-- chemins absolus : sans fichier d'origine, MuJoCo ne retrouve pas les dossiers relatifs -->
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
    """Compile scene.xml, avec la charge rigide (`payload`) ou la corde (`rope`) si fournie."""
    if rope is None:
        spec = mujoco.MjSpec.from_file(SCENE_PATH)
        if payload is not None:
            add_payload(spec, payload)
        return spec.compile()

    # La liaison `connect` relie les deux corps dans leur position de référence (qpos0), où le drone est
    # à la position de son corps dans x2.xml. On y place le drone à sa position de départ (keyframe
    # "hover"), pour que la corde, construite sous ce point de départ, y soit déjà accrochée.
    scene_spec = mujoco.MjSpec.from_file(SCENE_PATH)  # garder une référence : les vecteurs lus pointent dans sa mémoire
    start = np.array(scene_spec.key("hover").qpos)[:3]
    spec = mujoco.MjSpec.from_string(rope_mjcf(rope, start + [0, 0, rope.attach_offset]))
    spec.body("x2").pos = start
    spec.option.timestep = min(spec.option.timestep, rope.timestep)
    damping = rope.drag * rope.length / rope.point_count
    for joint in spec.joints:  # les 3 glissières de chaque point de la corde (sans nom, créées par flexcomp)
        if joint.parent.name.startswith("rope_"):
            joint.damping = [damping, 0, 0]
    return spec.compile()


# Une loi de commande renvoie les poussées des 4 rotors (N), dans l'ordre thrust1..thrust4.
Controller = Callable[[mujoco.MjModel, mujoco.MjData], np.ndarray]


def hover_control(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Poussée constante du keyframe "hover" : compense exactement le poids, en boucle ouverte.

    Aucun retour d'état : la moindre perturbation (ex. une force appliquée
    dans le viewer) fait dériver le drone.
    """
    return model.key("hover").ctrl.copy()


def off_control(model: mujoco.MjModel, _data: mujoco.MjData) -> np.ndarray:
    """Moteurs coupés : le drone tombe."""
    return np.zeros(model.nu)


@dataclass
class AltitudePIDParams:
    """Consigne et gains du PID d'altitude.

    Les gains sont exprimés en accélération (m/s² par m d'erreur, etc.) : ils
    sont multipliés par la masse du drone, donc ne dépendent pas de celle-ci.
    """

    target_altitude: float = 1.0  # m
    kp: float = 9.0
    ki: float = 3.0
    kd: float = 6.0
    integral_limit: float = 2.0  # m·s, borne l'intégrale (anti-windup)
    max_vertical_speed: float = 0.5  # m/s, vitesse à laquelle la consigne rejoint la cible


class AltitudePID:
    """Maintient le drone à une altitude cible, par un PID sur la poussée totale.

    La poussée totale est répartie également sur les 4 rotors : le PID ne
    corrige que l'altitude, pas l'attitude. Si le drone est incliné (ex.
    perturbation en rotation dans le viewer), il reste incliné et dérive
    horizontalement — ce sera le rôle d'un correcteur d'attitude.

    Poussée totale = m * (g + kp * e + ki * ∫e + kd * (vz_cible - vz)), avec e = z_cible - z :
    - le terme m * g (feedforward) compense le poids, le PID n'a plus qu'à
      corriger l'écart ;
    - le terme dérivé porte sur l'écart de vitesse (vitesse de la consigne -
      vitesse mesurée) : pas d'à-coup quand la cible change, et il ne freine
      pas la montée pendant la rampe ;
    - la consigne z_cible ne saute pas directement à `target_altitude` : elle
      la rejoint à `max_vertical_speed`. Sans cette rampe, une grande montée
      remplit l'intégrale et le drone dépasse nettement la cible (~13 % pour
      0.3 → 1 m).
    """

    def __init__(self, model: mujoco.MjModel, params: AltitudePIDParams) -> None:
        self.params = params
        self.mass = model.body_mass[model.body("x2").id]  # drone seul : une éventuelle charge est inconnue
        self.gravity = -model.opt.gravity[2]
        self.dt = model.opt.timestep
        self.integral = 0.0
        self.reference: float | None = None  # consigne courante, initialisée à l'altitude de départ

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        p = self.params
        # le freejoint est le premier joint : qpos[2] est l'altitude, qvel[2] la vitesse verticale (repère monde)
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
    """Gains du contrôleur de position (boucle externe) et d'attitude (boucle interne).

    Comme pour `AltitudePIDParams`, les gains sont exprimés en accélération
    (linéaire ou angulaire) : ils sont multipliés par la masse ou l'inertie.
    """

    kp_xy: float = 6.0
    ki_xy: float = 0.5
    kd_xy: float = 5.0
    kp_z: float = 9.0
    ki_z: float = 3.0
    kd_z: float = 6.0
    # m·s, par axe (anti-windup). Borne aussi la charge inconnue compensable : ki_z * limite * masse du drone
    # = 24 N, soit ~2.4 kg (avec 2.0, le drone ne portait pas plus de ~0.8 kg)
    integral_limit: float = 6.0
    # L'intégrale se vide `integral_unwind_gain` fois plus vite qu'elle ne se remplit (quand l'erreur change de
    # signe). Après avoir tiré sur une corde ancrée vers une cible hors de portée, elle est pleine ; sans
    # décharge rapide, le drone restait à ~60 cm de la cible 8 s après qu'elle était redevenue atteignable.
    integral_unwind_gain: float = 10.0
    max_speed: float = 1.0  # m/s, vitesse à laquelle la consigne rejoint la cible
    max_acceleration: float = 1.0  # m/s², accélération/freinage de la consigne
    max_tilt: float = np.deg2rad(30)  # inclinaison maximale demandée au drone
    kp_attitude: float = 100.0  # rad/s² par rad d'erreur d'orientation
    kd_attitude: float = 20.0  # rad/s² par rad/s de vitesse angulaire


class PositionController:
    """Amène le drone à la position et au cap (lacet) du corps mocap "target".

    Contrôle en cascade, comme sur les autopilotes de drones réels :
    1. boucle externe (position) : la consigne suit une trajectoire douce
       vers la cible (`update_reference`), un PID par axe donne
       l'accélération voulue, à laquelle on ajoute g pour compenser le poids ;
    2. cette accélération fixe à la fois la poussée totale et l'orientation
       voulue : le drone doit incliner son axe z dans la direction de
       l'accélération (c'est en s'inclinant qu'il se déplace latéralement),
       puis tourner autour de cet axe pour atteindre le cap visé ;
    3. boucle interne (attitude) : un PD sur l'erreur d'orientation donne les
       couples de roulis, tangage et lacet (contrôleur géométrique de Lee et
       al., 2010, qui reste valable pour les grands angles) ;
    4. mixage : la poussée totale et les 3 couples sont convertis en poussées
       des 4 rotors en inversant la matrice d'allocation, calculée depuis la
       position des rotors dans x2.xml.

    La boucle interne doit être nettement plus rapide que la boucle externe
    (ici ~10 rad/s contre ~2.5 rad/s) : la boucle externe suppose que le drone
    prend quasi instantanément l'orientation demandée.
    """

    def __init__(self, model: mujoco.MjModel, params: PositionControllerParams) -> None:
        self.params = params
        self.body_id = model.body("x2").id
        self.target_mocap_id = model.body("target").mocapid[0]
        self.mass = model.body_mass[self.body_id]  # drone seul : une éventuelle charge est inconnue
        self.gravity = -model.opt.gravity[2]
        self.dt = model.opt.timestep

        # tenseur d'inertie dans le repère du drone : MuJoCo le stocke diagonalisé
        # (body_inertia) dans un repère principal tourné de body_iquat
        principal_axes = np.zeros(9)
        mujoco.mju_quat2Mat(principal_axes, model.body_iquat[self.body_id])
        principal_axes = principal_axes.reshape(3, 3)
        self.inertia = principal_axes @ np.diag(model.body_inertia[self.body_id]) @ principal_axes.T

        self.allocation_inv = np.linalg.inv(self.allocation_matrix(model, self.body_id))

        self.reference: np.ndarray | None = None  # consigne courante, initialisée à la position de départ
        self.reference_velocity = np.zeros(3)
        self.integral = np.zeros(3)
        self.thrust_saturated = False  # moteurs saturés au pas précédent (voir `mix`), gèle l'intégrale

    @staticmethod
    def allocation_matrix(model: mujoco.MjModel, body_id: int) -> np.ndarray:
        """Matrice 4x4 : poussées des rotors -> [poussée totale, couple x, couple y, couple z] (repère drone).

        Les couples sont pris autour du centre de masse. Chaque rotor pousse
        selon z et crée un couple de lacet par réaction (6e composante de
        `gear`), de sens alterné d'un rotor à l'autre.
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
        # freejoint (premier joint) : qpos[:3] position (monde), qvel[:3] vitesse linéaire (monde), qvel[3:] vitesse angulaire (repère drone)
        position, velocity, angular_velocity = data.qpos[:3], data.qvel[:3], data.qvel[3:6]
        rotation = data.xmat[self.body_id].reshape(3, 3)  # colonnes = axes x, y, z du drone dans le repère monde

        target = data.mocap_pos[self.target_mocap_id]
        w, x, y, z = data.mocap_quat[self.target_mocap_id]
        target_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        # 1. trajectoire de consigne vers la cible, puis PID de position
        reference_acceleration = self.update_reference(position, target)

        error = self.reference - position
        previous_integral = self.integral
        integral_step = error * self.dt
        unwinding = integral_step * self.integral < 0  # par axe : l'erreur s'oppose à l'intégrale accumulée
        integral_step = np.where(unwinding, p.integral_unwind_gain * integral_step, integral_step)
        self.integral = np.clip(self.integral + integral_step, -p.integral_limit, p.integral_limit)

        kp = np.array([p.kp_xy, p.kp_xy, p.kp_z])
        ki = np.array([p.ki_xy, p.ki_xy, p.ki_z])
        kd = np.array([p.kd_xy, p.kd_xy, p.kd_z])
        acceleration = (
            reference_acceleration + kp * error + ki * self.integral + kd * (self.reference_velocity - velocity)
        )
        acceleration[2] += self.gravity

        # l'inclinaison nécessaire est atan(a_horizontale / a_verticale) : on borne la partie horizontale
        acceleration[2] = max(acceleration[2], 0.2 * self.gravity)  # toujours pousser un minimum vers le haut
        horizontal = acceleration[:2]
        max_horizontal = acceleration[2] * np.tan(p.max_tilt)
        tilt_limited = np.linalg.norm(horizontal) > max_horizontal
        if tilt_limited:
            acceleration[:2] = horizontal * (max_horizontal / np.linalg.norm(horizontal))

        # anti-windup : quand la commande est saturée (inclinaison bornée ou moteurs au maximum), l'erreur
        # ne peut pas se résorber (ex. cible hors de portée d'une corde ancrée) et l'intégrale grossirait
        # jusqu'à sa borne ; au retour d'une cible atteignable, le drone mettait alors plus de 8 s à se recaler
        if tilt_limited or self.thrust_saturated:
            self.integral = previous_integral

        # 2. poussée totale (projetée sur l'axe z actuel du drone) et orientation voulue
        thrust = self.mass * acceleration @ rotation[:, 2]
        z_desired = acceleration / np.linalg.norm(acceleration)
        heading = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        y_desired = np.cross(z_desired, heading)
        y_desired /= np.linalg.norm(y_desired)
        x_desired = np.cross(y_desired, z_desired)
        rotation_desired = np.column_stack([x_desired, y_desired, z_desired])

        # 3. PD d'attitude : erreur d'orientation e_R = 1/2 vee(R_d^T R - R^T R_d)
        error_matrix = rotation_desired.T @ rotation - rotation.T @ rotation_desired
        attitude_error = 0.5 * np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        angular_acceleration = -p.kp_attitude * attitude_error - p.kd_attitude * angular_velocity
        torque = self.inertia @ angular_acceleration + np.cross(angular_velocity, self.inertia @ angular_velocity)

        # 4. mixage vers les 4 rotors
        return self.mix(model, thrust, torque)

    def update_reference(self, position: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fait avancer la consigne vers la cible (vitesse et accélération bornées), renvoie son accélération.

        La cible peut sauter d'un coup (déplacement dans le viewer) : suivre
        directement ce saut ferait basculer violemment le drone. La consigne
        la rejoint donc à `max_speed` au plus, accélère et freine à
        `max_acceleration` (vitesse visée v = sqrt(2 a d) près de la cible).
        L'accélération renvoyée sert d'anticipation (feedforward) : sans
        elle, le drone réagit en retard aux changements de vitesse de la
        consigne et dépasse la cible d'environ 20 cm.
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
            # arrivée : on se cale exactement sur la cible
            self.reference = target.copy()
            self.reference_velocity = np.zeros(3)
            return np.zeros(3)
        self.reference += step
        return velocity_change / self.dt

    def mix(self, model: mujoco.MjModel, thrust: float, torque: np.ndarray) -> np.ndarray:
        """Convertit poussée totale + couples en poussées des rotors, avec des priorités en cas de saturation.

        Priorité 1, roulis/tangage, avant la poussée totale : si un rotor
        dépasse sa poussée maximale, on baisse la poussée de tous les rotors
        du même montant, ce qui conserve les écarts entre rotors, donc les
        couples. Écrêter chaque rotor séparément détruirait ces écarts : le
        drone, tirant de toutes ses forces sur une corde ancrée, restait
        bloqué incliné à 43°.

        Priorité 2, poussée et roulis/tangage avant le lacet. Le lacet n'est produit que par le couple de réaction des hélices
        (~0.02 N·m par N de poussée) : il demande de très grands écarts de
        poussée entre rotors. Sans précaution, un grand changement de cap
        exige des poussées négatives, ramenées à 0 par la saturation, ce qui
        fausse la poussée totale et le roulis/tangage (le drone s'envolait à
        7 m lors d'un virage de 90°). Comme sur les autopilotes réels, le
        couple de lacet est donc réduit juste assez pour que les poussées
        restent dans les limites des moteurs : poussée et roulis/tangage
        sont prioritaires, le virage se fait simplement plus lentement.
        """
        low, high = model.actuator_ctrlrange.T
        without_yaw = self.allocation_inv @ np.array([thrust, torque[0], torque[1], 0.0])

        # priorité 1 : baisser la poussée totale juste assez pour que le rotor le plus chargé tienne
        collective = self.allocation_inv[:, 0]  # part de chaque rotor dans 1 N de poussée totale
        excess = np.max((without_yaw - high) / collective)
        self.thrust_saturated = excess > 0
        if self.thrust_saturated:
            without_yaw -= excess * collective

        yaw_only = self.allocation_inv @ np.array([0.0, 0.0, 0.0, torque[2]])

        # plus grande fraction s de lacet telle que low <= without_yaw + s * yaw_only <= high
        scale = 1.0
        for base, delta, lo, hi in zip(without_yaw, yaw_only, low, high):
            if delta > 0:
                scale = min(scale, (hi - base) / delta)
            elif delta < 0:
                scale = min(scale, (lo - base) / delta)
        return without_yaw + max(scale, 0.0) * yaw_only


# Chaque entrée construit la loi de commande à partir du modèle et des arguments de la ligne de commande.
CONTROLLERS: dict[str, Callable[[mujoco.MjModel, argparse.Namespace], Controller]] = {
    "hover": lambda _model, _args: hover_control,
    "off": lambda _model, _args: off_control,
    "altitude": lambda model, args: AltitudePID(model, AltitudePIDParams(target_altitude=args.target_altitude)),
    "position": lambda model, _args: PositionController(model, PositionControllerParams()),
}


def run_simulation(model: mujoco.MjModel, data: mujoco.MjData, control_fn: Controller) -> None:
    ctrl_low, ctrl_high = model.actuator_ctrlrange.T

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # caméra qui suit le drone tout en restant pilotable à la souris (zoom, rotation)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("x2").id
        viewer.cam.distance = 1.5  # zoom sur le drone
        viewer.cam.azimuth = 180  # vue de derrière le drone
        viewer.cam.elevation = -20.0  # angle de surplomb (négatif = au dessus)

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
        help="loi de commande utilisée pendant la simulation",
    )
    parser.add_argument(
        "--target-altitude",
        type=float,
        default=AltitudePIDParams.target_altitude,
        help="altitude cible en mètres (contrôleur altitude)",
    )
    parser.add_argument(
        "--target-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="position initiale de la cible en mètres (contrôleur position), déplaçable ensuite dans le viewer",
    )
    parser.add_argument(
        "--payload",
        action="store_true",
        help="suspend une charge sous le drone (tige + sphère, liaison rotule), inconnue des contrôleurs",
    )
    parser.add_argument(
        "--rod-count",
        type=int,
        default=PayloadParams.rod_count,
        help="nombre de tiges reliées par des rotules (1 = pendule, 2 = double pendule ; avec --payload)",
    )
    parser.add_argument(
        "--payload-mass",
        type=float,
        default=PayloadParams.sphere_mass,
        help="masse de la sphère en kg (avec --payload)",
    )
    parser.add_argument(
        "--rod-length",
        type=float,
        default=PayloadParams.rod_length,
        help="longueur de chaque tige en mètres (avec --payload)",
    )
    parser.add_argument(
        "--rope",
        action="store_true",
        help="accroche une corde souple sous le drone (masse répartie, posée en partie au sol au départ)",
    )
    parser.add_argument(
        "--rope-anchor",
        action="store_true",
        help="épingle l'autre extrémité de la corde au sol (avec --rope)",
    )
    parser.add_argument(
        "--rope-length",
        type=float,
        default=None,
        help=f"longueur de la corde en mètres (avec --rope ; défaut {RopeParams.length} m, "
        f"{ANCHORED_ROPE_LENGTH} m avec --rope-anchor)",
    )
    parser.add_argument(
        "--rope-mass", type=float, default=RopeParams.mass, help="masse totale de la corde en kg (avec --rope)"
    )
    parser.add_argument(
        "--rope-points",
        type=int,
        default=RopeParams.point_count,
        help="nombre de points matériels de la corde (avec --rope)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="affiche la position et les poussées à chaque pas de temps"
    )
    args = parser.parse_args()
    if args.payload and args.rope:
        parser.error("--payload et --rope sont exclusifs : choisir l'un ou l'autre")
    if args.rope_anchor and not args.rope:
        parser.error("--rope-anchor s'utilise avec --rope")
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
    mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)  # drone à 30 cm du sol (plus haut avec charge)
    target_mocap_id = model.body("target").mocapid[0]
    if args.target_position is not None:
        data.mocap_pos[target_mocap_id] = args.target_position
    elif rope is not None and rope.anchored:
        # corde ancrée : la cible à la verticale du départ serait hors de portée (corde tendue dès le
        # départ), on la place à mi-chemin entre le drone et l'ancre, à 1 m de haut
        anchor = model.geom("rope_anchor").pos
        data.mocap_pos[target_mocap_id] = [(data.qpos[0] + anchor[0]) / 2, (data.qpos[1] + anchor[1]) / 2, 1.0]
    else:
        # cible par défaut (scene.xml) au moins 40 cm au-dessus de la longueur de la charge : sinon elle
        # toucherait le sol (le drone s'affaisse au décollage, la charge lui étant inconnue)
        load_length = payload.total_length if payload else rope.length if rope else 0.0
        data.mocap_pos[target_mocap_id, 2] = max(data.mocap_pos[target_mocap_id, 2], load_length + 0.4)

    control_fn = CONTROLLERS[args.controller](model, args)
    run_simulation(model, data, control_fn)


if __name__ == "__main__":
    main()
