"""
╔══════════════════════════════════════════════════════════════════╗
║           AERODYNAMIC SOLVER — Panel Method + BL                ║
║     Surface pressure, forces, streamline velocity field         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy.spatial import ConvexHull
from dataclasses import dataclass, field
from typing import Dict, Optional
import trimesh


@dataclass
class AeroResults:
    """Container for all aerodynamic analysis results."""
    # Mesh data
    vertices: np.ndarray = None
    faces: np.ndarray = None
    face_normals: np.ndarray = None
    face_areas: np.ndarray = None
    face_centers: np.ndarray = None

    # Scalar fields (per-face)
    cp: np.ndarray = None           # Pressure coefficient
    cf: np.ndarray = None           # Skin friction coefficient
    velocity_mag: np.ndarray = None  # Surface velocity magnitude

    # Forces
    drag_N: float = 0.0
    lift_N: float = 0.0
    side_N: float = 0.0
    drag_pressure_N: float = 0.0
    drag_friction_N: float = 0.0

    # Coefficients
    cd: float = 0.0
    cl: float = 0.0
    cs: float = 0.0
    cd_pressure: float = 0.0
    cd_friction: float = 0.0

    # Moments
    cm_pitch: float = 0.0
    cn_yaw: float = 0.0
    croll: float = 0.0

    # Reference values
    frontal_area_m2: float = 0.0
    planform_area_m2: float = 0.0
    velocity_ms: float = 0.0
    reynolds_number: float = 0.0
    dynamic_pressure: float = 0.0
    mach_number: float = 0.0

    # Flow axis auto-detection
    flow_axis: int = 2          # 0=X, 1=Y, 2=Z — auto-detected as longest axis
    lateral_axis: int = 0       # Perpendicular lateral axis
    vertical_axis: int = 1      # Perpendicular vertical axis

    # Physical properties
    mass_kg: float = 0.0
    volume_m3: float = 0.0
    com_mm: np.ndarray = None
    inertia_kgm2: np.ndarray = None

    # Velocity field for streamlines (structured grid)
    vel_field_points: np.ndarray = None
    vel_field_vectors: np.ndarray = None

    # Multi-speed sweep
    speed_sweep: Dict = field(default_factory=dict)


class AeroSolver:
    """
    Aerodynamic solver using improved panel method with:
    - Modified Newtonian pressure distribution
    - Boundary layer skin friction estimation
    - Proper frontal area via convex hull projection
    - Velocity field generation for streamlines
    """

    def __init__(self, mesh: trimesh.Trimesh, config):
        self.mesh = mesh
        self.config = config
        self.results = AeroResults()

    def run_full_analysis(self, velocity_kmh: Optional[float] = None) -> AeroResults:
        """Execute the complete analysis pipeline."""
        v_kmh = velocity_kmh or self.config.solver.velocity_kmh
        print(f"\n{'═'*60}")
        print(f"  AERODYNAMIC ANALYSIS @ {v_kmh:.0f} km/h")
        print(f"{'═'*60}")

        self._prepare_mesh()
        self._compute_reference_areas()
        self._compute_physics()
        self._compute_pressure_distribution(v_kmh)
        self._compute_skin_friction(v_kmh)
        self._compute_forces(v_kmh)
        self._compute_moments()
        self._generate_velocity_field(v_kmh)
        self._run_speed_sweep()

        self._print_results()
        return self.results

    def _prepare_mesh(self):
        """Refine mesh if needed and extract geometry. Auto-detect flow axis."""
        cfg = self.config.solver

        # Subdivide low-poly meshes for better accuracy
        if len(self.mesh.faces) < cfg.max_faces_for_subdivide:
            print(f"  ▸ Refining mesh ({len(self.mesh.faces)} → ", end="")
            self.mesh = self.mesh.subdivide()
            print(f"{len(self.mesh.faces)} faces)")

        # Convert units to meters
        scale = 0.001 if cfg.mesh_units == "mm" else 1.0

        self.results.vertices = self.mesh.vertices * scale
        self.results.faces = self.mesh.faces
        self.results.face_normals = self.mesh.face_normals
        self.results.face_areas = self.mesh.area_faces * (scale ** 2)  # m²
        self.results.face_centers = self.mesh.triangles_center * scale

        # Auto-detect flow axis: longest dimension = car length = flow direction
        extents = self.mesh.extents
        fa = int(np.argmax(extents))  # longest axis
        others = [i for i in range(3) if i != fa]
        # Vertical = the one with smaller range among the other two
        la, va = (others[0], others[1]) if extents[others[0]] >= extents[others[1]] else (others[1], others[0])
        self.results.flow_axis = fa
        self.results.lateral_axis = la
        self.results.vertical_axis = va
        axis_names = ['X', 'Y', 'Z']
        print(f"  ▸ Flow axis: {axis_names[fa]} (length={extents[fa]:.1f}mm), "
              f"Lateral: {axis_names[la]}, Vertical: {axis_names[va]}")

    def _compute_reference_areas(self):
        """Compute frontal and planform areas via convex hull projection."""
        verts = self.results.vertices
        fa = self.results.flow_axis
        la = self.results.lateral_axis
        va = self.results.vertical_axis

        # Frontal area: project onto plane perpendicular to flow (lateral × vertical)
        try:
            frontal_pts = verts[:, [la, va]]
            hull = ConvexHull(frontal_pts)
            self.results.frontal_area_m2 = hull.volume
        except Exception:
            la_range = verts[:, la].max() - verts[:, la].min()
            va_range = verts[:, va].max() - verts[:, va].min()
            self.results.frontal_area_m2 = la_range * va_range * 0.85

        # Planform area: project onto plane perpendicular to vertical (flow × lateral)
        try:
            plan_pts = verts[:, [fa, la]]
            hull = ConvexHull(plan_pts)
            self.results.planform_area_m2 = hull.volume
        except Exception:
            fa_range = verts[:, fa].max() - verts[:, fa].min()
            la_range = verts[:, la].max() - verts[:, la].min()
            self.results.planform_area_m2 = fa_range * la_range * 0.85

        print(f"  ▸ Frontal area: {self.results.frontal_area_m2 * 1e4:.2f} cm²")
        print(f"  ▸ Planform area: {self.results.planform_area_m2 * 1e4:.2f} cm²")

    def _compute_physics(self):
        """Compute mass, volume, center of mass, inertia."""
        cfg = self.config.solver
        density_kg_m3 = cfg.material_density_g_cm3 * 1000.0

        vol_mm3 = self.mesh.volume
        vol_m3 = vol_mm3 / 1e9
        self.results.volume_m3 = vol_m3
        self.results.mass_kg = vol_m3 * density_kg_m3
        self.results.com_mm = self.mesh.center_mass
        self.results.inertia_kgm2 = (self.mesh.moment_inertia * 1e-15) * density_kg_m3
        print(f"  ▸ Mass: {self.results.mass_kg * 1000:.1f} g")

    def _compute_pressure_distribution(self, velocity_kmh: float):
        """Modified Newtonian + wake model for surface Cp (axis-aware)."""
        v_ms = velocity_kmh / 3.6
        air = self.config.air
        fa = self.results.flow_axis
        la = self.results.lateral_axis
        va = self.results.vertical_axis

        # Flow direction: wind blows along negative flow_axis
        flow_dir = np.zeros(3)
        flow_dir[fa] = -1.0  # Wind comes from +flow_axis direction
        normals = self.results.face_normals

        # cos(θ) between face normal and freestream (+flow_axis)
        cos_theta = normals[:, fa]  # dot with +flow_axis unit vector

        # Reynolds & Mach (use flow axis extent as L_ref)
        L_ref = (self.results.vertices[:, fa].max() - self.results.vertices[:, fa].min())
        Re = air.density * v_ms * L_ref / air.dynamic_viscosity
        Ma = v_ms / air.speed_of_sound
        self.results.reynolds_number = Re
        self.results.mach_number = Ma

        cp_max = 1.0
        cp = np.zeros(len(normals))

        # Windward faces
        windward = cos_theta > 0
        cp[windward] = cp_max * (cos_theta[windward] ** 2)
        stagnation = cos_theta > 0.95
        cp[stagnation] *= 1.05

        # Leeward / wake
        leeward = cos_theta <= 0
        cp[leeward] = -0.3 - 0.15 * np.abs(cos_theta[leeward])

        # Side faces (lateral axis)
        side_faces = np.abs(normals[:, la]) > 0.7
        cp[side_faces & ~windward] = np.clip(
            -0.8 * np.abs(cos_theta[side_faces & ~windward]) - 0.3, -1.5, 0
        )
        # Top faces (vertical axis, positive direction)
        top_faces = np.abs(normals[:, va]) > 0.7
        cp[top_faces & ~windward] = np.clip(
            -0.6 * np.abs(cos_theta[top_faces & ~windward]) - 0.4, -1.5, 0
        )

        self.results.cp = cp
        v_ratio = np.sqrt(np.clip(1.0 - cp, 0.0, 3.0))
        self.results.velocity_mag = v_ratio * v_ms

        print(f"  ▸ Re = {Re:.2e}, Ma = {Ma:.4f}")
        print(f"  ▸ Cp range: [{cp.min():.3f}, {cp.max():.3f}]")

    def _compute_skin_friction(self, velocity_kmh: float):
        """Estimate skin friction drag using flat-plate correlations."""
        v_ms = velocity_kmh / 3.6
        air = self.config.air
        cfg = self.config.solver

        normals = self.results.face_normals
        areas = self.results.face_areas
        centers = self.results.face_centers

        # Distance from leading edge along flow axis
        fa = self.results.flow_axis
        x_min = self.results.vertices[:, fa].min()
        x_local = centers[:, fa] - x_min
        x_local = np.clip(x_local, 1e-6, None)

        # Local Reynolds number
        Re_local = air.density * v_ms * x_local / air.dynamic_viscosity

        # Skin friction coefficient (Blasius laminar / Schlichting turbulent)
        cf = np.zeros(len(normals))
        laminar = Re_local < cfg.transition_reynolds
        turbulent = ~laminar

        cf[laminar] = 0.664 / np.sqrt(np.clip(Re_local[laminar], 1, None))
        cf[turbulent] = 0.027 / (np.clip(Re_local[turbulent], 1, None) ** (1.0 / 7.0))

        self.results.cf = cf

    def _compute_forces(self, velocity_kmh: float):
        """Integrate surface pressures and friction to get total forces."""
        v_ms = velocity_kmh / 3.6
        air = self.config.air

        q = 0.5 * air.density * v_ms ** 2
        self.results.dynamic_pressure = q
        self.results.velocity_ms = v_ms

        normals = self.results.face_normals
        areas = self.results.face_areas
        cp = self.results.cp
        cf = self.results.cf

        A_ref = self.results.frontal_area_m2

        # Pressure force: F_p = -Cp * q * A * n̂   (per face)
        pressure_forces = (-cp * q * areas)[:, np.newaxis] * normals
        total_pressure = np.sum(pressure_forces, axis=0)

        # Friction force: always opposes flow (+X direction)
        friction_mag = cf * q * areas
        total_friction_x = np.sum(friction_mag)  # All in drag direction

        # Decompose using detected axes
        fa = self.results.flow_axis
        la = self.results.lateral_axis
        va = self.results.vertical_axis
        self.results.drag_pressure_N = -total_pressure[fa]  # Along flow axis
        self.results.drag_friction_N = total_friction_x
        self.results.drag_N = self.results.drag_pressure_N + self.results.drag_friction_N
        self.results.lift_N = total_pressure[va]   # Vertical axis
        self.results.side_N = total_pressure[la]   # Lateral axis

        # Coefficients
        if A_ref > 0 and q > 0:
            self.results.cd = self.results.drag_N / (q * A_ref)
            self.results.cd_pressure = self.results.drag_pressure_N / (q * A_ref)
            self.results.cd_friction = self.results.drag_friction_N / (q * A_ref)
            self.results.cl = self.results.lift_N / (q * A_ref)
            self.results.cs = self.results.side_N / (q * A_ref)

    def _compute_moments(self):
        """Compute aerodynamic moment coefficients about CoM."""
        com = self.results.com_mm * 0.001 if self.results.com_mm is not None else np.zeros(3)
        centers = self.results.face_centers
        normals = self.results.face_normals
        areas = self.results.face_areas
        cp = self.results.cp
        q = self.results.dynamic_pressure

        # Moment arms from CoM
        r = centers - com
        # Pressure forces per face
        f_per_face = (-cp * q * areas)[:, np.newaxis] * normals
        # Moments = r × F
        moments = np.cross(r, f_per_face)
        total_moment = np.sum(moments, axis=0)

        A_ref = self.results.frontal_area_m2
        fa = self.results.flow_axis
        L_ref = self.results.vertices[:, fa].max() - self.results.vertices[:, fa].min()

        if A_ref > 0 and q > 0 and L_ref > 0:
            denom = q * A_ref * L_ref
            self.results.cm_pitch = total_moment[1] / denom
            self.results.cn_yaw = total_moment[2] / denom
            self.results.croll = total_moment[0] / denom

    def _generate_velocity_field(self, velocity_kmh: float):
        """Generate a 3D velocity field around the body using mesh-aware flow.

        Uses trimesh signed-distance and closest-point queries so that
        streamlines and particles wrap around the *actual* car geometry
        instead of a simple ellipsoid approximation.
        """
        from trimesh.proximity import ProximityQuery

        v_ms = velocity_kmh / 3.6
        fa = self.results.flow_axis
        la = self.results.lateral_axis
        va = self.results.vertical_axis

        verts = self.results.vertices
        bounds_min = verts.min(axis=0)
        bounds_max = verts.max(axis=0)
        extent = bounds_max - bounds_min
        center = (bounds_min + bounds_max) / 2

        # --- Build the mesh in solver-space (meters) for proximity queries ---
        solver_mesh = trimesh.Trimesh(
            vertices=self.results.vertices,
            faces=self.results.faces,
            face_normals=self.results.face_normals,
            process=False,
        )
        # Ensure the mesh is watertight for signed distance
        if not solver_mesh.is_watertight:
            solver_mesh.fill_holes()
        proximity = ProximityQuery(solver_mesh)

        # --- Domain ---
        domain_min = np.copy(bounds_min)
        domain_max = np.copy(bounds_max)
        domain_min[fa] -= extent[fa] * 1.5
        domain_max[fa] += extent[fa] * 3.0
        domain_min[la] = center[la] - extent[la] * 1.5
        domain_max[la] = center[la] + extent[la] * 1.5
        domain_min[va] -= extent[va] * 0.3
        domain_max[va] += extent[va] * 1.5

        n = 45  # Higher resolution for better detail
        axes = [np.linspace(domain_min[i], domain_max[i], n) for i in range(3)]
        grids = np.meshgrid(axes[0], axes[1], axes[2], indexing='ij')
        points = np.column_stack([g.ravel() for g in grids])
        n_pts = len(points)

        print(f"  ▸ Computing mesh-aware velocity field ({n}³ = {n**3} pts)...")

        # --- Signed distance: negative = inside mesh ---
        try:
            signed_dist = trimesh.proximity.signed_distance(solver_mesh, points)
        except Exception:
            # Fallback: use unsigned distance + simple inside check
            closest_pts, distances, _ = proximity.on_surface(points)
            signed_dist = distances
            try:
                inside_mask_check = solver_mesh.contains(points)
                signed_dist[inside_mask_check] *= -1
            except Exception:
                pass

        # Closest points and face IDs on the mesh surface
        closest_pts, abs_dist, face_ids = proximity.on_surface(points)

        # Surface normals at the closest face
        surf_normals = self.results.face_normals[face_ids]

        # Characteristic length for normalization
        char_len = extent[fa]

        # --- Initialize freestream ---
        vel = np.zeros((n_pts, 3))
        vel[:, fa] = -v_ms

        # --- Classify regions ---
        inside = signed_dist < 0               # Inside mesh
        near_surface = (signed_dist >= 0) & (abs_dist < char_len * 0.35)
        wake_candidate = signed_dist >= 0       # Outside mesh

        # --- 1. Zero velocity inside the body ---
        vel[inside] = 0.0

        # --- 2. Near-surface deflection: flow wraps around actual geometry ---
        if np.any(near_surface):
            ns_normals = surf_normals[near_surface]
            ns_dist = abs_dist[near_surface]

            # Smooth influence: strongest at surface, fades with distance
            # Normalize distance by characteristic length
            norm_dist = ns_dist / (char_len * 0.35)
            influence = np.clip(1.0 - norm_dist, 0, 1) ** 2

            # Current velocity at these points
            v_current = vel[near_surface].copy()

            # Project out the normal component (make flow tangential)
            # v_tangential = v - (v · n̂) * n̂
            v_dot_n = np.sum(v_current * ns_normals, axis=1, keepdims=True)
            v_normal = v_dot_n * ns_normals
            v_tangential = v_current - v_normal

            # Ensure tangential flow maintains speed
            tang_speed = np.linalg.norm(v_tangential, axis=1, keepdims=True)
            tang_speed = np.clip(tang_speed, 1e-6, None)
            v_tangential = v_tangential / tang_speed * v_ms

            # Push flow outward near surface to prevent penetration
            outward_push = ns_normals * (v_ms * 0.25 * influence[:, np.newaxis])

            # Blend between freestream and tangential+push based on influence
            vel[near_surface] = (
                v_current * (1.0 - influence[:, np.newaxis]) +
                (v_tangential + outward_push) * influence[:, np.newaxis]
            )

        # --- 3. Wake region: downstream of the actual body ---
        # Detect wake by checking if a point is downstream AND the flow-axis
        # coordinate is past the body's trailing edge
        trailing_edge = bounds_max[fa] if vel[0, fa] < 0 else bounds_min[fa]
        # Flow goes in -fa direction, so downstream means coord < trailing_edge
        # Actually: wind blows -fa, so point with coord < body_min[fa] is downstream
        body_min_fa = bounds_min[fa]

        d_from_center = points - center
        is_downstream = (
            (points[:, fa] < body_min_fa) &      # Past trailing edge
            (~inside) &
            (abs_dist < char_len * 1.5)           # Close enough to feel wake
        )

        if np.any(is_downstream):
            # Distance downstream from trailing edge
            downstream_dist = body_min_fa - points[is_downstream, fa]
            # Lateral distance from centerline
            lat_dist = np.sqrt(
                (points[is_downstream, la] - center[la]) ** 2 +
                (points[is_downstream, va] - center[va]) ** 2
            )
            # Wake width grows with downstream distance
            wake_width = extent[la] * 0.4 + downstream_dist * 0.3
            in_wake_core = lat_dist < wake_width

            # Velocity deficit in wake
            wake_factor = np.exp(-downstream_dist / (char_len * 1.5))
            wake_factor[~in_wake_core] *= 0.3  # Weaker outside core

            vel[is_downstream, fa] += v_ms * 0.4 * wake_factor  # Reduce speed
            # Add lateral spreading
            lat_dir = np.zeros((np.sum(is_downstream), 3))
            lat_dir[:, la] = points[is_downstream, la] - center[la]
            lat_dir[:, va] = points[is_downstream, va] - center[va]
            lat_mag = np.linalg.norm(lat_dir, axis=1, keepdims=True)
            lat_mag = np.clip(lat_mag, 1e-6, None)
            lat_dir = lat_dir / lat_mag
            vel[is_downstream] += lat_dir * (v_ms * 0.08 * wake_factor[:, np.newaxis])

        # --- 4. Acceleration around the body (Venturi effect at sides/top) ---
        side_accel = (
            near_surface &
            ~inside &
            (np.abs(surf_normals[:, la]) > 0.5) | (np.abs(surf_normals[:, va]) > 0.5)
        )
        if np.any(side_accel):
            # Speed up tangential flow near curved surfaces
            sa_dist = abs_dist[side_accel]
            sa_influence = np.clip(1.0 - sa_dist / (char_len * 0.2), 0, 1)
            speed = np.linalg.norm(vel[side_accel], axis=1, keepdims=True)
            speed = np.clip(speed, 1e-6, None)
            vel_dir = vel[side_accel] / speed
            vel[side_accel] += vel_dir * (v_ms * 0.2 * sa_influence[:, np.newaxis])

        # --- Store results ---
        self.results.vel_field_points = points.reshape(n, n, n, 3)
        self.results.vel_field_vectors = vel.reshape(n, n, n, 3)

        self.results._vel_grid_origin = tuple(domain_min)
        self.results._vel_grid_spacing = tuple(
            (domain_max[i] - domain_min[i]) / (n - 1) for i in range(3)
        )
        self.results._vel_grid_dims = (n, n, n)

        # Store mesh reference for visualizer particle collision
        self.results._solver_mesh = solver_mesh

        print(f"  ▸ Velocity field: {n}³ = {n**3} mesh-aware points generated")

    def _run_speed_sweep(self):
        """Run analysis at multiple speeds for comparative data."""
        speeds = self.config.solver.velocity_sweep
        sweep = {"speed_kmh": [], "cd": [], "cl": [], "drag_N": [], "lift_N": [],
                 "downforce_N": [], "ld_ratio": [], "power_W": []}

        air = self.config.air
        A_ref = self.results.frontal_area_m2

        fa = self.results.flow_axis
        va = self.results.vertical_axis
        for v_kmh in speeds:
            v_ms = v_kmh / 3.6
            q = 0.5 * air.density * v_ms ** 2

            # Reuse Cp distribution (shape-dependent, not speed-dependent for subsonic)
            cp = self.results.cp
            cf = self.results.cf
            normals = self.results.face_normals
            areas = self.results.face_areas

            pf = (-cp * q * areas)[:, np.newaxis] * normals
            tp = np.sum(pf, axis=0)
            friction_drag = np.sum(cf * q * areas)

            drag = -tp[fa] + friction_drag
            lift = tp[va]
            cd = drag / (q * A_ref) if (q * A_ref) > 0 else 0
            cl = lift / (q * A_ref) if (q * A_ref) > 0 else 0
            power = drag * v_ms

            sweep["speed_kmh"].append(v_kmh)
            sweep["cd"].append(round(cd, 4))
            sweep["cl"].append(round(cl, 4))
            sweep["drag_N"].append(round(drag, 3))
            sweep["lift_N"].append(round(lift, 3))
            sweep["downforce_N"].append(round(-lift, 3))
            sweep["ld_ratio"].append(round(-lift / drag, 3) if drag != 0 else 0)
            sweep["power_W"].append(round(power, 2))

        self.results.speed_sweep = sweep
        print(f"  ▸ Speed sweep: {len(speeds)} velocities analyzed")

    def _print_results(self):
        """Print formatted results summary."""
        r = self.results
        print(f"\n{'─'*50}")
        print(f"  RESULTS SUMMARY")
        print(f"{'─'*50}")
        print(f"  Drag Force:       {r.drag_N:>10.3f} N")
        print(f"    ├─ Pressure:    {r.drag_pressure_N:>10.3f} N")
        print(f"    └─ Friction:    {r.drag_friction_N:>10.3f} N")
        print(f"  Lift Force:       {r.lift_N:>10.3f} N")
        print(f"  Downforce:        {-r.lift_N:>10.3f} N")
        print(f"  Side Force:       {r.side_N:>10.3f} N")
        print(f"  Cd (total):       {r.cd:>10.4f}")
        print(f"  Cl:               {r.cl:>10.4f}")
        print(f"  L/D Ratio:        {-r.lift_N/r.drag_N if r.drag_N else 0:>10.3f}")
        print(f"  Frontal Area:     {r.frontal_area_m2*1e4:>10.2f} cm²")
        print(f"  Re:               {r.reynolds_number:>10.2e}")
        print(f"  Ma:               {r.mach_number:>10.4f}")
        print(f"{'─'*50}")
