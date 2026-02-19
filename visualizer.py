"""
╔══════════════════════════════════════════════════════════════════╗
║       WIND TUNNEL VISUALIZER — PyVista 3D Engine               ║
║    Professional CFD-style rendering with interactive controls   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pyvista as pv
from pathlib import Path


class WindTunnelVisualizer:
    """
    Interactive 3D visualization using PyVista, styled like SimScale/Autodesk CFD.
    Supports: pressure contours, streamlines, velocity fields, section cuts.
    """

    def __init__(self, mesh_trimesh, results, config):
        self.trimesh_mesh = mesh_trimesh
        self.results = results
        self.config = config
        self.plotter = None
        self.pv_mesh = None
        self._screenshots = []

    def _build_pyvista_mesh(self):
        """Convert trimesh to PyVista mesh with scalar data."""
        verts = self.trimesh_mesh.vertices
        faces_tri = self.trimesh_mesh.faces
        # PyVista format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        n_faces = len(faces_tri)
        pv_faces = np.column_stack([
            np.full(n_faces, 3, dtype=np.int64),
            faces_tri
        ]).ravel()

        self.pv_mesh = pv.PolyData(verts, pv_faces)

        # Add scalar fields
        if self.results.cp is not None:
            cp = self.results.cp
            if len(cp) != n_faces:
                # Resample if sizes don't match
                indices = np.linspace(0, len(cp) - 1, n_faces).astype(int)
                cp = cp[indices]
            self.pv_mesh.cell_data["Pressure Coefficient (Cp)"] = cp

        if self.results.velocity_mag is not None:
            vm = self.results.velocity_mag
            if len(vm) != n_faces:
                indices = np.linspace(0, len(vm) - 1, n_faces).astype(int)
                vm = vm[indices]
            self.pv_mesh.cell_data["Surface Velocity (m/s)"] = vm

        if self.results.cf is not None:
            cf = self.results.cf
            if len(cf) != n_faces:
                indices = np.linspace(0, len(cf) - 1, n_faces).astype(int)
                cf = cf[indices]
            self.pv_mesh.cell_data["Skin Friction (Cf)"] = cf

    def _build_velocity_grid(self):
        """Build a PyVista UniformGrid from the velocity field."""
        r = self.results
        if r.vel_field_points is None:
            return None

        dims = r._vel_grid_dims
        origin = r._vel_grid_origin
        spacing = r._vel_grid_spacing

        grid = pv.ImageData(
            dimensions=dims,
            spacing=spacing,
            origin=origin
        )

        vectors = r.vel_field_vectors.reshape(-1, 3)
        magnitude = np.linalg.norm(vectors, axis=1)

        grid.point_data["velocity"] = vectors
        grid.point_data["velocity_magnitude"] = magnitude

        return grid

    def _add_ground_plane(self, plotter):
        """Add a ground plane with grid lines."""
        cfg = self.config.viz
        if not cfg.show_ground_plane:
            return

        verts = self.trimesh_mesh.vertices
        bounds = self.pv_mesh.bounds
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 2

        z_min = verts[:, 2].min()
        center_x = (bounds[0] + bounds[1]) / 2
        center_y = (bounds[2] + bounds[3]) / 2

        ground = pv.Plane(
            center=(center_x, center_y, z_min - 0.5),
            direction=(0, 0, 1),
            i_size=extent * 2,
            j_size=extent * 1.5,
            i_resolution=20,
            j_resolution=15,
        )
        plotter.add_mesh(
            ground,
            color=cfg.ground_color,
            opacity=cfg.ground_opacity,
            show_edges=True,
            edge_color="#3a3a5e",
            line_width=0.5,
            lighting=False,
        )

    def _add_wind_tunnel_walls(self, plotter):
        """Add transparent wind tunnel walls for context."""
        bounds = self.pv_mesh.bounds
        extent_x = bounds[1] - bounds[0]
        extent_y = bounds[3] - bounds[2]
        extent_z = bounds[5] - bounds[4]
        center = self.pv_mesh.center

        # Tunnel box (slightly larger than domain)
        tunnel = pv.Box(bounds=[
            center[0] - extent_x * 2, center[0] + extent_x * 3,
            center[1] - extent_y * 1.5, center[1] + extent_y * 1.5,
            bounds[4] - 1, center[2] + extent_z * 2,
        ])
        plotter.add_mesh(
            tunnel,
            color="#2a4a6e",
            opacity=0.03,
            show_edges=True,
            edge_color="#4a6a8e",
            line_width=1,
            lighting=False,
        )

    def _add_streamlines(self, plotter, grid):
        """Add streamline particle traces through the velocity field (axis-aware)."""
        if grid is None:
            self._add_fallback_streamlines(plotter)
            return

        cfg = self.config.viz
        r = self.results
        fa = r.flow_axis
        la = r.lateral_axis
        va = r.vertical_axis

        bounds = self.pv_mesh.bounds  # [xmin,xmax,ymin,ymax,zmin,zmax]
        bmin = np.array([bounds[0], bounds[2], bounds[4]])
        bmax = np.array([bounds[1], bounds[3], bounds[5]])
        center = (bmin + bmax) / 2
        ext = bmax - bmin

        # Seed points: grid upstream of body along the flow axis
        n_seeds = self.config.solver.streamline_density
        # Lateral spread
        seed_la = np.linspace(center[la] - ext[la] * 0.9, center[la] + ext[la] * 0.9, n_seeds)
        # Vertical spread
        seed_va = np.linspace(bmin[va] + ext[va] * 0.05, bmin[va] + ext[va] * 1.4, int(n_seeds * 0.8))
        # Upstream position along flow axis
        seed_fa = bmin[fa] - ext[fa] * 1.2

        seed_points = []
        for l in seed_la:
            for v in seed_va:
                pt = np.zeros(3)
                pt[fa] = seed_fa
                pt[la] = l
                pt[va] = v
                seed_points.append(pt)
        seed_points = np.array(seed_points)
        seed_poly = pv.PolyData(seed_points)

        try:
            streamlines = grid.streamlines_from_source(
                seed_poly,
                vectors="velocity",
                max_time=ext[fa] * 6,
                max_steps=self.config.solver.streamline_max_steps,
                integration_direction="forward",
                initial_step_length=ext[fa] * 0.008,
            )

            if streamlines.n_points > 0:
                if "velocity" in streamlines.point_data:
                    vel = streamlines.point_data["velocity"]
                    mag = np.linalg.norm(vel, axis=1)
                    streamlines.point_data["Speed"] = mag
                    plotter.add_mesh(
                        streamlines.tube(radius=ext[fa] * cfg.streamline_tube_radius_factor),
                        scalars="Speed", cmap=cfg.streamline_cmap,
                        opacity=cfg.streamline_opacity, show_scalar_bar=False, lighting=True,
                    )
                else:
                    plotter.add_mesh(
                        streamlines.tube(radius=ext[fa] * cfg.streamline_tube_radius_factor),
                        color="#00ccff", opacity=cfg.streamline_opacity, lighting=True,
                    )
                print(f"  \u25b8 Streamlines: {streamlines.n_points} points rendered")
            else:
                print("  \u25b8 Streamlines: No valid streamlines, using fallback")
                self._add_fallback_streamlines(plotter)
        except Exception as e:
            print(f"  \u25b8 Streamlines error: {e}, using fallback")
            self._add_fallback_streamlines(plotter)

    def _add_fallback_streamlines(self, plotter):
        """Add dense line-based streamlines as fallback (axis-aware, more sheets)."""
        r = self.results
        fa = r.flow_axis
        la = r.lateral_axis
        va = r.vertical_axis

        bounds = self.pv_mesh.bounds
        bmin = np.array([bounds[0], bounds[2], bounds[4]])
        bmax = np.array([bounds[1], bounds[3], bounds[5]])
        center = (bmin + bmax) / 2
        ext = bmax - bmin

        # Dense stream sheets: 18 lateral x 12 vertical
        n_la = 18
        n_va = 12
        la_pts = np.linspace(center[la] - ext[la] * 0.8, center[la] + ext[la] * 0.8, n_la)
        va_pts = np.linspace(bmin[va] + ext[va] * 0.05, bmin[va] + ext[va] * 1.3, n_va)

        # Flow goes from +fa to -fa (upstream to downstream)
        start_fa = bmin[fa] - ext[fa] * 1.5
        end_fa = bmax[fa] + ext[fa] * 3.0
        n_pts = 80  # Points per streamline

        for l_val in la_pts:
            for v_val in va_pts:
                pts = np.zeros((n_pts, 3))
                pts[:, fa] = np.linspace(start_fa, end_fa, n_pts)
                pts[:, la] = l_val
                pts[:, va] = v_val

                # Check if this line intersects the body
                hit_la = (l_val > bmin[la]) and (l_val < bmax[la])
                hit_va = (v_val > bmin[va]) and (v_val < bmax[va])

                if hit_la and hit_va:
                    # Turbulent — add deflection around the body
                    noise_la = np.random.normal(0, ext[la] * 0.04, size=n_pts)
                    noise_va = np.random.normal(0, ext[va] * 0.03, size=n_pts)
                    body_region = (pts[:, fa] > bmin[fa] - ext[fa]*0.2) & (pts[:, fa] < bmax[fa] + ext[fa]*0.5)
                    noise_la[body_region] *= 3.0
                    noise_va[body_region] *= 2.5
                    pts[:, la] += noise_la
                    pts[:, va] += noise_va
                    
                    # Simulated velocity drop near body
                    dist_to_center = np.sqrt(((pts[:, la]-center[la])/ext[la])**2 + ((pts[:, va]-center[va])/ext[va])**2)
                    speed_factor = np.clip(dist_to_center * 1.5, 0.3, 1.1)
                else:
                    speed_factor = np.ones(n_pts) * 1.0

                # Color by simulated speed
                speed = r.velocity_ms * speed_factor
                
                line = pv.Spline(pts, n_pts)
                line["Speed"] = speed
                plotter.add_mesh(
                    line, scalars="Speed", cmap="turbo", 
                    clim=[0, r.velocity_ms*1.2],
                    opacity=0.6, line_width=2.0, show_scalar_bar=False,
                    render_lines_as_tubes=True
                )

        print(f"  \u25b8 Fallback streamlines: {n_la}x{n_va} = {n_la*n_va} lines (colored)")

    def _add_force_arrows(self, plotter):
        """Add drag and lift force vectors on the model (axis-aware)."""
        r = self.results
        fa = r.flow_axis
        va = r.vertical_axis
        center = np.array(self.pv_mesh.center)
        bounds = self.pv_mesh.bounds
        extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        scale = extent * self.config.viz.arrow_scale

        # Drag arrow (red, along -flow_axis)
        if abs(r.drag_N) > 0.001:
            drag_dir = np.zeros(3)
            drag_dir[fa] = -1.0 * np.sign(r.drag_N)
            arrow = pv.Arrow(
                start=center, direction=drag_dir,
                scale=scale * min(abs(r.drag_N) / 5.0, 3.0),
                tip_radius=0.15, shaft_radius=0.05,
            )
            plotter.add_mesh(arrow, color=self.config.viz.arrow_color_drag, lighting=True)
            plotter.add_point_labels(
                [center + drag_dir * scale * 1.5],
                [f"Drag: {r.drag_N:.2f} N"],
                font_size=12, text_color="white", font_family="courier",
                shape_color="#ff4444", shape_opacity=0.7,
            )

        # Lift/Downforce arrow (green, along vertical_axis)
        if abs(r.lift_N) > 0.001:
            lift_dir = np.zeros(3)
            lift_dir[va] = 1.0 * np.sign(r.lift_N)
            arrow = pv.Arrow(
                start=center, direction=lift_dir,
                scale=scale * min(abs(r.lift_N) / 5.0, 3.0),
                tip_radius=0.15, shaft_radius=0.05,
            )
            plotter.add_mesh(arrow, color=self.config.viz.arrow_color_lift, lighting=True)
            label = f"Downforce: {-r.lift_N:.2f} N" if r.lift_N < 0 else f"Lift: {r.lift_N:.2f} N"
            plotter.add_point_labels(
                [center + lift_dir * scale * 1.5],
                [label],
                font_size=12, text_color="white", font_family="courier",
                shape_color="#44ff44", shape_opacity=0.7,
            )

    def _get_iso_camera(self):
        """Get axis-aware isometric camera position."""
        fa = self.results.flow_axis
        la = self.results.lateral_axis
        va = self.results.vertical_axis
        center = np.array(self.pv_mesh.center)
        bounds = self.pv_mesh.bounds
        ext = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        max_ext = max(ext)

        cam_pos = np.copy(center)
        cam_pos[fa] -= max_ext * 1.5   # Upstream of flow
        cam_pos[la] -= max_ext * 1.2   # Side offset
        cam_pos[va] += max_ext * 0.8   # Above

        up = np.zeros(3)
        up[va] = 1.0  # Vertical axis is "up"

        return [cam_pos.tolist(), center.tolist(), up.tolist()]

    def _add_hud(self, plotter):
        """Add heads-up display with simulation results."""
        r = self.results
        v_kmh = r.velocity_ms * 3.6

        hud_text = (
            f"╔═══ WIND TUNNEL RESULTS ═══╗\n"
            f"║ V∞ = {v_kmh:.0f} km/h ({r.velocity_ms:.1f} m/s)\n"
            f"║ Re = {r.reynolds_number:.2e}\n"
            f"║ Ma = {r.mach_number:.4f}\n"
            f"╠═══════════════════════════╣\n"
            f"║ Cd = {r.cd:.4f}\n"
            f"║ Cl = {r.cl:.4f}\n"
            f"║ Drag = {r.drag_N:.3f} N\n"
            f"║ Downforce = {-r.lift_N:.3f} N\n"
            f"║ L/D = {-r.lift_N/r.drag_N if r.drag_N else 0:.3f}\n"
            f"╠═══════════════════════════╣\n"
            f"║ Mass = {r.mass_kg*1000:.1f} g\n"
            f"║ A_frontal = {r.frontal_area_m2*1e4:.1f} cm²\n"
            f"╚═══════════════════════════╝"
        )

        plotter.add_text(
            hud_text,
            position="upper_left",
            font_size=9,
            color="white",
            font="courier",
            shadow=True,
        )

        plotter.add_text(
            "RevolutionX Wind Tunnel Simulator",
            position="upper_right",
            font_size=8,
            color="#66aaff",
            font="courier",
            shadow=True,
        )

    def show_pressure(self, export_dir: Path = None):
        """Display pressure coefficient contour visualization."""
        print("\n[VIZ] Rendering pressure contour view...")
        self._build_pyvista_mesh()
        grid = self._build_velocity_grid()

        cfg = self.config.viz
        p = pv.Plotter(window_size=cfg.window_size, title="RevolutionX — Pressure Contours")
        p.set_background(cfg.background_color, top=cfg.background_color_top)

        # Pressure contours on mesh
        p.add_mesh(
            self.pv_mesh,
            scalars="Pressure Coefficient (Cp)",
            cmap=cfg.pressure_cmap,
            clim=cfg.cp_range,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Cp",
                "color": "white",
                "title_font_size": 14,
                "label_font_size": 11,
                "position_x": 0.85,
                "width": 0.08,
                "shadow": True,
            },
            smooth_shading=True,
            lighting=True,
        )

        self._add_ground_plane(p)
        self._add_wind_tunnel_walls(p)
        self._add_streamlines(p, grid)
        self._add_force_arrows(p)
        self._add_hud(p)

        p.camera_position = self._get_iso_camera()
        p.enable_anti_aliasing("ssaa")

        if export_dir:
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            self._take_screenshots(p, export_dir, "pressure")

        p.show()

    def show_velocity(self, export_dir: Path = None):
        """Display surface velocity visualization."""
        print("\n[VIZ] Rendering velocity view...")
        self._build_pyvista_mesh()
        grid = self._build_velocity_grid()

        cfg = self.config.viz
        p = pv.Plotter(window_size=cfg.window_size, title="RevolutionX — Velocity Field")
        p.set_background(cfg.background_color, top=cfg.background_color_top)

        v_max = self.results.velocity_ms * cfg.velocity_range_factor

        p.add_mesh(
            self.pv_mesh,
            scalars="Surface Velocity (m/s)",
            cmap=cfg.velocity_cmap,
            clim=[0, v_max],
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "V (m/s)",
                "color": "white",
                "title_font_size": 14,
                "label_font_size": 11,
                "position_x": 0.85,
                "width": 0.08,
                "shadow": True,
            },
            smooth_shading=True,
            lighting=True,
        )

        # Add velocity cut plane through center
        if grid is not None:
            try:
                center = self.pv_mesh.center
                axis_names = ['x', 'y', 'z']
                slice_normal = axis_names[self.results.lateral_axis]
                slice_plane = grid.slice(normal=slice_normal, origin=center)
                if slice_plane.n_points > 0 and "velocity_magnitude" in slice_plane.point_data:
                    p.add_mesh(
                        slice_plane,
                        scalars="velocity_magnitude",
                        cmap="turbo",
                        opacity=0.5,
                        show_scalar_bar=False,
                    )
            except Exception:
                pass

        self._add_ground_plane(p)
        self._add_wind_tunnel_walls(p)
        self._add_streamlines(p, grid)
        self._add_hud(p)

        p.camera_position = self._get_iso_camera()
        p.enable_anti_aliasing("ssaa")

        if export_dir:
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            self._take_screenshots(p, export_dir, "velocity")

        p.show()

    def show_combined(self, export_dir: Path = None):
        """Show multi-panel view: pressure + velocity + streamlines."""
        print("\n[VIZ] Rendering combined multi-view...")
        self._build_pyvista_mesh()
        grid = self._build_velocity_grid()
        cfg = self.config.viz

        p = pv.Plotter(
            shape=(1, 2),
            window_size=(cfg.window_size[0], cfg.window_size[1]),
            title="RevolutionX — Combined CFD Analysis",
        )

        # LEFT: Pressure contours
        p.subplot(0, 0)
        p.set_background(cfg.background_color, top=cfg.background_color_top)
        p.add_text("Pressure (Cp)", position="upper_edge", font_size=12, color="white")
        p.add_mesh(
            self.pv_mesh, scalars="Pressure Coefficient (Cp)",
            cmap=cfg.pressure_cmap, clim=cfg.cp_range,
            smooth_shading=True, show_scalar_bar=True,
            scalar_bar_args={"title": "Cp", "color": "white", "position_x": 0.85, "width": 0.06},
        )
        self._add_ground_plane(p)
        self._add_streamlines(p, grid)
        self._add_force_arrows(p)

        # RIGHT: Velocity
        p.subplot(0, 1)
        p.set_background(cfg.background_color, top=cfg.background_color_top)
        p.add_text("Surface Velocity", position="upper_edge", font_size=12, color="white")
        v_max = self.results.velocity_ms * cfg.velocity_range_factor
        p.add_mesh(
            self.pv_mesh, scalars="Surface Velocity (m/s)",
            cmap=cfg.velocity_cmap, clim=[0, v_max],
            smooth_shading=True, show_scalar_bar=True,
            scalar_bar_args={"title": "V (m/s)", "color": "white", "position_x": 0.85, "width": 0.06},
        )
        self._add_ground_plane(p)

        p.link_views()
        p.enable_anti_aliasing("ssaa")

        if export_dir:
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            self._take_screenshots(p, export_dir, "combined")

        p.show()

    def show_realtime(self, export_dir: Path = None):
        """Show real-time animated particles flowing around the car."""
        import time
        print("\n[VIZ] Rendering REAL-TIME animated particle flow...")
        self._build_pyvista_mesh()
        grid = self._build_velocity_grid()
        cfg = self.config.viz
        r = self.results
        fa = r.flow_axis
        la = r.lateral_axis
        va = r.vertical_axis

        p = pv.Plotter(window_size=cfg.window_size, title="RevolutionX — Real-Time Wind Tunnel")
        p.set_background(cfg.background_color, top=cfg.background_color_top)

        # Car mesh with pressure
        p.add_mesh(
            self.pv_mesh,
            scalars="Pressure Coefficient (Cp)",
            cmap=cfg.pressure_cmap, clim=cfg.cp_range,
            smooth_shading=True, show_scalar_bar=True,
            scalar_bar_args={"title": "Cp", "color": "white",
                             "title_font_size": 14, "label_font_size": 11,
                             "position_x": 0.85, "width": 0.08, "shadow": True},
        )
        self._add_ground_plane(p)
        self._add_wind_tunnel_walls(p)
        self._add_force_arrows(p)
        self._add_hud(p)

        p.add_text(
            "▶ REAL-TIME PARTICLE ANIMATION",
            position="lower_left", font_size=10,
            color="#00ff88", font="courier", shadow=True,
        )

        # Prepare particle system
        bounds = self.pv_mesh.bounds
        bmin = np.array([bounds[0], bounds[2], bounds[4]])
        bmax = np.array([bounds[1], bounds[3], bounds[5]])
        ext = bmax - bmin
        center = (bmin + bmax) / 2

        # Particle parameters - SimScale style: Lots of particles, streak effect
        n_particles = 2000
        v_ms = r.velocity_ms
        dt = ext[fa] * 0.02
        
        # Geometry for streaks
        streak_scale = ext[fa] * 0.04

        # Initialize particle positions
        def spawn_particles(n):
            pts = np.zeros((n, 3))
            # Spread widely upstream
            pts[:, fa] = bmin[fa] - ext[fa] * np.random.uniform(0.5, 2.5, n)
            pts[:, la] = center[la] + ext[la] * np.random.uniform(-1.2, 1.2, n)
            pts[:, va] = bmin[va] + ext[va] * np.random.uniform(-0.2, 1.6, n)
            return pts

        particles = spawn_particles(n_particles)
        velocities = np.zeros_like(particles)
        
        # Create polydata for particles
        particle_cloud = pv.PolyData(particles)
        particle_cloud["Velocity Magnitude"] = np.zeros(n_particles)
        particle_cloud["vectors"] = velocities # For orientation

        # Use arrow/streak glyphs for motion blur effect
        # We use a simple line or elongated sphere as the glyph
        glyph_source = pv.Line(pointa=(-0.5, 0, 0), pointb=(0.5, 0, 0))
        
        # Initial add
        actor = p.add_mesh(
            particle_cloud, 
            scalars="Velocity Magnitude",
            cmap="turbo",
            clim=[0, v_ms * 1.3],
            show_scalar_bar=True,
            scalar_bar_args={"title": "Flow Speed (m/s)", "color": "white", "width": 0.08, "position_x": 0.85},
            render_points_as_spheres=True, # Fallback
            point_size=6,
            opacity=0.8
        )

        body_center = center
        body_radii = ext / 2 * 1.05
        body_radii = np.clip(body_radii, 1e-6, None)

        # Set camera
        p.camera_position = self._get_iso_camera()
        p.enable_anti_aliasing("ssaa")
        
        # Pre-calculate random turbulence
        turb_la_base = np.random.normal(0, 1, (10000, n_particles)) * 0.0
        turb_idx = 0

        # Animation callback
        def update_particles(caller, event):
            nonlocal particles, velocities, turb_idx
            turb_idx += 1
            
            d = particles - body_center
            
            # vectorized distance check
            # Ellipsoid distance approximation
            # We normalize coordinate system to sphere
            d_norm = d / body_radii
            r_sq = np.sum(d_norm**2, axis=1)
            r_dist = np.sqrt(r_sq)
            
            # --- Vectorized Physics ---
            
            # 1. Base flow
            vel_target = np.zeros_like(particles)
            vel_target[:, fa] = -v_ms

            # 2. Obstacle Avoidance (Potential Flow approximation)
            # Influence factor: 1.0 at surface, decaying
            influence = np.exp(-1.5 * (r_dist - 0.5)**2)
            influence[r_dist > 3.0] = 0
            
            # Radial push vector (normal to ellipsoid surface)
            push_dir = d_norm / (r_dist[:, np.newaxis] + 1e-6)
            
            # Blend: flow diverts around
            # Tangential component preservation is hard in simple math, 
            # so we just add the push component and reduce axial flow
            
            # Slow down axial flow near nose/tail
            # High pressure zone at nose
            nose_region = (d_norm[:, fa] > 0.8) & (r_dist < 1.5)
            vel_target[nose_region, fa] *= 0.2
            
            # Add radial diversion
            divert_strength = v_ms * influence * 1.2
            vel_target += push_dir * divert_strength[:, np.newaxis]
            
            # Wake turbulence (downstream)
            # Downstream is where d_norm[fa] > 0 (since flow comes from +fa) — wait, flow is -fa
            # Correction: flow comes from +fa, moves to -fa.
            # So downstream is where coord is LESS than body (more negative)
            # Upstream = positive fa relative to center
            # Downstream = negative fa relative to center
            
            # My coordinate system assumption in solver was: Wind blows along -flow_axis.
            # So particles move in -data[fa].
            # Downstream is coordinates < body_center[fa]
            
            # Let's verify assumption:
            # Solver: vel[:, fa] = -v_ms. Correct.
            # So downstream is local coord < 0.
            
            is_downstream = (d_norm[:, fa] < -0.8) & (np.sqrt(d_norm[:, la]**2 + d_norm[:, va]**2) < 1.0)
            
            # Add turbulence in wake
            rows = particles.shape[0]
            if np.any(is_downstream):
                # Simple random jitter
                jitter = np.random.normal(0, v_ms*0.3, (np.sum(is_downstream), 3))
                vel_target[is_downstream] += jitter

            # 3. Apply
            velocities = vel_target
            particles += velocities * dt

            # 4. Respawn
            # Out bounds check (too far negative along fa)
            # bmin is min value. flow goes to -inf.
            # So if p[fa] < bmin[fa] - margin -> respawn
            out_mask = (particles[:, fa] < bmin[fa] - ext[fa]*1.0) | \
                       (particles[:, la] < center[la] - ext[la]*2.0) | \
                       (particles[:, la] > center[la] + ext[la]*2.0) | \
                       (particles[:, va] < bmin[va] - ext[va]*0.5) | \
                       (particles[:, va] > bmax[va] + ext[va]*2.0)
                       
            # Also respawn if inside body
            inside_mask = r_dist < 0.8
            respawn_mask = out_mask | inside_mask
            
            n_respawn = np.sum(respawn_mask)
            if n_respawn > 0:
                particles[respawn_mask] = spawn_particles(n_respawn)
                velocities[respawn_mask] = 0 # reset vel

            # 5. Update PolyData
            particle_cloud.points = particles
            
            # Update scalars for color
            speed = np.linalg.norm(velocities, axis=1)
            particle_cloud["Velocity Magnitude"] = speed

        if export_dir:
            export_dir = Path(export_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            self._take_screenshots(p, export_dir, "realtime")

        # Animation loop
        p.show(interactive_update=True)
        
        while True:
            try:
                update_particles(None, None)
                p.update()
            except AttributeError:
                break # Window closed
            except Exception:
                break

    def _take_screenshots(self, plotter_unused, export_dir: Path, prefix: str):
        """Capture screenshots using a dedicated off-screen plotter."""
        fa = self.results.flow_axis
        la = self.results.lateral_axis
        va = self.results.vertical_axis
        center = np.array(self.pv_mesh.center)
        bounds = self.pv_mesh.bounds
        ext = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
        max_ext = max(ext)
        cfg = self.config.viz

        # Generate axis-aware camera positions
        views = {}
        # Isometric
        iso = np.copy(center)
        iso[fa] -= max_ext * 1.5; iso[la] -= max_ext * 1.2; iso[va] += max_ext * 0.8
        views["iso"] = iso
        # Front (looking along flow axis)
        front = np.copy(center)
        front[fa] -= max_ext * 2.5; front[va] += max_ext * 0.3
        views["front"] = front
        # Side
        side = np.copy(center)
        side[la] -= max_ext * 2.5; side[va] += max_ext * 0.3
        views["side"] = side
        # Top
        top = np.copy(center)
        top[va] += max_ext * 3.0
        views["top"] = top

        up = np.zeros(3)
        up[va] = 1.0

        for view_name, pos in views.items():
            try:
                p = pv.Plotter(off_screen=True, window_size=cfg.window_size)
                p.set_background(cfg.background_color, top=cfg.background_color_top)
                p.add_mesh(
                    self.pv_mesh,
                    scalars="Pressure Coefficient (Cp)",
                    cmap=cfg.pressure_cmap, clim=cfg.cp_range,
                    smooth_shading=True, show_scalar_bar=True,
                    scalar_bar_args={"title": "Cp", "color": "white"},
                )
                self._add_ground_plane(p)
                p.camera_position = [pos.tolist(), center.tolist(), up.tolist()]

                path = export_dir / f"{prefix}_{view_name}.png"
                p.screenshot(str(path), transparent_background=False)
                self._screenshots.append(str(path))
                p.close()
                print(f"  ▸ Screenshot saved: {path.name}")
            except Exception as e:
                print(f"  ▸ Screenshot failed ({view_name}): {e}")

    def get_screenshots(self):
        return self._screenshots
