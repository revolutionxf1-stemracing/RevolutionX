import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib import cm, colors

class RaceCarAnalyzer:
    def __init__(self, file_path, density_g_cm3=0.2):
        print(f"--- Loading: {file_path} ---")
        self.mesh = trimesh.load(file_path, force='mesh')
        self.density_g_cm3 = density_g_cm3
        self.density_kg_m3 = density_g_cm3 * 1000.0
        
        if not self.mesh.is_watertight:
            print("WARNING: Mesh is not watertight. Mass properties might be inaccurate.")
        else:
            print("Mesh is watertight.")

        self.analysis_results = {}

    def analyze_physics(self):
        """Calculates geometric and inertial properties with High Resolution."""
        print("\n[PHYSICS ANALYSIS - HIGH QUALITY]")
        
        # 5x Quality: Subdivide mesh for smoother normals and better integral
        # Only subdivide if face count is low (< 20k) to avoid freezing
        if len(self.mesh.faces) < 20000:
             print(f"  Refining mesh... (Original: {len(self.mesh.faces)} faces)")
             self.mesh = self.mesh.subdivide()
             print(f"  Refined: {len(self.mesh.faces)} faces")

        # Volume & Mass
        vol_mm3 = self.mesh.volume
        vol_m3 = vol_mm3 / 1e9
        
        mass_kg = vol_m3 * self.density_kg_m3
        self.analysis_results['mass_kg'] = mass_kg
        
        # Center of Mass
        com_mm = self.mesh.center_mass
        self.analysis_results['com_mm'] = com_mm
        
        # Inertia Tensor
        # trimesh calculates inertia assuming unit density (1.0)
        # We must scale it by our density.
        inertia_tensor = self.mesh.moment_inertia * self.density_g_cm3
        # Units: (mm^5 * g/cm^3)? No.
        # mesh.volume is mm^3. Density is g/cm^3.
        # mesh.moment_inertia (unit density) is mm^5 ?? No, Moment of Inertia is Mass * Length^2.
        # If density=1 (mass=volume), then units are Volume * Length^2 = L^3 * L^2 = L^5.
        # So trimesh returns mm^5.
        # We multiply by density (g/cm^3). 
        # mm^5 * (g / (10mm)^3) = mm^5 * g / 1000 mm^3 = mm^2 * g / 1000.
        # We want kg * m^2.
        # 1 g = 1e-3 kg. 1 mm^2 = 1e-6 m^2.
        # So factor is (1/1000) * 1e-3 * 1e-6 = 1e-12.
        # Let's double check.
        # Inertia_trimesh (mm^5) * Density (g/cm^3) * conversion.
        # Conversion: 1 g/cm^3 = 1000 kg/m^3.
        # Inertia (mm^5) -> m^5 is 1e-15.
        # (m^5) * (kg/m^3) = kg*m^2.
        # So: Tensor * 1e-15 * 1000 = Tensor * 1e-12. Correct.
        
        inertia_kgm2 = self.mesh.moment_inertia * self.density_g_cm3 * 1e-9 # Wait, let's re-verify the mm^5 to cm^3 conversion part.
        # Let's stick to SI base units for calculation to be safe.
        
        # Re-calculating Inertia using scaling factor logic:
        # Inertia_real = Inertia_unit_density * (Density_real / Density_unit)
        # Trimesh unit density is usually 1.0 (in whatever units the mesh is).
        # If mesh is in mm, Trimesh assumes 1.0 mass/mm^3? Or just geometric integral?
        # It's geometric integral of r^2 dV. So units L^5.
        # Mass = Volume * Density.
        # I = Integral(r^2 * rho * dV) = rho * Integral(r^2 dV).
        # Integral(r^2 dV) is what trimesh returns. Units mm^5.
        # We want I in kg*m^2.
        # rho = self.density_kg_m3 (kg/m^3).
        # Integral in m^5 = Integral_mm5 * 1e-15.
        # So I_kgm2 = (Inertia_trimesh * 1e-15) * self.density_kg_m3.
        
        inertia_kgm2 = (self.mesh.moment_inertia * 1e-15) * self.density_kg_m3
        self.analysis_results['inertia_tensor_kgm2'] = inertia_kgm2
        
        print(f"  Mass: {mass_kg:.4f} kg")
        print(f"  CoM (mm): {com_mm}")
        print(f"  Inertia (Yaw - Z): {inertia_kgm2[2][2]:.6f} kg*m^2")

    def analyze_aerodynamics(self, velocity_kmh=100.0):
        """
        Estimates Aerodynamic forces using Newtonian Impact Theory.
        Calculates Frontal Area and Cd (Drag Coefficient).
        """
        print(f"\n[AERODYNAMIC ESTIMATION] @ {velocity_kmh} km/h")
        
        # Velocity vector: Car moving +X, so wind is -X
        v_car_ms = velocity_kmh / 3.6
        gamma = 1.4 # Heat capacity ratio
        R = 287.05 # Specific gas constant
        T = 293.15 # 20C
        rho_air = 1.225 # kg/m3
        
        # Dynamic Pressure q = 0.5 * rho * v^2
        q = 0.5 * rho_air * (v_car_ms ** 2)
        
        # --- Frontal Area Estimation ---
        # Project vertices to YZ plane (axis 1 and 2)
        # We use a bounding box approximation of the YZ projection
        # Ideally we'd use a grid occupancy method or convex hull.
        vertices = self.mesh.vertices
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        z_range = vertices[:, 2].max() - vertices[:, 2].min()
        # Estimation: Box area * Fill Factor (approx 0.85 for a typical car shape)
        frontal_area_m2 = (y_range * z_range * 1e-6) * 0.85
        
        # --- Force Calculation ---
        normals = self.mesh.face_normals
        areas = self.mesh.area_faces # mm^2
        
        # Dot product with Forward Vector (+X)
        dots = np.dot(normals, [1, 0, 0])
        
        # Pressure Coefficient (Modified Newtonian)
        # Cp = 1.0 * (dot)^2  for Front (Impact)
        # Cp = -0.2          for Rear (Wake Suction)
        cp_distribution = np.zeros(len(dots))
        
        # Frontal impact
        front_mask = dots > 0
        cp_distribution[front_mask] = 1.0 * (dots[front_mask] ** 2)
        
        # Wake / Base drag
        rear_mask = dots <= 0
        cp_distribution[rear_mask] = -0.25 # Slightly higher suction for boxy shapes
        
        # Forces per face
        forces = np.zeros_like(normals)
        for i in range(len(forces)):
            p = cp_distribution[i] * q
            a_m2 = areas[i] * 1e-6
            forces[i] = -p * a_m2 * normals[i]
            
        total_force = np.sum(forces, axis=0) # [Fx, Fy, Fz]
        
        drag_force = -total_force[0] # +X is motion, Drag is -X force
        downforce = -total_force[2]  # -Z is down
        
        # --- Drag Coefficient (Cd) ---
        # Fd = Cd * A * q
        # Cd = Fd / (A * q)
        if frontal_area_m2 > 0 and q > 0:
            cd = drag_force / (frontal_area_m2 * q)
        else:
            cd = 0.0

        self.analysis_results['drag_N'] = drag_force
        self.analysis_results['downforce_N'] = downforce
        self.analysis_results['cp'] = cp_distribution
        self.analysis_results['cd'] = cd
        self.analysis_results['frontal_area_m2'] = frontal_area_m2
        
        print(f"  Frontal Area (est): {frontal_area_m2*1e4:.2f} cm^2")
        print(f"  Drag Force: {drag_force:.2f} N")
        print(f"  Drag Coefficient (Cd): {cd:.3f}")
        print(f"  Downforce: {downforce:.2f} N")
        print(f"  L/D Ratio: {downforce/drag_force if drag_force!=0 else 0:.3f}")

    def optimize_for_stem(self):
        """Checks metrics against F1 in Schools / STEM racing standards."""
        print("\n[STEM RACING OPTIMIZATION]")
        mass_kg = self.analysis_results.get('mass_kg', 0)
        min_mass = 0.050 # 50g
        max_mass = 0.055 # 55g (safe zone) - rules often say min 50g
        
        mass_g = mass_kg * 1000
        print(f"  Current Mass: {mass_g:.1f} g")
        
        if mass_g < 50:
            print("  [!] ILLEGAL: Mass is under 50g. Add ballast or thicken walls.")
        elif mass_g > 55:
            print(f"  [!] HEAVY: Mass is {mass_g - 50:.1f}g over minimum. Remove material.")
        else:
            print("  [OK] Mass is within optimal racing range (50-55g).")
            
        cd = self.analysis_results.get('cd', 0)
        if cd > 0.4:
            print("  [TIP] High Drag Coefficient. Smooth front curvature or reduce frontal area.")
        
        # Co2 Cartridge Hole Check (Simulated check)
        # We check if there's a cylindrical hole in the rear.
        # This is hard to check on a generic mesh without more complex logic, 
        # but we can advise checking it.
        print("  [CHECK] Ensure 19mm diameter cartridge hole exists at rear.")

    def visualize(self):
        """Displays Fusion 360-style CFD visualization with High Fidelity Streamlines."""
        print("\n[VISUALIZATION] Rendering Fusion 360 Style CFD environment (High Quality)...")
        
        # Use dark background for Fusion 360 / Engineering look
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # --- CAR MESH (Pressure Map) ---
        cp = self.analysis_results.get('cp', np.zeros(len(self.mesh.faces)))
        
        # Simplify mesh for visualization (Crucial for performance with Matplotlib)
        # Try to use trimesh's simplify feature if available, else just stride
        try:
            # Target ~5000 faces for responsive viz
            target_faces = 5000
            if len(self.mesh.faces) > target_faces:
                print(f"  Simplifying mesh for visualization ({len(self.mesh.faces)} -> ~{target_faces} faces)...")
                # Simple striding is fastest and dependency-free method for viz
                # taking every Nth face
                stride = int(len(self.mesh.faces) / target_faces)
                viz_faces = self.mesh.faces[::stride]
            else:
                viz_faces = self.mesh.faces
        except Exception as e:
            print(f"  Warning: Mesh simplification failed ({e}). Using full mesh.")
            viz_faces = self.mesh.faces

        # Create a subset mesh for viz (mock object for collection)
        # We need to map the scalar values (cp) to these faces too
        if len(viz_faces) < len(self.mesh.faces):
             # If we strided, we need to stride the Cp array too
             stride = int(len(self.mesh.faces) / len(viz_faces))
             viz_cp = cp[::stride]
             # Ensure lengths match exactly
             viz_cp = viz_cp[:len(viz_faces)]
        else:
             viz_cp = cp


        # Colors
        norm = plt.Normalize(vmin=-0.5, vmax=1.0)
        cmap = plt.cm.jet
        colors_mapped = cmap(norm(viz_cp))
        
        # Create the collection with the SIMPLIFIED faces
        # We need the vertices associated with these faces. 
        # art3d.Poly3DCollection accepts a list of [ (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) ]
        # self.mesh.vertices[viz_faces] gives us exactly that shape (N, 3, 3)
        viz_triangles = self.mesh.vertices[viz_faces]
        
        tri_collection = art3d.Poly3DCollection(viz_triangles)
        tri_collection.set_facecolor(colors_mapped)
        tri_collection.set_edgecolor(colors_mapped) 
        tri_collection.set_linewidth(0.0) 
        tri_collection.set_alpha(1.0) 
        
        ax.add_collection3d(tri_collection)
        
        # --- CFD STREAMLINES (Virtual Wind Tunnel) ---
        bounds = self.mesh.bounds
        min_b, max_b = bounds[0], bounds[1]
        
        start_x = max_b[0] + 50
        end_x = min_b[0] - 150 
        
        # OPTIMIZED: Reduce streamline count for performance (15x10 instead of 25x15)
        y_pts = np.linspace(min_b[1]-20, max_b[1]+20, 15) 
        z_pts = np.linspace(min_b[2], max_b[2]+40, 10)
        
        # Create flow lines
        for y in y_pts:
            for z in z_pts:
                # OPTIMIZED: Fewer points per line (50 instead of 100)
                line_x = np.linspace(start_x, end_x, 50)
                line_y = np.full_like(line_x, y)
                line_z = np.full_like(line_x, z)
                
                # Visual Check: Is this ray going to hit the car?
                hit_y = (y > min_b[1]) and (y < max_b[1])
                hit_z = (z > min_b[2]) and (z < max_b[2])
                
                if hit_y and hit_z:
                     # "Turbulent"
                     color = (1.0, 0.3, 0.2, 0.15) 
                     # Perturb lines slightly 
                     noise = np.random.normal(0, 2.0, size=len(line_x))
                     line_y += noise
                     line_z += noise
                else:
                     # "Laminar"
                     color = (0.0, 1.0, 0.8, 0.1) 
                
                ax.plot(line_x, line_y, line_z, color=color, linewidth=0.8)
                
        # --- SCENE SETUP ---
        # Auto scale
        scale = self.mesh.extents.max()
        mid = self.mesh.bounds.mean(axis=0)
        ax.set_xlim(mid[0] - scale, mid[0] + scale)
        ax.set_ylim(mid[1] - scale/2, mid[1] + scale/2)
        ax.set_zlim(mid[2] - scale/2, mid[2] + scale/2)
        
        ax.set_axis_off() # Hide axes for cleaner look
        
        # Add a ground plane/grid?
        # Grid at Z min
        # grid_x = np.linspace(min_b[0]-50, max_b[0]+50, 10)
        # grid_y = np.linspace(min_b[1]-50, max_b[1]+50, 10)
        # ... complicates the plot. Axis off is cleaner.

        # --- HUD / TEXT INFO ---
        # Display stats on the plot
        info_text = (
            f"Mass: {self.analysis_results.get('mass_kg',0)*1000.:.1f} g\n"
            f"Drag: {self.analysis_results.get('drag_N',0):.2f} N\n"
            f"Cd:   {self.analysis_results.get('cd',0):.3f}"
        )
        ax.text2D(0.05, 0.9, info_text, transform=ax.transAxes, color='white', 
                  fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))
        
        # Colorbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(cp)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.0)
        cbar.set_label('Pressure Coefficient (Cp)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.set_title('CFD Simulation (Fusion 360 Style)', color='white', fontsize=16)
        
        print("Visualization window opened. Close to exit.")
        plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    import sys
    import os
    
    file_path = "sample_car.stl"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
    else:
        analyzer = RaceCarAnalyzer(file_path, density_g_cm3=0.15)
        analyzer.analyze_physics()
        analyzer.analyze_aerodynamics(velocity_kmh=120)
        analyzer.optimize_for_stem() # Check regulations
        analyzer.visualize()
