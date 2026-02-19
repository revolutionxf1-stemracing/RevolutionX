"""
╔══════════════════════════════════════════════════════════════════╗
║           WIND TUNNEL SIMULATOR — CONFIGURATION                 ║
║     All simulation parameters in one place                      ║
╚══════════════════════════════════════════════════════════════════╝

Modify these values to change the simulation behavior.
Units are SI unless otherwise noted.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class AirProperties:
    """Atmospheric / fluid properties."""
    density: float = 1.225          # kg/m³ (sea level, 15°C)
    dynamic_viscosity: float = 1.81e-5  # Pa·s
    temperature: float = 288.15     # K (15°C)
    pressure: float = 101325.0      # Pa
    gamma: float = 1.4              # Heat capacity ratio (air)
    R: float = 287.05               # Specific gas constant J/(kg·K)

    @property
    def kinematic_viscosity(self) -> float:
        """ν = μ / ρ  [m²/s]"""
        return self.dynamic_viscosity / self.density

    @property
    def speed_of_sound(self) -> float:
        """a = sqrt(γ·R·T) [m/s]"""
        return np.sqrt(self.gamma * self.R * self.temperature)


@dataclass
class SolverSettings:
    """Numerical solver parameters."""
    # Velocity sweep
    velocity_kmh: float = 100.0             # Default single velocity
    velocity_sweep: List[float] = field(     # Multi-speed analysis
        default_factory=lambda: [40, 60, 80, 100, 120, 150, 180, 200]
    )

    # Mesh processing
    max_faces_for_subdivide: int = 15000     # Subdivide if under this
    target_viz_faces: int = 80000            # Target faces for PyVista viz
    mesh_units: str = "mm"                   # STL file units (mm or m)

    # Streamlines
    streamline_density: int = 20             # Grid points per axis for seeds
    streamline_length: float = 3.0           # Relative to model length
    streamline_max_steps: int = 500          # Integration steps per line

    # Panel method
    turbulence_intensity: float = 0.05       # 5% freestream turbulence
    transition_reynolds: float = 5e5         # Laminar→turbulent transition Re

    # Material (for mass estimation)
    material_density_g_cm3: float = 0.15     # g/cm³ (PLA ~1.24, Foam ~0.1-0.3)


@dataclass
class VisualizationSettings:
    """Rendering and display settings."""
    # Theme
    background_color: str = "#1a1a2e"        # Dark navy
    background_color_top: str = "#16213e"    # Gradient top
    mesh_color: str = "#e0e0e0"              # Default mesh color
    mesh_opacity: float = 1.0

    # Colormaps
    pressure_cmap: str = "RdYlBu_r"         # Pressure: Red(high) → Blue(low)
    velocity_cmap: str = "coolwarm"          # Velocity: Blue(low) → Red(high)
    streamline_cmap: str = "plasma"          # Streamlines

    # Color ranges
    cp_range: Tuple[float, float] = (-1.5, 1.5)  # Cp range for colorbar
    velocity_range_factor: float = 1.5       # velocity range = [0, factor * V∞]

    # Streamlines
    streamline_opacity: float = 0.7
    streamline_line_width: float = 2.5
    streamline_tube_radius_factor: float = 0.003  # Relative to model size

    # Arrows / vectors
    arrow_scale: float = 0.15               # Force arrow scale
    arrow_color_drag: str = "#ff4444"
    arrow_color_lift: str = "#44ff44"

    # Ground plane
    show_ground_plane: bool = True
    ground_color: str = "#2a2a3e"
    ground_opacity: float = 0.3

    # Window
    window_size: Tuple[int, int] = (1600, 900)

    # Screenshots
    screenshot_dpi: int = 200
    screenshot_format: str = "png"


@dataclass
class ExportSettings:
    """Report and data export settings."""
    report_title: str = "Wind Tunnel Simulation Report"
    company_name: str = "RevolutionX Engineering"
    report_theme: str = "dark"               # "dark" or "light"
    export_csv: bool = True
    export_json: bool = True
    export_html: bool = True
    screenshot_views: List[str] = field(
        default_factory=lambda: ["front", "side", "top", "iso"]
    )


@dataclass
class STEMRacingRules:
    """F1 in Schools / STEM Racing compliance rules."""
    min_mass_g: float = 50.0
    max_mass_g: float = 65.0
    cartridge_diameter_mm: float = 19.0
    max_width_mm: float = 75.0
    max_height_mm: float = 120.0
    max_length_mm: float = 320.0
    max_wheelbase_mm: float = 280.0


@dataclass
class SimulationConfig:
    """Master configuration — combines all sub-configs."""
    air: AirProperties = field(default_factory=AirProperties)
    solver: SolverSettings = field(default_factory=SolverSettings)
    viz: VisualizationSettings = field(default_factory=VisualizationSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    stem_rules: STEMRacingRules = field(default_factory=STEMRacingRules)

    def summary(self) -> str:
        """Human-readable summary of current config."""
        return (
            f"═══ Simulation Configuration ═══\n"
            f"  Air:  ρ={self.air.density} kg/m³, T={self.air.temperature-273.15:.1f}°C, "
            f"μ={self.air.dynamic_viscosity:.2e} Pa·s\n"
            f"  Flow: V={self.solver.velocity_kmh} km/h, "
            f"TI={self.solver.turbulence_intensity*100:.1f}%\n"
            f"  Mesh: units={self.solver.mesh_units}, "
            f"material ρ={self.solver.material_density_g_cm3} g/cm³\n"
            f"  Viz:  {self.viz.window_size[0]}×{self.viz.window_size[1]}, "
            f"cmap={self.viz.pressure_cmap}\n"
        )
