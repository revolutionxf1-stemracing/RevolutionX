"""
╔══════════════════════════════════════════════════════════════════════╗
║       REVOLUTIONX WIND TUNNEL SIMULATOR v2.0                        ║
║    Professional CFD Analysis — STL Upload → Simulation → Report     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Usage:                                                              ║
║    GUI mode:     py -3.14 wind_tunnel_simulator.py                   ║
║    CLI mode:     py -3.14 wind_tunnel_simulator.py model.stl         ║
║    With export:  py -3.14 wind_tunnel_simulator.py model.stl --export║
║    No GUI viz:   py -3.14 wind_tunnel_simulator.py model.stl --no-gui║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import argparse
from pathlib import Path

import trimesh
import numpy as np

# Local modules
from simulation_config import SimulationConfig
from aero_solver import AeroSolver
from visualizer import WindTunnelVisualizer
from report_generator import ReportGenerator


def print_banner():
    print("\n" + "═" * 62)
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║     REVOLUTIONX WIND TUNNEL SIMULATOR  v2.0         ║")
    print("  ║     Professional Aerodynamic Analysis                ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print("═" * 62)


def select_stl_file() -> str:
    """Open a file dialog to select an STL file using tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        file_path = filedialog.askopenfilename(
            title="RevolutionX — Select STL Model",
            filetypes=[
                ("STL Files", "*.stl"),
                ("All Mesh Files", "*.stl *.obj *.ply *.off"),
                ("All Files", "*.*"),
            ],
            initialdir=os.getcwd(),
        )
        root.destroy()
        
        if file_path:
            return file_path
        else:
            print("\n  [!] No file selected. Exiting.")
            sys.exit(0)
    except ImportError:
        print("\n  [!] tkinter not available. Please provide STL path as argument.")
        print("  Usage: py -3.14 wind_tunnel_simulator.py <path_to.stl>")
        sys.exit(1)


def select_velocity() -> float:
    """Ask user for velocity via simple dialog or terminal."""
    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        velocity = simpledialog.askfloat(
            "RevolutionX — Flow Velocity",
            "Enter freestream velocity (km/h):\n\n"
            "Common values:\n"
            "  • City car:     50 km/h\n"
            "  • Highway:     120 km/h\n"
            "  • F1 Schools:  80 km/h\n"
            "  • Race car:    200 km/h\n",
            initialvalue=100.0,
            minvalue=1.0,
            maxvalue=500.0,
        )
        root.destroy()

        return velocity if velocity else 100.0
    except Exception:
        return 100.0


def select_viz_mode() -> str:
    """Let user select visualization mode."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.title("RevolutionX — Visualization Mode")
        root.configure(bg="#1a1a2e")
        root.geometry("400x420")
        root.attributes("-topmost", True)

        selected = tk.StringVar(value="pressure")

        tk.Label(
            root, text="Select Visualization Mode",
            bg="#1a1a2e", fg="#58a6ff", font=("Segoe UI", 14, "bold")
        ).pack(pady=15)

        modes = [
            ("pressure", "🔴 Pressure Contours (Cp)", "Surface pressure distribution with streamlines"),
            ("velocity", "🔵 Velocity Field", "Surface velocity with cut planes"),
            ("combined", "🟣 Combined Multi-View", "Side-by-side pressure + velocity"),
            ("realtime", "🟢 Real-Time Particles", "Live animated particles flowing around the car"),
        ]

        for val, label, desc in modes:
            frame = tk.Frame(root, bg="#161b22", bd=1, relief="solid")
            frame.pack(fill="x", padx=20, pady=4)
            tk.Radiobutton(
                frame, text=label, variable=selected, value=val,
                bg="#161b22", fg="#e6edf3", selectcolor="#0d1117",
                font=("Segoe UI", 11), anchor="w",
                activebackground="#161b22", activeforeground="#58a6ff",
            ).pack(anchor="w", padx=10, pady=2)
            tk.Label(
                frame, text=desc, bg="#161b22", fg="#8b949e",
                font=("Segoe UI", 9), anchor="w",
            ).pack(anchor="w", padx=30, pady=(0, 5))

        def confirm():
            root.quit()
            root.destroy()

        tk.Button(
            root, text="▶  Run Simulation", command=confirm,
            bg="#238636", fg="white", font=("Segoe UI", 12, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2",
        ).pack(pady=15)

        root.mainloop()
        return selected.get()
    except Exception:
        return "pressure"


def run_simulation(stl_path: str, config: SimulationConfig, velocity: float = None,
                   viz_mode: str = "pressure", show_gui: bool = True,
                   export: bool = False):
    """Main simulation pipeline."""
    
    if velocity:
        config.solver.velocity_kmh = velocity

    # ─── 1. Load STL ─────────────────────────────────────────
    print(f"\n  Loading: {Path(stl_path).name}")
    t0 = time.time()
    mesh = trimesh.load(stl_path, force='mesh')
    print(f"  ▸ Loaded: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    print(f"  ▸ Watertight: {'✅ Yes' if mesh.is_watertight else '⚠️ No (mass estimate may be inaccurate)'}")
    print(f"  ▸ Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}")
    print(f"  ▸ Load time: {time.time()-t0:.1f}s")

    # ─── 2. Run Solver ───────────────────────────────────────
    print(f"\n{config.summary()}")
    solver = AeroSolver(mesh, config)
    t1 = time.time()
    results = solver.run_full_analysis(config.solver.velocity_kmh)
    print(f"\n  ▸ Solver time: {time.time()-t1:.1f}s")

    # ─── 3. Visualization ───────────────────────────────────
    export_dir = None
    if export:
        export_dir = Path(stl_path).parent / "reports"
        export_dir.mkdir(exist_ok=True)

    visualizer = WindTunnelVisualizer(mesh, results, config)

    if show_gui:
        print(f"\n  Launching visualization: {viz_mode}")
        if viz_mode == "pressure":
            visualizer.show_pressure(export_dir=export_dir)
        elif viz_mode == "velocity":
            visualizer.show_velocity(export_dir=export_dir)
        elif viz_mode == "combined":
            visualizer.show_combined(export_dir=export_dir)
        elif viz_mode == "realtime":
            visualizer.show_realtime(export_dir=export_dir)

    # ─── 4. Export Report ────────────────────────────────────
    if export:
        screenshots = visualizer.get_screenshots()
        reporter = ReportGenerator(
            results, config,
            stl_filename=Path(stl_path).name,
            screenshots=screenshots,
        )
        reporter.export_all(str(export_dir))

    print(f"\n{'═'*62}")
    print(f"  SIMULATION COMPLETE — Total: {time.time()-t0:.1f}s")
    print(f"{'═'*62}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="RevolutionX Wind Tunnel Simulator v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stl_file", nargs="?", default=None, help="Path to STL file")
    parser.add_argument("--velocity", "-v", type=float, default=None, help="Velocity in km/h")
    parser.add_argument("--export", "-e", action="store_true", help="Export report + data")
    parser.add_argument("--no-gui", action="store_true", help="Skip 3D visualization")
    parser.add_argument("--mode", "-m", choices=["pressure", "velocity", "combined", "realtime"],
                        default=None, help="Visualization mode")
    parser.add_argument("--density", "-d", type=float, default=None,
                        help="Material density in g/cm³")

    args = parser.parse_args()

    print_banner()

    config = SimulationConfig()

    # STL file selection
    stl_path = args.stl_file
    if not stl_path:
        stl_path = select_stl_file()

    if not os.path.exists(stl_path):
        print(f"\n  [ERROR] File not found: {stl_path}")
        sys.exit(1)

    # Velocity
    velocity = args.velocity
    if velocity is None and not args.no_gui:
        velocity = select_velocity()
    elif velocity is None:
        velocity = config.solver.velocity_kmh

    # Material density
    if args.density:
        config.solver.material_density_g_cm3 = args.density

    # Viz mode
    viz_mode = args.mode
    if viz_mode is None and not args.no_gui:
        viz_mode = select_viz_mode()
    elif viz_mode is None:
        viz_mode = "pressure"

    # Run
    run_simulation(
        stl_path=stl_path,
        config=config,
        velocity=velocity,
        viz_mode=viz_mode,
        show_gui=not args.no_gui,
        export=args.export,
    )


if __name__ == "__main__":
    main()
