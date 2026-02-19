"""
╔══════════════════════════════════════════════════════════════════╗
║       REPORT GENERATOR — Professional HTML + CSV + JSON        ║
║    SimScale-quality engineering report with embedded visuals    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import csv
import base64
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from jinja2 import Template


class ReportGenerator:
    """Generate professional engineering reports from simulation results."""

    def __init__(self, results, config, stl_filename: str, screenshots: list = None):
        self.results = results
        self.config = config
        self.stl_filename = stl_filename
        self.screenshots = screenshots or []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def export_all(self, output_dir: str):
        """Export HTML report, CSV data, and JSON config."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        base_name = Path(self.stl_filename).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        html_path = out / f"report_{base_name}_{ts}.html"
        csv_path = out / f"data_{base_name}_{ts}.csv"
        json_path = out / f"config_{base_name}_{ts}.json"
        sweep_csv = out / f"speed_sweep_{base_name}_{ts}.csv"

        if self.config.export.export_html:
            self._export_html(html_path)
        if self.config.export.export_csv:
            self._export_csv(csv_path)
            self._export_sweep_csv(sweep_csv)
        if self.config.export.export_json:
            self._export_json(json_path)

        print(f"\n{'═'*50}")
        print(f"  REPORTS EXPORTED")
        print(f"{'═'*50}")
        for p in [html_path, csv_path, sweep_csv, json_path]:
            if p.exists():
                print(f"  ▸ {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    def _embed_image(self, path: str) -> str:
        """Convert image to base64 data URI for embedding in HTML."""
        try:
            p = Path(path)
            if p.exists():
                data = p.read_bytes()
                b64 = base64.b64encode(data).decode()
                return f"data:image/png;base64,{b64}"
        except Exception:
            pass
        return ""

    def _export_csv(self, path: Path):
        """Export main results as CSV."""
        r = self.results
        rows = [
            ["Parameter", "Value", "Unit"],
            ["Model", self.stl_filename, ""],
            ["Date", self.timestamp, ""],
            ["Velocity", f"{r.velocity_ms:.2f}", "m/s"],
            ["Velocity", f"{r.velocity_ms * 3.6:.1f}", "km/h"],
            ["Reynolds Number", f"{r.reynolds_number:.2e}", ""],
            ["Mach Number", f"{r.mach_number:.4f}", ""],
            ["Dynamic Pressure", f"{r.dynamic_pressure:.2f}", "Pa"],
            ["Frontal Area", f"{r.frontal_area_m2:.6f}", "m²"],
            ["Frontal Area", f"{r.frontal_area_m2 * 1e4:.2f}", "cm²"],
            ["Planform Area", f"{r.planform_area_m2:.6f}", "m²"],
            ["", "", ""],
            ["FORCES", "", ""],
            ["Total Drag", f"{r.drag_N:.4f}", "N"],
            ["Pressure Drag", f"{r.drag_pressure_N:.4f}", "N"],
            ["Friction Drag", f"{r.drag_friction_N:.4f}", "N"],
            ["Lift", f"{r.lift_N:.4f}", "N"],
            ["Downforce", f"{-r.lift_N:.4f}", "N"],
            ["Side Force", f"{r.side_N:.4f}", "N"],
            ["", "", ""],
            ["COEFFICIENTS", "", ""],
            ["Cd (total)", f"{r.cd:.5f}", ""],
            ["Cd (pressure)", f"{r.cd_pressure:.5f}", ""],
            ["Cd (friction)", f"{r.cd_friction:.5f}", ""],
            ["Cl", f"{r.cl:.5f}", ""],
            ["L/D Ratio", f"{-r.lift_N / r.drag_N if r.drag_N else 0:.4f}", ""],
            ["", "", ""],
            ["MOMENTS", "", ""],
            ["Cm (pitch)", f"{r.cm_pitch:.5f}", ""],
            ["Cn (yaw)", f"{r.cn_yaw:.5f}", ""],
            ["Croll", f"{r.croll:.5f}", ""],
            ["", "", ""],
            ["PHYSICAL", "", ""],
            ["Mass", f"{r.mass_kg:.4f}", "kg"],
            ["Mass", f"{r.mass_kg * 1000:.1f}", "g"],
            ["Volume", f"{r.volume_m3 * 1e6:.2f}", "cm³"],
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _export_sweep_csv(self, path: Path):
        """Export speed sweep data as CSV."""
        sweep = self.results.speed_sweep
        if not sweep:
            return
        df = pd.DataFrame(sweep)
        df.to_csv(path, index=False)

    def _export_json(self, path: Path):
        """Export simulation configuration and results as JSON."""
        r = self.results
        data = {
            "simulation": {
                "model": self.stl_filename,
                "timestamp": self.timestamp,
                "solver": "Modified Newtonian Panel Method + BL",
                "version": "2.0.0",
            },
            "conditions": {
                "velocity_ms": r.velocity_ms,
                "velocity_kmh": r.velocity_ms * 3.6,
                "air_density_kg_m3": self.config.air.density,
                "air_temperature_K": self.config.air.temperature,
                "air_viscosity_Pa_s": self.config.air.dynamic_viscosity,
                "reynolds_number": r.reynolds_number,
                "mach_number": r.mach_number,
            },
            "results": {
                "forces": {
                    "drag_N": round(r.drag_N, 5),
                    "drag_pressure_N": round(r.drag_pressure_N, 5),
                    "drag_friction_N": round(r.drag_friction_N, 5),
                    "lift_N": round(r.lift_N, 5),
                    "downforce_N": round(-r.lift_N, 5),
                    "side_N": round(r.side_N, 5),
                },
                "coefficients": {
                    "cd": round(r.cd, 5),
                    "cd_pressure": round(r.cd_pressure, 5),
                    "cd_friction": round(r.cd_friction, 5),
                    "cl": round(r.cl, 5),
                    "ld_ratio": round(-r.lift_N / r.drag_N if r.drag_N else 0, 4),
                },
                "moments": {
                    "cm_pitch": round(r.cm_pitch, 5),
                    "cn_yaw": round(r.cn_yaw, 5),
                    "croll": round(r.croll, 5),
                },
                "physical": {
                    "mass_kg": round(r.mass_kg, 5),
                    "volume_m3": round(r.volume_m3, 8),
                    "frontal_area_m2": round(r.frontal_area_m2, 6),
                },
            },
            "speed_sweep": r.speed_sweep,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_html(self, path: Path):
        """Generate professional HTML report."""
        r = self.results
        ld = -r.lift_N / r.drag_N if r.drag_N else 0

        # Embed screenshots
        images_html = ""
        for ss in self.screenshots:
            data_uri = self._embed_image(ss)
            if data_uri:
                name = Path(ss).stem.replace("_", " ").title()
                images_html += f'<div class="img-card"><img src="{data_uri}" alt="{name}"><p>{name}</p></div>\n'

        # Speed sweep chart data
        sweep = r.speed_sweep
        sweep_rows = ""
        chart_labels = "[]"
        chart_cd = "[]"
        chart_drag = "[]"
        if sweep:
            chart_labels = json.dumps(sweep["speed_kmh"])
            chart_cd = json.dumps(sweep["cd"])
            chart_drag = json.dumps(sweep["drag_N"])
            for i in range(len(sweep["speed_kmh"])):
                sweep_rows += f"""<tr>
                    <td>{sweep['speed_kmh'][i]}</td>
                    <td>{sweep['cd'][i]}</td><td>{sweep['cl'][i]}</td>
                    <td>{sweep['drag_N'][i]}</td><td>{sweep['downforce_N'][i]}</td>
                    <td>{sweep['ld_ratio'][i]}</td><td>{sweep['power_W'][i]}</td>
                </tr>"""

        # STEM compliance
        mass_g = r.mass_kg * 1000
        rules = self.config.stem_rules
        mass_ok = rules.min_mass_g <= mass_g <= rules.max_mass_g
        cd_ok = r.cd < 0.45

        html = REPORT_TEMPLATE.replace("{{TITLE}}", self.config.export.report_title)
        html = html.replace("{{COMPANY}}", self.config.export.company_name)
        html = html.replace("{{MODEL}}", self.stl_filename)
        html = html.replace("{{TIMESTAMP}}", self.timestamp)
        html = html.replace("{{VELOCITY_KMH}}", f"{r.velocity_ms * 3.6:.0f}")
        html = html.replace("{{VELOCITY_MS}}", f"{r.velocity_ms:.1f}")
        html = html.replace("{{RE}}", f"{r.reynolds_number:.2e}")
        html = html.replace("{{MA}}", f"{r.mach_number:.4f}")
        html = html.replace("{{CD}}", f"{r.cd:.4f}")
        html = html.replace("{{CL}}", f"{r.cl:.4f}")
        html = html.replace("{{DRAG}}", f"{r.drag_N:.3f}")
        html = html.replace("{{DRAG_P}}", f"{r.drag_pressure_N:.3f}")
        html = html.replace("{{DRAG_F}}", f"{r.drag_friction_N:.3f}")
        html = html.replace("{{LIFT}}", f"{r.lift_N:.3f}")
        html = html.replace("{{DOWNFORCE}}", f"{-r.lift_N:.3f}")
        html = html.replace("{{SIDE}}", f"{r.side_N:.3f}")
        html = html.replace("{{LD}}", f"{ld:.3f}")
        html = html.replace("{{FRONTAL_AREA}}", f"{r.frontal_area_m2 * 1e4:.2f}")
        html = html.replace("{{MASS_G}}", f"{mass_g:.1f}")
        html = html.replace("{{CM}}", f"{r.cm_pitch:.4f}")
        html = html.replace("{{CN}}", f"{r.cn_yaw:.4f}")
        html = html.replace("{{CROLL}}", f"{r.croll:.4f}")
        html = html.replace("{{IMAGES}}", images_html)
        html = html.replace("{{SWEEP_ROWS}}", sweep_rows)
        html = html.replace("{{CHART_LABELS}}", chart_labels)
        html = html.replace("{{CHART_CD}}", chart_cd)
        html = html.replace("{{CHART_DRAG}}", chart_drag)
        html = html.replace("{{MASS_STATUS}}", "✅ PASS" if mass_ok else "❌ FAIL")
        html = html.replace("{{CD_STATUS}}", "✅ PASS" if cd_ok else "⚠️ HIGH")
        html = html.replace("{{MASS_CLASS}}", "pass" if mass_ok else "fail")
        html = html.replace("{{CD_CLASS}}", "pass" if cd_ok else "warn")

        path.write_text(html, encoding="utf-8")
        print(f"  ▸ HTML report: {path}")


# ─── HTML Template ─────────────────────────────────────────────
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}} — {{MODEL}}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--text2:#8b949e;
--accent:#58a6ff;--accent2:#3fb950;--red:#f85149;--orange:#d29922;--glass:rgba(22,27,34,0.85)}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
line-height:1.6;padding:0}
.header{background:linear-gradient(135deg,#0d1117 0%,#161b22 50%,#1a2332 100%);
padding:40px;border-bottom:1px solid var(--border);text-align:center}
.header h1{font-size:28px;color:var(--accent);margin-bottom:8px;letter-spacing:1px}
.header .subtitle{color:var(--text2);font-size:14px}
.header .company{color:var(--accent2);font-size:12px;text-transform:uppercase;letter-spacing:3px;margin-bottom:12px}
.container{max-width:1200px;margin:0 auto;padding:30px 20px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:24px 0}
.kpi{background:var(--glass);backdrop-filter:blur(10px);border:1px solid var(--border);
border-radius:12px;padding:20px;text-align:center;transition:transform .2s}
.kpi:hover{transform:translateY(-2px)}
.kpi .value{font-size:32px;font-weight:700;color:var(--accent)}
.kpi .label{font-size:12px;color:var(--text2);text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.kpi .unit{font-size:14px;color:var(--text2)}
h2{font-size:20px;color:var(--accent);margin:32px 0 16px;padding-bottom:8px;border-bottom:1px solid var(--border)}
h2::before{content:'▸ ';color:var(--accent2)}
table{width:100%;border-collapse:collapse;margin:16px 0;background:var(--card);border-radius:8px;overflow:hidden}
th{background:rgba(88,166,255,0.1);color:var(--accent);padding:12px 16px;text-align:left;font-size:13px;
text-transform:uppercase;letter-spacing:0.5px}
td{padding:10px 16px;border-top:1px solid var(--border);font-size:14px}
tr:hover td{background:rgba(88,166,255,0.04)}
.img-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px;margin:16px 0}
.img-card{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.img-card img{width:100%;height:auto;display:block}
.img-card p{padding:10px;text-align:center;color:var(--text2);font-size:13px}
.chart-container{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin:16px 0}
.badge{display:inline-block;padding:4px 10px;border-radius:20px;font-size:12px;font-weight:600}
.pass{background:rgba(63,185,80,0.15);color:var(--accent2)}
.fail{background:rgba(248,81,73,0.15);color:var(--red)}
.warn{background:rgba(210,153,34,0.15);color:var(--orange)}
.footer{text-align:center;padding:30px;color:var(--text2);font-size:12px;border-top:1px solid var(--border);margin-top:40px}
@media(max-width:768px){.kpi-grid{grid-template-columns:repeat(2,1fr)}.img-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
<div class="company">{{COMPANY}}</div>
<h1>{{TITLE}}</h1>
<div class="subtitle">Model: {{MODEL}} &nbsp;|&nbsp; Generated: {{TIMESTAMP}}</div>
</div>
<div class="container">

<h2>Key Performance Indicators</h2>
<div class="kpi-grid">
<div class="kpi"><div class="value">{{CD}}</div><div class="label">Drag Coefficient</div></div>
<div class="kpi"><div class="value">{{CL}}</div><div class="label">Lift Coefficient</div></div>
<div class="kpi"><div class="value">{{DRAG}}</div><div class="label">Drag Force</div><div class="unit">N</div></div>
<div class="kpi"><div class="value">{{DOWNFORCE}}</div><div class="label">Downforce</div><div class="unit">N</div></div>
<div class="kpi"><div class="value">{{LD}}</div><div class="label">L/D Ratio</div></div>
<div class="kpi"><div class="value">{{MASS_G}}</div><div class="label">Mass</div><div class="unit">g</div></div>
</div>

<h2>Flow Conditions</h2>
<table>
<tr><th>Parameter</th><th>Value</th><th>Unit</th></tr>
<tr><td>Freestream Velocity</td><td>{{VELOCITY_KMH}} ({{VELOCITY_MS}})</td><td>km/h (m/s)</td></tr>
<tr><td>Reynolds Number</td><td>{{RE}}</td><td>—</td></tr>
<tr><td>Mach Number</td><td>{{MA}}</td><td>—</td></tr>
<tr><td>Frontal Area</td><td>{{FRONTAL_AREA}}</td><td>cm²</td></tr>
</table>

<h2>Force & Coefficient Analysis</h2>
<table>
<tr><th>Component</th><th>Force (N)</th><th>Coefficient</th></tr>
<tr><td>Total Drag</td><td>{{DRAG}}</td><td>Cd = {{CD}}</td></tr>
<tr><td>↳ Pressure Drag</td><td>{{DRAG_P}}</td><td>—</td></tr>
<tr><td>↳ Friction Drag</td><td>{{DRAG_F}}</td><td>—</td></tr>
<tr><td>Lift / Downforce</td><td>{{LIFT}} / {{DOWNFORCE}}</td><td>Cl = {{CL}}</td></tr>
<tr><td>Side Force</td><td>{{SIDE}}</td><td>—</td></tr>
</table>

<h2>Moment Coefficients</h2>
<table>
<tr><th>Moment</th><th>Coefficient</th></tr>
<tr><td>Pitch (Cm)</td><td>{{CM}}</td></tr>
<tr><td>Yaw (Cn)</td><td>{{CN}}</td></tr>
<tr><td>Roll (Croll)</td><td>{{CROLL}}</td></tr>
</table>

<h2>CFD Visualization</h2>
<div class="img-grid">{{IMAGES}}</div>

<h2>Speed Sweep Analysis</h2>
<div class="chart-container"><canvas id="sweepChart" height="100"></canvas></div>
<table>
<tr><th>Speed (km/h)</th><th>Cd</th><th>Cl</th><th>Drag (N)</th><th>Downforce (N)</th><th>L/D</th><th>Power (W)</th></tr>
{{SWEEP_ROWS}}
</table>

<h2>STEM Racing Compliance</h2>
<table>
<tr><th>Check</th><th>Status</th></tr>
<tr><td>Mass ({{MASS_G}} g) — Rule: 50-65g</td><td><span class="badge {{MASS_CLASS}}">{{MASS_STATUS}}</span></td></tr>
<tr><td>Cd ({{CD}}) — Target: &lt; 0.45</td><td><span class="badge {{CD_CLASS}}">{{CD_STATUS}}</span></td></tr>
<tr><td>Cartridge Hole (19mm)</td><td><span class="badge warn">⚠️ CHECK MANUALLY</span></td></tr>
</table>

</div>
<div class="footer">
Generated by {{COMPANY}} Wind Tunnel Simulator v2.0 &nbsp;|&nbsp; Solver: Modified Newtonian Panel Method + BL
</div>

<script>
const ctx=document.getElementById('sweepChart');
if(ctx){new Chart(ctx,{type:'line',data:{labels:{{CHART_LABELS}},datasets:[
{label:'Cd',data:{{CHART_CD}},borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.1)',yAxisID:'y',tension:0.3,fill:true},
{label:'Drag (N)',data:{{CHART_DRAG}},borderColor:'#f85149',backgroundColor:'rgba(248,81,73,0.1)',yAxisID:'y1',tension:0.3,fill:true}
]},options:{responsive:true,plugins:{legend:{labels:{color:'#e6edf3'}}},
scales:{x:{title:{display:true,text:'Speed (km/h)',color:'#8b949e'},ticks:{color:'#8b949e'},grid:{color:'#30363d'}},
y:{title:{display:true,text:'Cd',color:'#58a6ff'},ticks:{color:'#58a6ff'},grid:{color:'#30363d'},position:'left'},
y1:{title:{display:true,text:'Drag (N)',color:'#f85149'},ticks:{color:'#f85149'},grid:{drawOnChartArea:false},position:'right'}
}}});}
</script>
</body></html>"""
