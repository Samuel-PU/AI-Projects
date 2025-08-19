# --- clean_figures.py ---
# Generates two clean figures:
#   figs/system_overview_clean.[pdf|png]
#   figs/threat_model_matrix.[pdf|png]

import matplotlib
matplotlib.use("Agg")  # safe on Windows/headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from pathlib import Path

# ---------- paths ----------
try:
    BASE = Path(__file__).resolve().parent
except NameError:
    BASE = Path.cwd()
OUT = BASE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

def add_box(ax, xy, wh, text, fontsize=10):
    x, y = xy; w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                linewidth=1.2, fill=False))
    ax.text(x + w/2, y + h*0.6, text, ha="center", va="center", fontsize=fontsize)

def add_arrow(ax, p0, p1, label=None, dashed=False, fs=9):
    arr = FancyArrowPatch(p0, p1,
                          arrowstyle=ArrowStyle("->", head_length=6, head_width=3),
                          linewidth=1.2,
                          linestyle="--" if dashed else "-",
                          mutation_scale=8)
    ax.add_patch(arr)
    if label:
        ax.text((p0[0]+p1[0])/2, (p0[1]+p1[1])/2 + 0.02, label,
                ha="center", va="bottom", fontsize=fs)

# ---------- Figure 1: System overview ----------
fig, ax = plt.subplots(figsize=(7.2, 4.3))   # one-column friendly
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

add_box(ax, (0.05, 0.65), (0.22, 0.2), "Client\nC, policy")
add_box(ax, (0.38, 0.45), (0.34, 0.4),
        "Orchestrator\n• NADGO pads\n• PF scheduler\n• CASQUE routing\n• ΔIₜ monitor + kill")
add_box(ax, (0.74, 0.58), (0.21, 0.22), "QPU Farm\nSub-circuits")
add_box(ax, (0.38, 0.15), (0.34, 0.18), "Audit Log")

add_arrow(ax, (0.27, 0.75), (0.38, 0.75), "submit C, policy")
add_arrow(ax, (0.72, 0.66), (0.74, 0.66), "execute")
add_arrow(ax, (0.74, 0.60), (0.72, 0.60), "results")
add_arrow(ax, (0.55, 0.45), (0.55, 0.33), "events")
add_arrow(ax, (0.38, 0.70), (0.27, 0.70), "result", fs=9)
add_arrow(ax, (0.48, 0.45), (0.20, 0.62), "abort notice", dashed=True, fs=8)

ax.text(0.5, 0.06, "Policy: keep ΔIₜ ≤ Δ_budget; abort if ΔIₜ ≥ Δ_kill.",
        ha="center", va="center", fontsize=9)

fig.tight_layout(pad=0.2)
fig.savefig(OUT / "system_overview_clean.pdf", bbox_inches="tight")
fig.savefig(OUT / "system_overview_clean.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---------- Figure 2: Threat model matrix (clean, table style) ----------
surfaces = [
    "S1 Timing & latency", "S2 Scheduler metadata", "S3 Co-tenant cross-talk",
    "S4 EM/acoustic leakage", "S5 Network telemetry", "S6 Result side-channels"
]
adv_cols = ["A1", "A2", "A3", "A4"]
def_cols = ["D1", "D2", "D3", "D4"]

attack = {"A1": {"S1", "S2"}, "A2": {"S3", "S6"}, "A3": {"S5"}, "A4": {"S4"}}
mitig  = {"D1": {"S2", "S6"}, "D2": {"S1", "S5"}, "D3": {"S3"}, "D4": {"S1", "S2", "S6"}}

fig, ax = plt.subplots(figsize=(7.2, 3.8))
n_rows = len(surfaces)
n_cols = 1 + len(adv_cols) + len(def_cols)
ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows + 2); ax.axis("off")

# grid
for c in range(n_cols + 1): ax.plot([c, c], [0, n_rows + 1], linewidth=1)
for r in range(n_rows + 2): ax.plot([0, n_cols], [r, r], linewidth=1)

# headers
ax.text(0.5, n_rows + 0.5, "Surface", ha="center", va="center")
for i, h in enumerate(adv_cols, 1): ax.text(i + 0.5, n_rows + 0.5, h, ha="center", va="center")
for j, h in enumerate(def_cols, 1 + len(adv_cols)): ax.text(j + 0.5, n_rows + 0.5, h, ha="center", va="center")

# rows
for r, s in enumerate(surfaces):
    y = n_rows - r - 0.5
    ax.text(0.05, y, s, ha="left", va="center")
    key = s.split()[0]  # "S1", ...
    for i, a in enumerate(adv_cols, 1):
        ax.text(i + 0.5, y, "×" if key in attack[a] else "", ha="center", va="center")
    for j, d in enumerate(def_cols, 1 + len(adv_cols)):
        ax.text(j + 0.5, y, "✓" if key in mitig[d] else "", ha="center", va="center")

ax.text(0, -0.4, "Legend: × adversary pressure   ✓ mitigation", ha="left", va="center")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "threat_model_matrix.pdf", bbox_inches="tight")
fig.savefig(OUT / "threat_model_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved to: {OUT.resolve()}")
