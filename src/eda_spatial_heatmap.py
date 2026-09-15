"""
Stage 2 addition - spatial view of total Internet traffic across Milan's
100x100 grid, to address the "spatial" half of the EDA rubric criterion
(everything done so far treats space only as a ranked list of totals, not
an actual map).

IMPORTANT CAVEAT (kept honest rather than guessed): the Milan grid's
square_id (1-10000) is derived here as row-major order - row = (id-1)//100,
col = (id-1)%100 - which is the standard, near-universal way such grids are
numbered. This was NOT possible to confirm against an official source
during this session (no geojson/coordinate file is bundled with the raw
data, and a web search did not turn up an explicit statement of which
corner numbering starts from or which compass direction rows/columns
increase in). So: this heatmap is reliable for showing RELATIVE spatial
structure (are high-traffic squares clustered together or scattered?) but
the specific up/down/left/right = compass-direction mapping is NOT
confirmed and should not be asserted as "north is up" etc. without
checking an authoritative source.

Run: python src/eda_spatial_heatmap.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

TOTALS = Path("data/processed/total_internet_traffic_by_square.csv")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

GRID_SIZE = 100
NAMED_SQUARES = {5161: "busiest (#1)", 5059: "#2", 5259: "#3", 4159: "Sq. 4159", 4556: "Sq. 4556"}


def square_id_to_rc(square_id: int) -> tuple[int, int]:
    idx = square_id - 1
    return idx // GRID_SIZE, idx % GRID_SIZE


def main():
    totals = pd.read_csv(TOTALS)
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for _, row in totals.iterrows():
        r, c = square_id_to_rc(int(row["square_id"]))
        grid[r, c] = row["total_internet_traffic"]

    print("Grid positions (row, col) of the named squares:")
    positions = {}
    for sq, label in NAMED_SQUARES.items():
        r, c = square_id_to_rc(sq)
        positions[sq] = (r, c)
        print(f"  square {sq} ({label}): row={r}, col={c}")

    print("\nDistances between named squares (grid cells, Euclidean):")
    ids = list(positions)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            (r1, c1), (r2, c2) = positions[a], positions[b]
            dist = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
            print(f"  {a} <-> {b}: {dist:.1f} cells apart")

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(grid, norm=LogNorm(vmin=max(grid[grid > 0].min(), 1), vmax=grid.max()),
                    cmap="inferno", origin="lower")
    cbar = fig.colorbar(im, ax=ax, label="Total internet traffic (log scale)")

    for sq, label in NAMED_SQUARES.items():
        r, c = positions[sq]
        ax.scatter(c, r, s=60, facecolors="none", edgecolors="cyan", linewidths=1.5)
        ax.annotate(str(sq), (c, r), color="cyan", fontsize=8, xytext=(4, 4),
                    textcoords="offset points")

    ax.set_title("Total internet traffic across Milan's 100x100 grid\n"
                  "(row/col derived from square_id; compass orientation NOT confirmed - see caveat in script)")
    ax.set_xlabel("Grid column (derived from square_id)")
    ax.set_ylabel("Grid row (derived from square_id)")
    fig.tight_layout()
    out_path = FIG_DIR / "spatial_heatmap_all_squares.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
