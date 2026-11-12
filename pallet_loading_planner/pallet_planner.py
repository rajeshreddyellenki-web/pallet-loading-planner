"""
Pallet Loading Planner
----------------------
A simple 3D bin-packing tool that decides where every carton goes on a pallet.

Rules enforced:
  * boxes stay inside the pallet footprint and max stacking height
  * total weight per pallet <= max weight
  * no overlapping boxes
  * every box must be at least 75% supported from below (no floating stacks)
  * boxes flagged "upright" are never tipped on their side (only rotated flat)

Method:
  * Extreme-point placement heuristic (fast and easy to explain)
  * Several sorting strategies are tried (biggest first, heaviest first, ...)
    and the plan that needs the fewest pallets with the best fill is chosen.

Usage:
  python pallet_planner.py sample_boxes.csv
  python pallet_planner.py sample_boxes.csv --length 1200 --width 800 \
        --height 1500 --max-weight 1000 --out plan.csv --plot
"""

import argparse
import csv
import itertools
from dataclasses import dataclass, field

MIN_SUPPORT = 0.75  # fraction of a box's base that must rest on something


# ----------------------------------------------------------------- data model
@dataclass(frozen=True)
class Box:
    id: str
    l: int
    w: int
    h: int
    weight: float
    upright: bool = False

    @property
    def volume(self):
        return self.l * self.w * self.h


@dataclass
class Placement:
    box: Box
    x: int
    y: int
    z: int
    l: int  # size as placed (after rotation)
    w: int
    h: int


@dataclass
class Pallet:
    index: int
    L: int
    W: int
    H: int
    max_weight: float
    placements: list = field(default_factory=list)
    points: list = field(default_factory=lambda: [(0, 0, 0)])
    weight: float = 0.0

    # ---- statistics
    @property
    def used_volume(self):
        return sum(p.box.volume for p in self.placements)

    @property
    def height_used(self):
        return max((p.z + p.h for p in self.placements), default=0)

    @property
    def volume_utilization(self):
        return self.used_volume / (self.L * self.W * self.H)

    @property
    def weight_utilization(self):
        return self.weight / self.max_weight

    def centre_of_gravity(self):
        if not self.placements or self.weight == 0:
            return (0.0, 0.0)
        cx = sum(p.box.weight * (p.x + p.l / 2) for p in self.placements) / self.weight
        cy = sum(p.box.weight * (p.y + p.w / 2) for p in self.placements) / self.weight
        return cx, cy


# ------------------------------------------------------------------ geometry
def orientations(box):
    """Possible (l, w, h) rotations, flattest first."""
    if box.upright:
        dims = {(box.l, box.w, box.h), (box.w, box.l, box.h)}
    else:
        dims = set(itertools.permutations((box.l, box.w, box.h)))
    return sorted(dims, key=lambda d: (d[2], -d[0] * d[1]))


def overlaps(p, x, y, z, l, w, h):
    return (
        x < p.x + p.l and p.x < x + l
        and y < p.y + p.w and p.y < y + w
        and z < p.z + p.h and p.z < z + h
    )


def support_ratio(pallet, x, y, z, l, w):
    if z == 0:
        return 1.0
    area = 0
    for p in pallet.placements:
        if p.z + p.h == z:
            ox = max(0, min(x + l, p.x + p.l) - max(x, p.x))
            oy = max(0, min(y + w, p.y + p.w) - max(y, p.y))
            area += ox * oy
    return area / (l * w)


def try_place(pallet, box):
    """Place box at the first valid extreme point. Returns True on success."""
    if pallet.weight + box.weight > pallet.max_weight:
        return False
    for (px, py, pz) in sorted(pallet.points, key=lambda p: (p[2], p[1], p[0])):
        for (l, w, h) in orientations(box):
            if px + l > pallet.L or py + w > pallet.W or pz + h > pallet.H:
                continue
            if any(overlaps(p, px, py, pz, l, w, h) for p in pallet.placements):
                continue
            if support_ratio(pallet, px, py, pz, l, w) < MIN_SUPPORT:
                continue
            pallet.placements.append(Placement(box, px, py, pz, l, w, h))
            pallet.weight += box.weight
            pallet.points.remove((px, py, pz))
            for np_ in ((px + l, py, pz), (px, py + w, pz), (px, py, pz + h)):
                if np_ not in pallet.points:
                    pallet.points.append(np_)
            return True
    return False


# ------------------------------------------------------------------- planner
STRATEGIES = {
    "largest_volume_first": lambda b: (-b.volume, -b.weight),
    "heaviest_first": lambda b: (-b.weight, -b.volume),
    "largest_base_first": lambda b: (-(b.l * b.w), -b.weight),
    "tallest_first": lambda b: (-b.h, -b.volume),
}


def pack_all(boxes, spec, key):
    """Fill pallet after pallet. Returns (pallets, unplaceable boxes)."""
    remaining = sorted(boxes, key=key)
    pallets = []
    while remaining:
        pallet = Pallet(len(pallets) + 1, **spec)
        left = [b for b in remaining if not try_place(pallet, b)]
        if not pallet.placements:  # nothing fits even on an empty pallet
            return pallets, left
        pallets.append(pallet)
        remaining = left
    return pallets, []


def plan(boxes, spec):
    """Try every strategy and keep the best plan."""
    best = None
    for name, key in STRATEGIES.items():
        pallets, rejected = pack_all(boxes, spec, key)
        avg_fill = sum(p.volume_utilization for p in pallets) / max(1, len(pallets))
        score = (-len(rejected), -len(pallets), avg_fill)
        if best is None or score > best[0]:
            best = (score, name, pallets, rejected)
    _, name, pallets, rejected = best
    return name, pallets, rejected


# ----------------------------------------------------------------------- I/O
def load_boxes(path):
    boxes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qty = int(row.get("quantity") or 1)
            upright = (row.get("upright") or "").strip().lower() in ("1", "y", "yes", "true")
            for i in range(1, qty + 1):
                boxes.append(Box(
                    id=f"{row['id']}-{i}",
                    l=int(row["length"]), w=int(row["width"]), h=int(row["height"]),
                    weight=float(row["weight"]), upright=upright,
                ))
    return boxes


def save_plan(pallets, path):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["pallet", "box_id", "x", "y", "z", "length", "width", "height", "weight"])
        for pal in pallets:
            for p in sorted(pal.placements, key=lambda p: (p.z, p.y, p.x)):
                wr.writerow([pal.index, p.box.id, p.x, p.y, p.z, p.l, p.w, p.h, p.box.weight])


def print_report(strategy, pallets, rejected):
    print(f"\nBest strategy : {strategy}")
    print(f"Pallets needed: {len(pallets)}")
    for pal in pallets:
        cx, cy = pal.centre_of_gravity()
        off_x, off_y = cx - pal.L / 2, cy - pal.W / 2
        print(f"\n  Pallet {pal.index}")
        print(f"    boxes            : {len(pal.placements)}")
        print(f"    weight           : {pal.weight:.1f} / {pal.max_weight:.0f} kg "
              f"({pal.weight_utilization:.0%})")
        print(f"    height used      : {pal.height_used} / {pal.H} mm")
        print(f"    volume fill      : {pal.volume_utilization:.1%}")
        print(f"    centre of gravity: {off_x:+.0f} mm (length), {off_y:+.0f} mm (width) "
              f"from pallet centre")
    if rejected:
        print(f"\n  WARNING: {len(rejected)} box(es) cannot be loaded "
              f"(too big or too heavy): {', '.join(b.id for b in rejected)}")


def plot(pallets):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("\n(matplotlib not installed - skipping --plot)")
        return
    import colorsys
    for pal in pallets:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        for i, p in enumerate(pal.placements):
            x, y, z, l, w, h = p.x, p.y, p.z, p.l, p.w, p.h
            v = [(x, y, z), (x + l, y, z), (x + l, y + w, z), (x, y + w, z),
                 (x, y, z + h), (x + l, y, z + h), (x + l, y + w, z + h), (x, y + w, z + h)]
            faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                     [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                     [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]
            colour = colorsys.hsv_to_rgb((i * 0.13) % 1, 0.5, 0.9)
            ax.add_collection3d(Poly3DCollection(faces, facecolors=colour,
                                                 edgecolors="k", linewidths=0.4, alpha=0.9))
        ax.set_xlim(0, pal.L); ax.set_ylim(0, pal.W); ax.set_zlim(0, pal.H)
        ax.set_box_aspect((pal.L, pal.W, pal.H))
        ax.set_title(f"Pallet {pal.index} - {pal.volume_utilization:.0%} full")
    plt.show()


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Pallet Loading Planner")
    ap.add_argument("boxes_csv", help="CSV: id,length,width,height,weight,quantity,upright")
    ap.add_argument("--length", type=int, default=1200, help="pallet length mm")
    ap.add_argument("--width", type=int, default=800, help="pallet width mm")
    ap.add_argument("--height", type=int, default=1500, help="max stack height mm")
    ap.add_argument("--max-weight", type=float, default=1000, help="max load kg")
    ap.add_argument("--out", default="loading_plan.csv", help="output plan CSV")
    ap.add_argument("--plot", action="store_true", help="3D view (needs matplotlib)")
    args = ap.parse_args()

    boxes = load_boxes(args.boxes_csv)
    spec = dict(L=args.length, W=args.width, H=args.height, max_weight=args.max_weight)
    print(f"Loaded {len(boxes)} boxes, pallet {args.length}x{args.width} mm, "
          f"max {args.height} mm high, {args.max_weight:.0f} kg")

    strategy, pallets, rejected = plan(boxes, spec)
    print_report(strategy, pallets, rejected)
    save_plan(pallets, args.out)
    print(f"\nPlacement coordinates saved to {args.out}")
    if args.plot:
        plot(pallets)


if __name__ == "__main__":
    main()
