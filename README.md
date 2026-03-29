# SketchUp Generator

A Python toolset for programmatically generating 3D model files compatible with **SketchUp 2017**. Models are exported as self-contained `.dae` (Collada) files with textures embedded directly in the file, so they can be shared and imported without carrying separate image files.

Everything runs inside a Docker container — no local Python environment required.

---

## Getting Started

**Build the image and open a shell:**

```bash
make bash
```

Your project directory is volume-mounted to `/app` inside the container, so any file you edit on the host is immediately available, and any generated file appears on the host without a rebuild.

The only time you need to re-run `make bash` (which triggers a rebuild) is when `requirements.txt` changes.

---

## Scripts

### `generate.py` — Single shape

Generates one 3D shape and exports it to a file.

```bash
python generate.py --shape <shape> [parameters] [--texture <name>] [--output <path>]
```

The output format is inferred from the file extension. Supported: `.dae`, `.obj`, `.stl`. Defaults to `<shape>.dae`.

#### Shapes and their parameters

| Shape | Parameters |
|---|---|
| `cube` | `--size` (mm) |
| `rectangle` | `--width`, `--depth`, `--height` (mm) |
| `cylinder` | `--radius`, `--height` (mm), `--axis` (`x`/`y`/`z`) |
| `sphere` | `--radius` (mm) |
| `lead_screw` | `--length`, `--diameter`, `--pitch` (mm), `--hand` (`right`/`left`) |

**Cylinder axis values:**
- `z` — vertical (default)
- `x` — horizontal, left-right
- `y` — horizontal, front-back

#### Examples

```bash
# 50 mm cube with OSB texture
python generate.py --shape cube --size 50 --texture osb

# 200×100×50 mm rectangle
python generate.py --shape rectangle --width 200 --depth 100 --height 50

# Horizontal cylinder, 30 mm radius, 150 mm long
python generate.py --shape cylinder --radius 30 --height 150 --axis x

# Sphere, 40 mm radius
python generate.py --shape sphere --radius 40

# Lead screw, 300 mm long, 8 mm diameter, 2 mm pitch
python generate.py --shape lead_screw --length 300 --diameter 8 --pitch 2

# Save to a specific path
python generate.py --shape rectangle --width 100 --depth 80 --height 40 --output output/shelf.dae
```

---

### `assemble.py` — Scene from JSON

Reads a JSON file describing multiple objects with shapes, parameters, textures, and positions, and exports them as a single combined model.

```bash
python assemble.py <scene.json> [--output <path>]
```

Output defaults to `<input stem>.dae`.

#### JSON format

```json
{
  "objects": [
    {
      "name": "base_plate",
      "shape": "rectangle",
      "parameters": {
        "width": 300,
        "depth": 300,
        "height": 20
      },
      "position": [0, 0, 0],
      "rotation": [0, 0, 0],
      "texture": "osb"
    },
    {
      "name": "post",
      "shape": "cylinder",
      "parameters": {
        "radius": 10,
        "height": 200,
        "axis": "z"
      },
      "position": [150, 10, 150]
    }
  ]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | No | Label used in output and node IDs |
| `shape` | string | Yes | One of the shapes listed above |
| `parameters` | object | No | Shape-specific parameters (same names as CLI flags). Unset parameters use the same defaults as `generate.py`. |
| `position` | `[x, y, z]` | No | Position in mm. Defaults to `[0, 0, 0]`. |
| `rotation` | `[rx, ry, rz]` | No | Euler rotation in degrees, XYZ order. Defaults to `[0, 0, 0]`. |
| `texture` | string | No | Texture name stem (see Textures section). Omit for flat grey. |

#### Example

```bash
python assemble.py examples/basic_shapes.json
python assemble.py examples/basic_shapes.json --output output/scene.dae
```

---

## Textures

Place image files in the `textures/` directory. Pass the filename **without extension** as the `--texture` argument or `"texture"` JSON field:

```
textures/
  osb.jpg       →  --texture osb
  plywood.png   →  --texture plywood
  concrete.jpg  →  --texture concrete
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`

Textures are applied using **triplanar UV projection** — each face is mapped along its dominant normal axis — and tile every **100 mm** for consistent real-world density across faces of any size.

Textures are **embedded as base64 data URIs** directly inside the `.dae` file, so the exported file is fully self-contained and can be copied or shared as a single file.

---

## Coordinate system

All dimensions are in **millimetres** in the scripts. Internally they are converted to metres for the Collada export, which SketchUp reads correctly.

The exported files use **Y-up orientation** to match SketchUp's import convention:

| Axis | Direction |
|---|---|
| X | left–right |
| Y | up (vertical) |
| Z | front–back |

---

## Project structure

```
sketchup-generator/
├── generate.py          # Single-shape CLI generator
├── assemble.py          # Multi-object scene assembler
├── requirements.txt     # Python dependencies
├── Dockerfile
├── Makefile
├── textures/            # Texture images (not committed — add your own)
├── examples/            # Example JSON scene files
│   └── basic_shapes.json
└── output/              # Generated .dae files (gitignored)
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `trimesh[easy]` | 3D geometry creation and mesh operations |
| `pycollada` | Collada `.dae` file generation with full material support |
| `numpy` | Numerical arrays and linear algebra |
| `numpy-stl` | Direct STL read/write |
| `Pillow` | Image loading for texture processing |
| `scipy` | Spatial algorithms used internally by trimesh |
