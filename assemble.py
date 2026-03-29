import argparse
import json
from types import SimpleNamespace

import numpy as np
import trimesh

from generate import (
    SHAPES, MM_TO_M, TEXTURES_DIR,
    find_texture, apply_texture, write_collada,
)


# Default values for every shape parameter, mirroring the CLI defaults in generate.py.
# Any key not present in the JSON "parameters" block falls back to these.
PARAM_DEFAULTS = {
    "size":     100.0,
    "width":    100.0,
    "depth":    100.0,
    "height":   100.0,
    "radius":    50.0,
    "axis":       "z",
    "length":   200.0,
    "diameter":   8.0,
    "pitch":      2.0,
    "hand":   "right",
}


def build_object(obj_def: dict, label: str) -> dict:
    """Build and position a single object; return a dict ready for write_collada.

    Rotation is applied before UV projection so the texture aligns with the
    object's final orientation. Translation is applied after UV projection so
    the tiling offset reflects world-space position.
    """
    shape = obj_def["shape"]
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape '{shape}'. Available: {list(SHAPES.keys())}")

    params = SimpleNamespace(**{**PARAM_DEFAULTS, **obj_def.get("parameters", {})})
    mesh   = SHAPES[shape](params)

    # Rotation before texture so UV normals match the final orientation
    rotation = obj_def.get("rotation", [0, 0, 0])
    if any(r != 0 for r in rotation):
        rx, ry, rz = (np.radians(r) for r in rotation)
        matrix = trimesh.transformations.euler_matrix(rx, ry, rz)
        mesh.apply_transform(matrix)

    # Texture (UV projection uses post-rotation geometry)
    uv           = None
    texture_path = None
    texture_name = obj_def.get("texture")
    if texture_name:
        path = find_texture(texture_name)
        if path is None:
            print(f"  Warning: texture '{texture_name}' not found in {TEXTURES_DIR}/, using grey")
        else:
            mesh, uv = apply_texture(mesh, path)
            texture_path = path

    # Translation after UV so world-space tiling is consistent across objects
    position = obj_def.get("position", [0, 0, 0])
    mesh.apply_translation([p * MM_TO_M for p in position])

    return {"mesh": mesh, "uv": uv, "texture_path": texture_path, "name": label}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble multiple 3D shapes from a JSON scene file into a single model."
    )
    parser.add_argument(
        "input",
        help="Path to the JSON scene file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Extension determines format: .dae, .obj, .stl (default: <input stem>.dae)",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        scene = json.load(f)

    objects = scene.get("objects", [])
    if not objects:
        print("No objects found in scene file.")
        return

    built = []
    for i, obj_def in enumerate(objects):
        label = obj_def.get("name") or f"object_{i}"
        print(f"  Building {label} ({obj_def['shape']})...")
        built.append(build_object(obj_def, label))

    output_path = args.output or args.input.rsplit(".", 1)[0] + ".dae"

    if output_path.endswith(".dae"):
        write_collada(built, output_path)
    else:
        # Non-.dae formats: concatenate geometry (texture not preserved)
        meshes   = [obj["mesh"] for obj in built]
        combined = trimesh.util.concatenate(meshes)
        combined.export(output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
