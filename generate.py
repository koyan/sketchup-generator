import argparse
import os
import sys
import trimesh
import numpy as np


# Collada (.dae) declares its internal unit as meters, which is what SketchUp
# reads on import. All size parameters are accepted in millimetres and converted
# here so the exported file represents the correct real-world dimensions.
MM_TO_M = 0.001

# Directory (relative to cwd) where texture images are looked up.
TEXTURES_DIR = "textures"

# Texture tiling scale: one tile every 100 mm keeps texel density consistent
# across faces of different sizes (e.g. a 300 mm face shows 3 tiles).
TEXTURE_SCALE_M = 0.1


# --- Texture helpers ---------------------------------------------------------

def find_texture(name: str) -> str | None:
    """Return the path to textures/<name>.<ext>, or None if no file is found.

    The name is the stem only (no extension); common image formats are tried
    in order so callers can pass e.g. 'osb' and get 'textures/osb.jpg'.
    """
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"):
        path = os.path.join(TEXTURES_DIR, name + ext)
        if os.path.exists(path):
            return path
    return None


def apply_texture(mesh: trimesh.Trimesh, texture_path: str) -> tuple:
    """Unshare vertices and compute triplanar UV coordinates.

    Returns (unshared_mesh, uv_array).  Does NOT set TextureVisuals — the caller
    passes both to write_collada which handles the material via pycollada directly.

    Each face is projected onto the plane perpendicular to its dominant normal.
    UVs are divided by TEXTURE_SCALE_M so the texture tiles every 100 mm.
    """
    faces_flat         = mesh.faces.ravel()
    new_verts          = mesh.vertices[faces_flat]                  # (N*3, 3)
    new_faces          = np.arange(len(faces_flat)).reshape(-1, 3)  # (N,   3)
    per_vertex_normals = np.repeat(mesh.face_normals, 3, axis=0)    # (N*3, 3)

    unshared = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)

    dominant = np.argmax(np.abs(per_vertex_normals), axis=1)
    uv = np.zeros((len(new_verts), 2), dtype=np.float32)
    for dom, (a, b) in enumerate([(1, 2), (0, 2), (0, 1)]):
        mask = dominant == dom
        uv[mask, 0] = new_verts[mask, a] / TEXTURE_SCALE_M
        uv[mask, 1] = new_verts[mask, b] / TEXTURE_SCALE_M

    return unshared, uv


def _embed_textures(dae_path: str, texture_map: dict) -> None:
    """Replace <init_from>./file.jpg</init_from> references with base64 data URIs.

    This makes the .dae self-contained: the image bytes are embedded directly
    in the XML so the file can be moved without carrying a separate texture file.

    texture_map: { bare_filename: absolute_source_path }
    """
    import base64

    mime_for = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png",  "bmp":  "image/bmp",
        "tif": "image/tiff", "tiff": "image/tiff",
    }

    with open(dae_path, encoding="utf-8") as f:
        content = f.read()

    for filename, src_path in texture_map.items():
        with open(src_path, "rb") as f:
            raw = f.read()
        ext      = os.path.splitext(filename)[1].lower().lstrip(".")
        mime     = mime_for.get(ext, "image/jpeg")
        b64      = base64.b64encode(raw).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        # pycollada writes the path exactly as passed to CImage, e.g. "./osb.jpg"
        content  = content.replace(f"./{filename}", data_uri)

    with open(dae_path, "w", encoding="utf-8") as f:
        f.write(content)


def write_collada(objects: list, output_path: str) -> None:
    """Write a self-contained Collada .dae using pycollada.

    Textures are embedded as base64 data URIs so the file can be copied or
    shared without needing to carry the original image files alongside it.

    Each entry in objects is a dict with:
        mesh         trimesh.Trimesh
        uv           np.ndarray | None  — (N, 2) UV coords; must be set when texture_path is set
        texture_path str | None         — path to texture image file
        name         str                — used as node / geometry identifier
    """
    import collada

    c = collada.Collada()
    c.assetInfo.upaxis    = "Y_UP"
    c.assetInfo.unitmeter = 1.0
    c.assetInfo.unitname  = "meter"

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    scene_nodes     = []
    textures_to_embed = {}  # bare_filename → absolute_source_path

    for i, obj in enumerate(objects):
        mesh     = obj["mesh"]
        uv       = obj.get("uv")
        tex_path = obj.get("texture_path")
        name     = obj.get("name", f"obj{i}")

        # --- Material -------------------------------------------------------
        if tex_path and uv is not None:
            tex_filename = os.path.basename(tex_path)
            # Register for post-write embedding; use a placeholder path for now
            textures_to_embed[tex_filename] = os.path.abspath(tex_path)

            img     = collada.material.CImage(f"img_{i}",     "./" + tex_filename)
            surface = collada.material.Surface(f"surf_{i}",   img, "2D")
            sampler = collada.material.Sampler2D(f"samp_{i}", surface)
            effect  = collada.material.Effect(
                f"effect_{i}", [surface, sampler], "lambert",
                diffuse=collada.material.Map(sampler, "UVSET0"),
            )
            c.images.append(img)
        else:
            # Flat light grey matching the face_colors fallback (200/255 ≈ 0.784)
            effect = collada.material.Effect(
                f"effect_{i}", [], "lambert",
                diffuse=(0.784, 0.784, 0.784, 1.0),
            )

        mat = collada.material.Material(f"mat_{i}", name, effect)
        c.effects.append(effect)
        c.materials.append(mat)

        # --- Geometry sources -----------------------------------------------
        verts = mesh.vertices.astype(np.float32)

        if uv is not None:
            # Unshared mesh: one face normal per vertex (repeat 3× per face)
            norms = np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32)
        else:
            # Shared-vertex mesh: smooth per-vertex normals for better shading
            norms = mesh.vertex_normals.astype(np.float32)

        vert_src = collada.source.FloatSource(f"verts_{i}", verts.ravel(), ("X", "Y", "Z"))
        norm_src = collada.source.FloatSource(f"norms_{i}", norms.ravel(), ("X", "Y", "Z"))
        sources  = [vert_src, norm_src]

        inlist = collada.source.InputList()
        inlist.addInput(0, "VERTEX",  f"#verts_{i}")
        inlist.addInput(1, "NORMAL",  f"#norms_{i}")

        if uv is not None:
            uv_src = collada.source.FloatSource(f"uvs_{i}", uv.astype(np.float32).ravel(), ("S", "T"))
            sources.append(uv_src)
            inlist.addInput(2, "TEXCOORD", f"#uvs_{i}", set="0")
            flat    = np.arange(len(verts))
            indices = np.column_stack([flat, flat, flat]).ravel()
        else:
            face_idx = mesh.faces.ravel()
            indices  = np.column_stack([face_idx, face_idx]).ravel()

        geom   = collada.geometry.Geometry(c, f"geom_{i}", name, sources)
        triset = geom.createTriangleSet(indices, inlist, f"mat_{i}")
        geom.primitives.append(triset)
        c.geometries.append(geom)

        # --- Scene node -----------------------------------------------------
        matnode  = collada.scene.MaterialNode(f"mat_{i}", mat, inputs=[])
        geomnode = collada.scene.GeometryNode(geom, [matnode])
        node     = collada.scene.Node(f"node_{i}", children=[geomnode])
        scene_nodes.append(node)

    root = collada.scene.Scene("scene", scene_nodes)
    c.scenes.append(root)
    c.scene = root
    c.write(output_path)

    # Post-process: replace file references with embedded base64 data URIs so
    # the .dae is fully self-contained and portable without a separate texture file.
    if textures_to_embed:
        _embed_textures(output_path, textures_to_embed)


# --- Shape builders ---------------------------------------------------------

def build_cube(size_mm: float) -> trimesh.Trimesh:
    """Return a cube mesh with all sides equal to size_mm."""
    s = size_mm * MM_TO_M
    return trimesh.creation.box(extents=[s, s, s])


def build_rectangle(width_mm: float, depth_mm: float, height_mm: float) -> trimesh.Trimesh:
    """Return a rectangular box mesh with independent width, depth, and height.

    SketchUp reads Collada in Y-up convention, so trimesh's Y slot is the vertical
    axis in SketchUp. Extents are ordered [X=width, Y=height, Z=depth] accordingly.
    """
    return trimesh.creation.box(extents=[
        width_mm  * MM_TO_M,  # X → SketchUp left-right
        height_mm * MM_TO_M,  # Y → SketchUp vertical (Y-up convention)
        depth_mm  * MM_TO_M,  # Z → SketchUp front-back
    ])


def build_cylinder(radius_mm: float, height_mm: float, axis: str) -> trimesh.Trimesh:
    """Return a cylinder mesh with the given radius and height.

    axis controls which world axis the cylinder runs along:
      'z' → vertical (default), 'x' → left-right, 'y' → front-back.
    sections=64 gives a smooth circular cross-section without excessive faces.
    """
    mesh = trimesh.creation.cylinder(
        radius=radius_mm * MM_TO_M,
        height=height_mm * MM_TO_M,
        sections=64,
    )
    # trimesh generates cylinders along Z. SketchUp reads Collada as Y-up, so:
    #   axis='z' (vertical)   → align with trimesh Y: rotate Z → Y (-90° around X)
    #   axis='x' (left-right) → align with trimesh X: rotate Z → X (+90° around Y)
    #   axis='y' (front-back) → trimesh Z is already front-back in SketchUp, no rotation
    if axis == "z":
        rotation = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rotation)
    elif axis == "x":
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        mesh.apply_transform(rotation)
    return mesh


def build_lead_screw(length_mm: float, diameter_mm: float, pitch_mm: float, hand: str) -> trimesh.Trimesh:
    """Return a lead screw mesh with a trapezoidal thread profile.

    The screw runs along the Z axis and is centered at the origin.
    The thread is generated by sweeping a trapezoidal cross-section along a helix.
    Crest width = pitch/2, flank angle ~30° (metric trapezoidal approximation).
    """
    length   = length_mm   * MM_TO_M
    r_major  = (diameter_mm / 2) * MM_TO_M
    pitch    = pitch_mm    * MM_TO_M

    # Thread height: 0.6 × pitch is the standard trapezoidal approximation
    r_minor = r_major - 0.6 * pitch

    n_turns       = length / pitch
    steps_per_turn = 60  # angular resolution; 60 gives smooth curves without excess faces
    n_steps       = max(int(n_turns * steps_per_turn), 1)
    n_rings       = n_steps + 1

    # Helix parameter centred at origin: t ∈ [-n_turns·π, +n_turns·π]
    # so z = t·pitch/(2π) runs from -length/2 to +length/2
    t_vals = np.linspace(-n_turns * np.pi, n_turns * np.pi, n_rings)

    angle_sign = 1 if hand == "right" else -1
    angles    = angle_sign * t_vals          # (n_rings,)
    z_centers = t_vals * pitch / (2 * np.pi) # (n_rings,)

    # Trapezoidal profile: 4 points in (radial, axial-offset) space.
    # Valley edges at r_minor ± pitch/2, crest edges at r_major ± pitch/4.
    profile_r  = np.array([r_minor, r_major, r_major, r_minor])
    profile_dz = np.array([-pitch / 2, -pitch / 4, pitch / 4, pitch / 2])
    n_p = len(profile_r)

    # Vectorised vertex generation: broadcast (n_rings,) × (n_p,) → (n_rings·n_p, 3)
    cos_a = np.cos(angles)[:, np.newaxis]   # (n_rings, 1)
    sin_a = np.sin(angles)[:, np.newaxis]
    x = (profile_r * cos_a).ravel()
    y = (profile_r * sin_a).ravel()
    z = (z_centers[:, np.newaxis] + profile_dz).ravel()
    verts = np.stack([x, y, z], axis=1)

    # Faces: connect adjacent rings with quads split into two triangles each
    i = np.arange(n_steps)[:, np.newaxis]
    j = np.arange(n_p - 1)[np.newaxis, :]
    a, b = (i * n_p + j).ravel(), (i * n_p + j + 1).ravel()
    c, d = ((i + 1) * n_p + j).ravel(), ((i + 1) * n_p + j + 1).ravel()
    faces = np.concatenate([np.stack([a, b, d], axis=1), np.stack([a, d, c], axis=1)])

    thread = trimesh.Trimesh(vertices=verts, faces=faces)

    # Shaft: plain cylinder with minor radius running the full length
    shaft = trimesh.creation.cylinder(radius=r_minor, height=length, sections=64)

    combined = trimesh.util.concatenate([shaft, thread])

    # Helix and shaft are generated along trimesh Z. Rotate Z → Y so the screw
    # stands vertically in SketchUp (Y-up convention).
    rotation = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    combined.apply_transform(rotation)

    return combined


def build_sphere(radius_mm: float) -> trimesh.Trimesh:
    """Return a sphere mesh with the given radius.

    subdivisions=4 gives a smooth enough appearance for most use cases while
    keeping the face count reasonable (~5 000 triangles).
    """
    return trimesh.creation.icosphere(subdivisions=4, radius=radius_mm * MM_TO_M)


# Map shape names to their builder functions.
# Each builder receives the parsed args so future shapes can use extra params.
SHAPES = {
    "cube": lambda args: build_cube(args.size),
    "rectangle": lambda args: build_rectangle(args.width, args.depth, args.height),
    "cylinder": lambda args: build_cylinder(args.radius, args.height, args.axis),
    "lead_screw": lambda args: build_lead_screw(args.length, args.diameter, args.pitch, args.hand),
    "sphere": lambda args: build_sphere(args.radius),
}


# --- Export -----------------------------------------------------------------

def export_mesh(mesh: trimesh.Trimesh, output_path: str) -> None:
    """Export mesh to a file; format is inferred from the file extension."""
    mesh.export(output_path)
    print(f"Saved: {output_path}")


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D model files compatible with SketchUp 2017."
    )
    parser.add_argument(
        "--shape",
        choices=list(SHAPES.keys()),
        default="cube",
        help="Shape to generate (default: cube)",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=100.0,
        help="Side length in millimetres, used by: cube (default: 100)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=100.0,
        help="Width in millimetres along the X axis, used by: rectangle (default: 100)",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=100.0,
        help="Depth in millimetres along the Y axis, used by: rectangle (default: 100)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=100.0,
        help="Height in millimetres along the Z axis, used by: rectangle (default: 100)",
    )
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z"],
        default="z",
        help="Axis the cylinder runs along — z: vertical, x: left-right, y: front-back (default: z)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Radius in millimetres, used by: sphere (default: 50)",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=200.0,
        help="Total length in millimetres, used by: lead_screw (default: 200)",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=8.0,
        help="Major (outer) diameter in millimetres, used by: lead_screw (default: 8)",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=2.0,
        help="Distance between thread crests in millimetres, used by: lead_screw (default: 2)",
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left"],
        default="right",
        help="Thread handedness, used by: lead_screw (default: right)",
    )
    parser.add_argument(
        "--texture",
        default=None,
        help="Texture name stem (e.g. 'osb' → textures/osb.jpg). Omit for plain grey.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Extension determines format: .dae, .obj, .stl (default: <shape>.dae)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = args.output or f"{args.shape}.dae"

    builder = SHAPES[args.shape]
    mesh = builder(args)

    uv           = None
    texture_path = None

    if args.texture:
        path = find_texture(args.texture)
        if path is None:
            print(f"Warning: texture '{args.texture}' not found in {TEXTURES_DIR}/, using grey")
            mesh.visual.face_colors = [200, 200, 200, 255]
        else:
            mesh, uv = apply_texture(mesh, path)
            texture_path = path
    else:
        mesh.visual.face_colors = [200, 200, 200, 255]

    if output_path.endswith(".dae"):
        write_collada(
            [{"mesh": mesh, "uv": uv, "texture_path": texture_path, "name": args.shape}],
            output_path,
        )
        print(f"Saved: {output_path}")
    else:
        export_mesh(mesh, output_path)


if __name__ == "__main__":
    main()
