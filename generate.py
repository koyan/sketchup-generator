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
    """Write a self-contained Collada .dae using lxml for direct XML construction.

    Writing the XML directly (rather than using pycollada) avoids pycollada's
    automatic <transparency>1.0</transparency> output, which SketchUp interprets
    as fully transparent and causes shapes to appear invisible/black.

    Textures are embedded as base64 data URIs so the file can be copied or
    shared without needing to carry the original image files alongside it.

    Each entry in objects must be a leaf dict:
        mesh         trimesh.Trimesh
        uv           np.ndarray | None
        texture_path str | None
        name         str
    """
    from lxml import etree as ET

    NS = "http://www.collada.org/2005/11/COLLADASchema"

    def _sub(parent, tag, text=None, **attrs):
        e = ET.SubElement(
            parent, f"{{{NS}}}{tag}",
            **{k: str(v) for k, v in attrs.items()},
        )
        if text is not None:
            e.text = str(text)
        return e

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # Assign a stable index to every object
    for i, obj in enumerate(objects):
        obj["_idx"] = i

    # --- Root ----------------------------------------------------------------
    root = ET.Element(f"{{{NS}}}COLLADA", version="1.4.1", nsmap={None: NS})

    # --- <asset> -------------------------------------------------------------
    asset = _sub(root, "asset")
    _sub(asset, "created",  text="2024-01-01")
    _sub(asset, "modified", text="2024-01-01")
    _sub(asset, "unit", name="meter", meter="1.0")
    _sub(asset, "up_axis", text="Y_UP")

    # --- <library_images> ----------------------------------------------------
    textures_to_embed = {}  # bare_filename → abs_path
    lib_images = _sub(root, "library_images")
    for obj in objects:
        i        = obj["_idx"]
        tex_path = obj.get("texture_path")
        if tex_path and obj.get("uv") is not None:
            tex_filename = os.path.basename(tex_path)
            textures_to_embed[tex_filename] = os.path.abspath(tex_path)
            img_el = _sub(lib_images, "image", id=f"img_{i}", name=f"img_{i}")
            _sub(img_el, "init_from", text="./" + tex_filename)

    # --- <library_effects> ---------------------------------------------------
    lib_effects = _sub(root, "library_effects")
    for obj in objects:
        i        = obj["_idx"]
        tex_path = obj.get("texture_path")
        uv       = obj.get("uv")
        effect   = _sub(lib_effects, "effect", id=f"effect_{i}")
        profile  = _sub(effect, "profile_COMMON")
        if tex_path and uv is not None:
            surf_param = _sub(profile, "newparam", sid=f"surf_{i}")
            surface    = _sub(surf_param, "surface", type="2D")
            _sub(surface, "init_from", text=f"img_{i}")
            samp_param = _sub(profile, "newparam", sid=f"samp_{i}")
            sampler    = _sub(samp_param, "sampler2D")
            _sub(sampler, "source", text=f"surf_{i}")
            tech    = _sub(profile, "technique", sid="common")
            diffuse = _sub(_sub(tech, "lambert"), "diffuse")
            _sub(diffuse, "texture", texture=f"samp_{i}", texcoord="UVSET0")
        else:
            tech    = _sub(profile, "technique", sid="common")
            diffuse = _sub(_sub(tech, "lambert"), "diffuse")
            _sub(diffuse, "color", sid="diffuse", text="0.784 0.784 0.784 1.0")

    # --- <library_materials> -------------------------------------------------
    lib_mats = _sub(root, "library_materials")
    for obj in objects:
        i    = obj["_idx"]
        name = obj.get("name", f"obj{i}")
        mat  = _sub(lib_mats, "material", id=f"mat_{i}", name=name)
        _sub(mat, "instance_effect", url=f"#effect_{i}")

    # --- <library_geometries> ------------------------------------------------
    lib_geoms = _sub(root, "library_geometries")
    for obj in objects:
        i    = obj["_idx"]
        mesh = obj["mesh"]
        uv   = obj.get("uv")
        name = obj.get("name", f"obj{i}")

        verts = mesh.vertices.astype(np.float32)
        norms = (
            np.repeat(mesh.face_normals, 3, axis=0).astype(np.float32)
            if uv is not None
            else mesh.vertex_normals.astype(np.float32)
        )

        geom_el = _sub(lib_geoms, "geometry", id=f"geom_{i}", name=name)
        mesh_el = _sub(geom_el,   "mesh")

        # Position source
        v_count = len(verts)
        src_v   = _sub(mesh_el, "source", id=f"verts_{i}")
        _sub(src_v, "float_array", id=f"verts_{i}-array", count=str(v_count * 3),
             text=" ".join(f"{x:.6g}" for x in verts.ravel()))
        acc_v = _sub(_sub(src_v, "technique_common"), "accessor",
                     source=f"#verts_{i}-array", count=str(v_count), stride="3")
        for ax in ("X", "Y", "Z"):
            _sub(acc_v, "param", name=ax, type="float")

        # Normal source
        n_count = len(norms)
        src_n   = _sub(mesh_el, "source", id=f"norms_{i}")
        _sub(src_n, "float_array", id=f"norms_{i}-array", count=str(n_count * 3),
             text=" ".join(f"{x:.6g}" for x in norms.ravel()))
        acc_n = _sub(_sub(src_n, "technique_common"), "accessor",
                     source=f"#norms_{i}-array", count=str(n_count), stride="3")
        for ax in ("X", "Y", "Z"):
            _sub(acc_n, "param", name=ax, type="float")

        # UV source
        if uv is not None:
            uv_arr   = uv.astype(np.float32)
            uv_count = len(uv_arr)
            src_uv   = _sub(mesh_el, "source", id=f"uvs_{i}")
            _sub(src_uv, "float_array", id=f"uvs_{i}-array", count=str(uv_count * 2),
                 text=" ".join(f"{x:.6g}" for x in uv_arr.ravel()))
            acc_uv = _sub(_sub(src_uv, "technique_common"), "accessor",
                          source=f"#uvs_{i}-array", count=str(uv_count), stride="2")
            for ax in ("S", "T"):
                _sub(acc_uv, "param", name=ax, type="float")

        # <vertices> — required bridge element in the COLLADA spec
        verts_vtx = _sub(mesh_el, "vertices", id=f"verts_{i}-vtx")
        _sub(verts_vtx, "input", semantic="POSITION", source=f"#verts_{i}")

        # <triangles>
        tri_count = len(mesh.faces)
        tri_el    = _sub(mesh_el, "triangles", count=str(tri_count), material=f"mat_{i}")
        _sub(tri_el, "input", semantic="VERTEX",   source=f"#verts_{i}-vtx", offset="0")
        _sub(tri_el, "input", semantic="NORMAL",   source=f"#norms_{i}",     offset="1")
        if uv is not None:
            _sub(tri_el, "input", semantic="TEXCOORD", source=f"#uvs_{i}", offset="2", set="0")
            flat    = np.arange(len(verts))
            indices = np.column_stack([flat, flat, flat]).ravel()
        else:
            face_idx = mesh.faces.ravel()
            indices  = np.column_stack([face_idx, face_idx]).ravel()
        _sub(tri_el, "p", text=" ".join(str(x) for x in indices))

    # --- <library_visual_scenes> ---------------------------------------------
    lib_vs    = _sub(root, "library_visual_scenes")
    vis_scene = _sub(lib_vs, "visual_scene", id="scene", name="scene")

    for obj in objects:
        i        = obj["_idx"]
        name     = obj.get("name", f"obj{i}")
        safe_id  = name.replace(" ", "_")
        node_el  = _sub(vis_scene, "node", id=f"node_{i}", name=safe_id, type="NODE")
        inst     = _sub(node_el, "instance_geometry", url=f"#geom_{i}")
        bind     = _sub(inst, "bind_material")
        tech     = _sub(bind, "technique_common")
        inst_mat = _sub(tech, "instance_material",
                        symbol=f"mat_{i}", target=f"#mat_{i}")
        if obj.get("texture_path") and obj.get("uv") is not None:
            _sub(inst_mat, "bind_vertex_input",
                 semantic="UVSET0", input_semantic="TEXCOORD", input_set="0")

    # --- <scene> -------------------------------------------------------------
    scene_el = _sub(root, "scene")
    _sub(scene_el, "instance_visual_scene", url="#scene")

    # --- Write ---------------------------------------------------------------
    tree = ET.ElementTree(root)
    tree.write(output_path, xml_declaration=True, encoding="utf-8", pretty_print=True)

    # Post-process: replace ./filename references with embedded base64 data URIs.
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


def build_cylinder(radius_mm: float, height_mm: float, axis: str, inner_radius_mm: float = 0) -> trimesh.Trimesh:
    """Return a cylinder (or hollow tube) mesh.

    axis controls which world axis the cylinder runs along:
      'z' → vertical (default), 'x' → left-right, 'y' → front-back.
    inner_radius_mm > 0 hollows the cylinder by boolean-subtracting a coaxial
      inner cylinder, producing a tube with the given bore radius.
    sections=64 gives a smooth circular cross-section without excessive faces.
    """
    mesh = trimesh.creation.cylinder(
        radius=radius_mm * MM_TO_M,
        height=height_mm * MM_TO_M,
        sections=64,
    )

    if inner_radius_mm > 0:
        if inner_radius_mm >= radius_mm:
            raise ValueError(
                f"--inner-radius ({inner_radius_mm}) must be less than --radius ({radius_mm})"
            )
        # Subtract a slightly taller inner cylinder to guarantee clean cap faces
        inner = trimesh.creation.cylinder(
            radius=inner_radius_mm * MM_TO_M,
            height=height_mm * MM_TO_M * 1.01,
            sections=64,
        )
        mesh = trimesh.boolean.difference([mesh, inner])

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
    "cylinder": lambda args: build_cylinder(args.radius, args.height, args.axis, args.inner_radius),
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
        help="Outer radius in millimetres, used by: cylinder, sphere (default: 50)",
    )
    parser.add_argument(
        "--inner-radius",
        dest="inner_radius",
        type=float,
        default=0.0,
        help="Inner bore radius in millimetres; hollows the cylinder when > 0, used by: cylinder (default: 0)",
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
