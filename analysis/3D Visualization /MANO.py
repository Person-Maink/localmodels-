import pickle
import sys
from pathlib import Path

import numpy as np
from vedo import Mesh, Plotter, Text2D


from _path_setup import PROJECT_ROOT  # ensures root imports work
from FILENAME import MANO_RIGHT_PATH  


RIGHT_COLOR = "crimson"
LEFT_COLOR = "royalblue"
VERTEX_LABEL_COLOR = "black"
FACE_LABEL_COLOR = "darkgreen"
MESH_ALPHA = 0.45
HAND_X_OFFSET = 0.20


def apply_numpy_legacy_aliases():
    alias_map = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
        "nan": float("nan"),
        "inf": float("inf"),
    }
    for name, value in alias_map.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def _to_numpy_array(value, dtype=np.float32):
    if hasattr(value, "r"):
        value = value.r
    return np.asarray(value, dtype=dtype)


def load_mano_template(path):
    apply_numpy_legacy_aliases()

    with open(path, "rb") as f:
        mano = pickle.load(f, encoding="latin1")

    if "v_template" not in mano:
        raise KeyError(f"Missing 'v_template' in MANO file: {path}")
    if "f" not in mano:
        raise KeyError(f"Missing 'f' in MANO file: {path}")

    verts = _to_numpy_array(mano["v_template"], dtype=np.float32)
    faces = _to_numpy_array(mano["f"], dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Invalid MANO vertices shape: {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid MANO faces shape: {faces.shape}")

    return verts, faces


def build_left_hand(verts, faces):
    verts_left = verts.copy()
    verts_left[:, 0] *= -1.0
    faces_left = faces[:, [0, 2, 1]].copy()
    return verts_left, faces_left


def make_labeled_mesh(verts, faces, side, offset):
    verts_world = verts + np.asarray(offset, dtype=np.float32).reshape(1, 3)

    mesh = Mesh([verts_world, faces])
    mesh.color(RIGHT_COLOR if side == "right" else LEFT_COLOR)
    mesh.alpha(MESH_ALPHA)
    mesh.linewidth(0.5)

    diag = float(np.linalg.norm(verts_world.max(axis=0) - verts_world.min(axis=0)))
    point_label_scale = max(diag * 0.012, 0.002) * 0.3
    face_label_scale = max(diag * 0.015, 0.0025) * 0.3

    vertex_labels = mesh.labels(
        "id",
        on="points",
        scale=point_label_scale,
        font="VTK",
        c=VERTEX_LABEL_COLOR,
    )
    face_labels = mesh.labels(
        "id",
        on="cells",
        scale=face_label_scale,
        font="VTK",
        c=FACE_LABEL_COLOR,
    )

    if hasattr(vertex_labels, "pickable"):
        vertex_labels.pickable(False)
    if hasattr(face_labels, "pickable"):
        face_labels.pickable(False)

    return {
        "side": side,
        "mesh": mesh,
        "vertex_labels": vertex_labels,
        "face_labels": face_labels,
        "verts_world": verts_world,
    }


def _event_get(evt, key, default=None):
    if isinstance(evt, dict):
        return evt.get(key, default)
    return getattr(evt, key, default)


def main():
    mano_path = Path(MANO_RIGHT_PATH)
    if not mano_path.exists():
        raise FileNotFoundError(f"MANO file not found: {mano_path}")

    verts_right, faces_right = load_mano_template(mano_path)
    verts_left, faces_left = build_left_hand(verts_right, faces_right)

    right_pack = make_labeled_mesh(
        verts=verts_right,
        faces=faces_right,
        side="right",
        offset=(+HAND_X_OFFSET, 0.0, 0.0),
    )
    left_pack = make_labeled_mesh(
        verts=verts_left,
        faces=faces_left,
        side="left",
        offset=(-HAND_X_OFFSET, 0.0, 0.0),
    )

    hands = {"right": right_pack, "left": left_pack}
    vertex_label_actors = [right_pack["vertex_labels"], left_pack["vertex_labels"]]
    face_label_actors = [right_pack["face_labels"], left_pack["face_labels"]]

    state = {
        "show_vertex_labels": True,
        "show_face_labels": True,
    }

    instructions = Text2D(
        "MANO Index Picker\n"
        "- Left click: print nearest vertex_id + face_id\n"
        "- Key 'v': toggle vertex labels\n"
        "- Key 'f': toggle face labels\n"
        "- Key 'q': quit",
        pos="top-left",
        s=0.9,
        c="black",
        bg="white",
        alpha=1.0,
    )

    plt = Plotter(bg="white", axes=1)

    for pack in (right_pack, left_pack):
        plt += pack["mesh"]
        plt += pack["vertex_labels"]
        plt += pack["face_labels"]

    plt += instructions

    # def infer_side_from_point(world_pt):
    #     dists = {}
    #     for side, pack in hands.items():
    #         vid = int(pack["mesh"].closest_point(world_pt, return_point_id=True))
    #         d = np.linalg.norm(pack["verts_world"][vid] - world_pt)
    #         dists[side] = d
    #     return min(dists, key=dists.get)

    # def on_click(evt):
    #     picked = _event_get(evt, "picked3d", None)
    #     if picked is None:
    #         return

    #     world_pt = np.asarray(picked, dtype=np.float32).reshape(3)
    #     side = infer_side_from_point(world_pt)
    #     mesh = hands[side]["mesh"]

    #     vertex_id = int(mesh.closest_point(world_pt, return_point_id=True))
    #     face_id = int(mesh.closest_point(world_pt, return_cell_id=True))

    #     print(
    #         "[pick] "
    #         f"side={side} "
    #         f"vertex_id={vertex_id} "
    #         f"face_id={face_id} "
    #         f"picked=({world_pt[0]:.6f}, {world_pt[1]:.6f}, {world_pt[2]:.6f})"
    #     )

    def set_vertex_labels_visible(visible):
        for actor in vertex_label_actors:
            actor.on() if visible else actor.off()

    def set_face_labels_visible(visible):
        for actor in face_label_actors:
            actor.on() if visible else actor.off()

    def on_key(evt):
        key = _event_get(evt, "keypress", "")
        if not key:
            return

        key = str(key).lower()
        if key == "v":
            state["show_vertex_labels"] = not state["show_vertex_labels"]
            set_vertex_labels_visible(state["show_vertex_labels"])
            plt.render()
            print(f"[toggle] vertex_labels={state['show_vertex_labels']}")
        elif key == "f":
            state["show_face_labels"] = not state["show_face_labels"]
            set_face_labels_visible(state["show_face_labels"])
            plt.render()
            print(f"[toggle] face_labels={state['show_face_labels']}")
        elif key == "q":
            plt.close()

    # plt.add_callback("LeftButtonPress", on_click)
    plt.add_callback("KeyPress", on_key)

    print(f"Loaded MANO template: {mano_path}")
    print(f"Right hand: {len(verts_right)} vertices, {len(faces_right)} faces")
    print(f"Left hand:  {len(verts_left)} vertices, {len(faces_left)} faces")

    plt.show(title="MANO Face/Vertex Index Visualizer", viewup="z")
    plt.close()


if __name__ == "__main__":
    main()
