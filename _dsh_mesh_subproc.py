"""CPU-heavy mesh work that must NOT run inside the ComfyUI process.

Both modes below are CPU-bound C/C++ calls that hold the Python GIL for tens of
seconds to minutes. While they run in the server process, ComfyUI answers no
HTTP request at all (looks like a dead machine) - so they run here instead.

  <python> _dsh_mesh_subproc.py uvwrap      <in.npz> <out.npz>
  <python> _dsh_mesh_subproc.py postprocess <in.npz> <out.npz> \
           [--remove-floaters 1|0] [--remove-degenerate 1|0] \
           [--reduce-faces 1|0] [--max-facenum N]

  in.npz : vertices, faces
  out.npz: vertices, faces            (postprocess)
           vertices, faces, uv        (uvwrap)

Deliberately torch-free: no CUDA init, ~1s startup. hy3dshape's postprocessors
module is loaded straight from its file with two tiny stubs, because importing
it as a package would pull in torch + the model pipelines.
"""
import argparse
import functools
import importlib.util
import os
import sys
import time
import types

import numpy as np
import trimesh

PACK_ROOT = os.path.dirname(os.path.abspath(__file__))


class _Timer:
    """Stand-in for hy3dshape.utils.utils.synchronize_timer.

    The real one only does anything when HY3DGEN_DEBUG=1 (it then uses CUDA
    events, which do not exist in this CPU process); otherwise it is a no-op.
    """

    def __init__(self, name=None):
        self.name = name
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        return lambda: time.time() - self._t0

    def __exit__(self, *exc):
        if os.environ.get("HY3DGEN_DEBUG", "0") == "1" and self.name:
            print("%s takes %.1f ms" % (self.name, (time.time() - self._t0) * 1000.0), flush=True)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def _load_upstream_postprocessors():
    """Import hy3dshape/hy3dshape/postprocessors.py without its package __init__."""
    root = os.path.join(PACK_ROOT, "hy3dshape", "hy3dshape")

    def stub(name, path=None, **attrs):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = [path] if path else []
            sys.modules[name] = module
        for key, value in attrs.items():
            setattr(module, key, value)

    stub("hy3dshape", os.path.join(PACK_ROOT, "hy3dshape"))
    stub("hy3dshape.hy3dshape", root)
    stub("hy3dshape.hy3dshape.models", os.path.join(root, "models"))
    # only used for isinstance() checks on the input type - our input is a Trimesh
    stub("hy3dshape.hy3dshape.models.autoencoders", os.path.join(root, "models", "autoencoders"),
         Latent2MeshOutput=type("Latent2MeshOutput", (), {}))
    stub("hy3dshape.hy3dshape.utils", os.path.join(root, "utils"), synchronize_timer=_Timer)

    source = os.path.join(root, "postprocessors.py")
    spec = importlib.util.spec_from_file_location("hy3dshape.hy3dshape.postprocessors", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("FloaterRemover", "DegenerateFaceRemover", "FaceReducer"):
        if not hasattr(module, name):
            raise RuntimeError("upstream postprocessors.py no longer defines %s" % name)
    return module


def cmd_uvwrap(inp, outp):
    data = np.load(inp)
    mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)
    unwrapped = mesh.unwrap()  # same call as hy3dpaint.utils.uvwrap_utils.mesh_uv_wrap
    np.savez(outp, vertices=unwrapped.vertices, faces=unwrapped.faces,
             uv=np.asarray(unwrapped.visual.uv))


def cmd_postprocess(inp, outp, remove_floaters, remove_degenerate, reduce_faces, max_facenum):
    postprocessors = _load_upstream_postprocessors()
    data = np.load(inp)
    mesh = trimesh.Trimesh(vertices=data["vertices"], faces=data["faces"], process=False)

    if remove_floaters:
        mesh = postprocessors.FloaterRemover()(mesh)
        print("Removed floaters, resulting in %d vertices and %d faces"
              % (len(mesh.vertices), len(mesh.faces)), flush=True)
    if remove_degenerate:
        mesh = postprocessors.DegenerateFaceRemover()(mesh)
        print("Removed degenerate faces, resulting in %d vertices and %d faces"
              % (len(mesh.vertices), len(mesh.faces)), flush=True)
    if reduce_faces:
        mesh = postprocessors.FaceReducer()(mesh, max_facenum=max_facenum)
        print("Reduced faces, resulting in %d vertices and %d faces"
              % (len(mesh.vertices), len(mesh.faces)), flush=True)

    np.savez(outp, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("uvwrap", "postprocess"))
    parser.add_argument("inp")
    parser.add_argument("outp")
    parser.add_argument("--remove-floaters", type=int, default=1)
    parser.add_argument("--remove-degenerate", type=int, default=1)
    parser.add_argument("--reduce-faces", type=int, default=1)
    parser.add_argument("--max-facenum", type=int, default=400000)
    args = parser.parse_args()

    if args.mode == "uvwrap":
        cmd_uvwrap(args.inp, args.outp)
    else:
        cmd_postprocess(args.inp, args.outp, args.remove_floaters, args.remove_degenerate,
                        args.reduce_faces, args.max_facenum)
    return 0


if __name__ == "__main__":
    sys.exit(main())
