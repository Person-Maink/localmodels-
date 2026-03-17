import os
import shlex
import subprocess
import multiprocessing as mp
from concurrent import futures

from preproc.datasets import update_args
from preproc.export_hamer import export_sequence_results

ROOT_DIR = os.path.abspath(f"{__file__}/../../../")
SRC_DIR = os.path.join(ROOT_DIR, "third-party/hamer")

def launch_hamer(gpus, seq, img_dir, res_dir, name, datatype, overwrite=False):
    """
    run hamer using GPU pool
    """
    cur_proc = mp.current_process()
    print("PROCESS", cur_proc.name, cur_proc._identity)
    # 1-indexed processes
    # worker_id = cur_proc._identity[0] - 1 if len(cur_proc._identity) > 0 else 0
    # gpu = gpus[worker_id % len(gpus)]
    gpu = gpus[0]

    HAMER_DIR = SRC_DIR
    print("HAMER DIR", HAMER_DIR)

    demo_res_path = os.path.join(res_dir, f"demo_{name}.pkl")
    cmd_args = [
        "python",
        "-u",
        "run.py",
        "--img_folder",
        img_dir,
        "--res_folder",
        demo_res_path,
        "--batch_size=48",
        "--side_view",
        "--save_mesh",
        "--full_frame",
        "--type",
        datatype,
        "--checkpoint",
        ROOT_DIR,
        # "--render"
    ]

    print(f"cd {HAMER_DIR}; CUDA_VISIBLE_DEVICES={gpu} " + " ".join(shlex.quote(arg) for arg in cmd_args))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return subprocess.call(cmd_args, cwd=HAMER_DIR, env=env)


def process_seq(
    gpus,
    out_root,
    seq,
    img_dir,
    out_name="hamer_out",
    datatype=None,
    track_name="track_preds",
    shot_name="shot_idcs",
    overwrite=False,
):
    """
    Run and export HAMER results
    """
    name = os.path.basename(seq)
    res_root = f"{out_root}/{out_name}/{seq}"
    os.makedirs(res_root, exist_ok=True)
    res_dir = os.path.join(res_root, "results")
    os.makedirs(res_dir, exist_ok=True)
    res_path = f"{res_root}/{name}.pkl"

    if overwrite or not os.path.isfile(res_path):
        res = launch_hamer(gpus, seq, img_dir, res_dir, name, datatype, overwrite)
        demo_res_path = f"{res_dir}/demo_{name}.pkl"
        print(f"rename {demo_res_path} into ", res_path)
        if not os.path.isfile(demo_res_path):
            raise FileNotFoundError(
                f"HaMeR did not produce {demo_res_path}. "
                f"HaMeR exit code={res}. "
                "Check the HaMeR traceback above. "
                "If network is blocked, set HAMER_DETECTRON2_CKPT to a local "
                "model_final_f05665.pkl path."
            )
        os.rename(demo_res_path, res_path)
        assert res == 0, "HAMER FAILED"

    # export the HAMER predictions
    track_dir = f"{out_root}/{track_name}/{seq}"
    shot_path = f"{out_root}/{shot_name}/{seq}.json"

    export_sequence_results(res_path, track_dir, shot_path)
    return 0


def get_out_dir(src_root, src_dir, src_token, out_token):
    """
    :param src_root (str) root of all data
    :param src_dir (str) img input dir
    :param src_token (str) parent name of image input dir
    :param out_token (str) name of output dir
    """
    src_suffix = src_dir.removeprefix(src_root)
    out_dir = f"{out_root}/{src_suffix}"
    return out_dir.replace(src_token, out_token)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="posetrack", help="dataset to process")
    parser.add_argument("--root", default=None, help="root dir of data, default None")
    parser.add_argument("--split", default="val", help="split of dataset, default val")
    parser.add_argument(
        "--img_name", default=None, help="input image directory name, default None"
    )
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--gpus", nargs="*", default=[0])
    parser.add_argument("-y", "--overwrite", action="store_true")

    args = parser.parse_args()
    args = update_args(args)

    out_root = f"{args.root}/slahmr/{args.split}"

    print(f"running phalp on {len(args.img_dirs)} image directories")
    if len(args.gpus) > 1:
        with futures.ProcessPoolExecutor(max_workers=len(args.gpus)) as exe:
            for img_dir, seq in zip(args.img_dirs, args.seqs):
                exe.submit(
                    process_seq,
                    args.gpus,
                    out_root,
                    seq,
                    img_dir,
                    overwrite=args.overwrite,
                )
    else:
        for img_dir, seq in zip(args.img_dirs, args.seqs):
            process_seq(args.gpus, out_root, seq, img_dir, overwrite=args.overwrite)
