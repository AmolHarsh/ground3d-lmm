#!/usr/bin/env python3
"""
Ground3D frame extractor.

At INFERENCE the model needs only the RGB frame image referenced by a QA sample's
(scene, frame). Pose / intrinsics / depth are NOT used. This is the entire preprocessing
step a user must run: given (dataset, scene, frame), produce that one RGB image.

Two ways to call it — both extract exactly ONE frame for ONE question:

  (A) Point it at the question's QA json; scene+frame+dataset are read from meta_info:
      python extract_frame.py --qa_file .../distance_estimation/scene0019_00_frame_195.json \
                              --root . --out /tmp/f.jpg

  (B) Or pass scene/frame explicitly:
      python extract_frame.py --dataset scannet  --root . --scene scene0000_00 --frame 1000 --out /tmp/f.jpg
      python extract_frame.py --dataset scannetpp --root . --scene 00777c41d4 --frame 1014 --out /tmp/f.png

`--root` is the data root holding `scannet/` and/or `scannetpp/` (this sample folder works as-is).
ScanNet vs ScanNet++ is auto-detected from the scene id (`sceneXXXX_XX` -> ScanNet, else ScanNet++).
"""
import argparse, json, os, re, shutil, subprocess, sys


def detect_dataset(scene):
    """ScanNet scene ids look like scene0019_00; ScanNet++ ids are hashes (e.g. 09c1414f1b)."""
    return "scannet" if re.fullmatch(r"scene\d{4}_\d{2}", scene) else "scannetpp"


def resolve_from_qa(qa_file):
    """Read (dataset, scene, frame) for a single question from its QA json meta_info."""
    d = json.load(open(qa_file))

    def find_meta(o):
        if isinstance(o, dict):
            if "meta_info" in o:
                return o["meta_info"]
            for v in o.values():
                r = find_meta(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find_meta(v)
                if r:
                    return r
        return None

    m = find_meta(d)
    if m and "scene" in m and "frame_num" in m:
        scene, frame = m["scene"], int(m["frame_num"])
    else:
        # fall back to the filename convention <scene>_frame_<frame>.json
        base = os.path.basename(qa_file)
        mm = re.match(r"(.+)_frame_(\d+)\.json$", base)
        if not mm:
            sys.exit(f"Could not read scene/frame from {qa_file} (no meta_info, unexpected filename).")
        scene, frame = mm.group(1), int(mm.group(2))
    return detect_dataset(scene), scene, frame


def extract_scannet(root, scene, frame, out):
    """ScanNet RGB lives at scannet/scans/<scene>/color/<frame>.jpg after SensReader extraction."""
    src = os.path.join(root, "scannet", "scans", scene, "color", f"{frame}.jpg")
    if not os.path.exists(src):
        sys.exit(
            f"[scannet] {src} not found.\n"
            f"  -> Run the official SensReader once to unpack <scene>.sens into color/depth/pose/intrinsic:\n"
            f"     python reader.py --filename {scene}.sens --output_path scannet/scans/{scene} --export_color_images\n"
            f"  Then re-run this script."
        )
    shutil.copyfile(src, out)
    print(f"[scannet] {scene} frame {frame} -> {out}")


def extract_scannetpp(root, scene, frame, out):
    """ScanNet++ iPhone RGB.

    Reproduces the dataset-generation method (scanet++/code/1_question_generation_filter.py):
    the QA `frame_num` is the 0-based rgb.mkv frame index, read via OpenCV
    `cap.set(CAP_PROP_POS_FRAMES, frame); cap.read()` and written as `<frame>.jpg`.
    Falls back to a pre-decoded `rgb/frame_<N:06d>.png` (used by the bundled sample), then ffmpeg.
    """
    iphone = os.path.join(root, "scannetpp", "data", scene, "iphone")
    mkv = os.path.join(iphone, "rgb.mkv")

    # primary: match the generation pipeline exactly (OpenCV POS_FRAMES seek)
    if os.path.exists(mkv):
        try:
            import cv2
        except ImportError:
            cv2 = None
        if cv2 is not None:
            cap = cv2.VideoCapture(mkv)
            if not cap.isOpened():
                sys.exit(f"[scannetpp] could not open {mkv}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, frame_bgr = cap.read()
            cap.release()
            if not ok or frame_bgr is None:
                sys.exit(f"[scannetpp] failed to read frame {frame} from {mkv}")
            cv2.imwrite(out, frame_bgr)
            print(f"[scannetpp] {scene} frame {frame} (cv2 POS_FRAMES, matches generation) -> {out}")
            return
        # cv2 unavailable -> ffmpeg (0-based decode count; equivalent indexing)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mkv,
                        "-vf", f"select=eq(n\\,{frame})", "-vframes", "1", out], check=True)
        print(f"[scannetpp] {scene} frame {frame} (ffmpeg fallback) -> {out}")
        return

    # fallback: pre-decoded frame shipped in the sample
    pre = os.path.join(iphone, "rgb", f"frame_{frame:06d}.png")
    if os.path.exists(pre):
        shutil.copyfile(pre, out)
        print(f"[scannetpp] {scene} frame {frame} (pre-decoded sample) -> {out}")
        return
    sys.exit(f"[scannetpp] neither {mkv} nor {pre} found. Download iphone/rgb.mkv for {scene}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_file", help="a single question's QA json; scene+frame read from meta_info")
    ap.add_argument("--dataset", choices=["scannet", "scannetpp"], help="(only if not using --qa_file)")
    ap.add_argument("--root", default=".", help="data root containing scannet/ and/or scannetpp/")
    ap.add_argument("--scene", help="(only if not using --qa_file)")
    ap.add_argument("--frame", type=int, help="(only if not using --qa_file)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.qa_file:
        dataset, scene, frame = resolve_from_qa(a.qa_file)
    else:
        if not (a.dataset and a.scene and a.frame is not None):
            ap.error("provide --qa_file, or all of --dataset/--scene/--frame")
        dataset, scene, frame = a.dataset, a.scene, a.frame

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    (extract_scannet if dataset == "scannet" else extract_scannetpp)(a.root, scene, frame, a.out)


if __name__ == "__main__":
    main()
