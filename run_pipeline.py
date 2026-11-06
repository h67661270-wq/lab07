"""
run_pipeline.py — 一键流水线：Blender 导出控制图 → sd-cli 纯 CPU 深度控制生成写实图（实验指南步骤 3）

用法（在作业目录下执行，文件按 lab07 根目录自动定位，也可在任意位置运行）：
    python scripts/run_pipeline.py --name test512 --res 512 --steps 4 --cfg 1.2
    python scripts/run_pipeline.py --name test768 --res 768 --steps 4 --seed 7      # 分辨率对比
    python scripts/run_pipeline.py --name r512s8 --res 512 --steps 8               # 步数对比
    python scripts/run_pipeline.py --name my512 --control-image C:/abs/depth.png   # 跳过 Blender
    # 拓展·步骤 5 第二段：低强度 img2img 光色统一（depth ControlNet 保持姿态）
    python scripts/run_pipeline.py --name headfig_face_harmonized --init-img output/headfig_face.png \
        --strength 0.35 --control-image output/headfig_depth.png --steps 6 --cfg 1.3 --control-strength 0.7
    python scripts/run_pipeline.py --summarize    # 汇总 output/*_timing.json，打印耗时表（填报告用）
    python scripts/run_pipeline.py --dry-run ...  # 只打印将要执行的命令，不实际运行

流水线参数（默认值与实验指南"流水线参数"表一致）：
    --res               512   输出分辨率，512 或 768
    --steps             4     采样步数，Turbo 模型 4~8 步
    --cfg               1.2   CFG 强度，Turbo 模型保持 1.0~1.5
    --control-strength  0.9   ControlNet 强度，1.0 表示完全覆盖初始信息
    --seed              42    随机种子，固定后可复现
    --threads           -1    -1 表示占满所有核心
    --control-image     无    跳过 Blender，直接使用已有深度图（建议传绝对路径）

提示词（实验指南固定）：
    正向 photorealistic render, studio soft lighting, PBR material, preserve original 3d geometry, sharp
    负向 deformed geometry, perspective changed

输出：
    output/<name>_photoreal.png   写实结果
    output/<name>_sdcli.log       sd-cli 完整输出（采样/VAE 耗时从日志解析，报告可查）
    output/<name>_timing.json     本次运行的参数与耗时汇总（供 --summarize 与报告第六节使用）
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # lab07 根目录
BLENDER_EXPORT_SCRIPT = ROOT / "blender" / "export_control_maps.py"
DEFAULT_BLEND = ROOT / "assets" / "demo_white_model.blend"
DEFAULT_OUTDIR = ROOT / "output"
DEFAULT_MODEL = ROOT / "models" / "sd_turbo-q4_0.gguf"
DEFAULT_CONTROLNET = ROOT / "models" / "control_depth_sd21-fp16.gguf"
DEFAULT_SDCLI = ROOT / "vendor" / "sdcpp-ext" / "sd-cli.exe"

DEFAULT_PROMPT = ("photorealistic render, studio soft lighting, PBR material, "
                  "preserve original 3d geometry, sharp")
DEFAULT_NEGATIVE = "deformed geometry, perspective changed"

# 部分 sd-cli 构建的采样器参数名是 --sample-method，如遇不识别可改此常量
SAMPLING_FLAG = "--sampling-method"

TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms\b|milliseconds\b|s\b|sec\b|seconds\b)", re.I)


# ---------------- 定位外部程序 ----------------

def find_blender(soft=False):
    """优先 BLENDER_EXE 环境变量，其次 PATH，最后常见安装目录。soft=True 时找不到返回 None。"""
    cand = []
    env = os.environ.get("BLENDER_EXE")
    if env:
        cand.append(Path(env))
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", r"C:\Program Files (86)")):
        cand.append(Path(root) / "Blender Foundation")
    found = None
    for c in cand:
        if c.is_file():
            found = c
            break
        if c.is_dir():
            for dirpath, _dirs, files in os.walk(c):
                if "blender.exe" in files:
                    found = Path(dirpath) / "blender.exe"
                    break
            if found:
                break
    if found is None:
        w = shutil.which("blender")
        if w:
            found = Path(w)
    if found is None:
        if soft:
            return None
        sys.exit("找不到 Blender，请设置环境变量 BLENDER_EXE 指向 blender.exe")
    return found


def find_sdcli(override=None, soft=False):
    """sd-cli 优先级：--sd-cli 参数 > SD_CLI 环境变量 > vendor/sdcpp-ext/ > PATH。"""
    cand = []
    if override:
        cand.append(Path(override))
    env = os.environ.get("SD_CLI")
    if env:
        cand.append(Path(env))
    cand.append(DEFAULT_SDCLI)
    for c in cand:
        if c.is_file():
            return c
    w = shutil.which("sd-cli") or shutil.which("sd-cli.exe")
    if w:
        return Path(w)
    vendor = ROOT / "vendor"
    if vendor.is_dir():
        for p in sorted(vendor.rglob("sd-cli.exe")):
            return p
    if soft:
        return None
    sys.exit("找不到 sd-cli.exe：请放到 vendor/sdcpp-ext/ 下（与 ggml-*.dll、stable-diffusion.dll 同目录），"
             "或用 --sd-cli / 环境变量 SD_CLI 指定完整路径")


# ---------------- 工具函数 ----------------

def fmt_cmd(cmd):
    """打印命令时给含空格的参数加引号，方便复制到报告。"""
    def q(s):
        s = str(s)
        return '"' + s + '"' if (" " in s or "\t" in s) else s
    return " ".join(q(c) for c in cmd)


def file_mb(p):
    try:
        return os.path.getsize(p) / 1048576.0
    except OSError:
        return -1.0


def check_model(path, what, dry_run=False):
    p = Path(path)
    if p.is_file():
        print("[pipeline] %s: %s (%.2f MB)" % (what, p, file_mb(p)))
        return True
    msg = "%s 缺失: %s（SD-Turbo 约 1.70GB / ControlNet 约 695MB）" % (what, p)
    if dry_run:
        print("[pipeline] [dry-run] 警告： " + msg)
        return False
    sys.exit(msg)


def find_depth(outdir, name):
    """Blender File Output 节点会在文件名后加帧号：优先 <name>_depth_*.png，其次 <name>_depth.png。"""
    cands = sorted(Path(outdir).glob("%s_depth_*.png" % name))
    if cands:
        return cands[-1]
    alt = Path(outdir) / ("%s_depth.png" % name)
    if alt.is_file():
        return alt
    return None


def depth_matches_res(path, res):
    """复用已有深度图前检查分辨率是否与本次 --res 一致（不一致必须重导出）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return w == res and h == res
    except Exception:
        return True  # 检查不了就当作可用


def ensure_rgb(path):
    """指南FAQ：control image load failed —— 相对路径按 sd-cli 工作目录解析；图为单通道灰度会加载失败。
    这里统一转绝对路径 + 灰度转 RGB。"""
    p = Path(path).resolve()
    try:
        from PIL import Image
        with Image.open(p) as im:
            if im.mode != "RGB":
                new = p.with_name(p.stem + "_rgb.png")
                im.convert("RGB").save(new)
                print("[pipeline] 深度图 %s 为 %s 模式，已转 RGB -> %s" % (p.name, im.mode, new.name))
                return new
    except Exception as e:
        print("[pipeline] 无法检查深度图模式(%s)，按原样传给 sd-cli" % e)
    return p


# ---------------- 步骤 1：Blender 无头导出 ----------------

def blender_export(blender, blend, outdir, name, res, samples):
    cmd = [str(blender), "-b", str(blend), "-P", str(BLENDER_EXPORT_SCRIPT),
           "--", "--outdir", str(outdir), "--name", name,
           "--res", str(res), "--samples", str(samples)]
    print("[pipeline] 步骤1/2 Blender 导出 depth/normal/clay：")
    print("[pipeline]   " + fmt_cmd(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit("Blender 导出失败，返回码 %d" % r.returncode)
    return dt


# ---------------- sd-cli 输出的耗时解析 ----------------

def parse_sdcli_timing(text):
    """尽量从 sd-cli 输出解析各阶段耗时（秒）。不同版本措辞有差异：
    匹配不到时以墙钟时间为准，完整日志会存 <name>_sdcli.log 供报告引用。"""
    out = {}
    for line in text.replace("\r", "\n").splitlines():
        times = []
        for m in TIME_RE.finditer(line):
            v = float(m.group(1))
            if m.group(2).lower().startswith("ms"):
                v /= 1000.0
            times.append(v)
        if not times:
            continue
        low = line.lower()
        if "step" in low and "/" in low:      # 单步进度行（如 step 2/4 ...）不计入
            continue
        if "vae" in low or "decode" in low or "decod" in low:
            key = "vae_s"
        elif "sampling" in low or "sample" in low or "diffusion" in low:
            key = "sampling_s"
        elif "clip" in low or "prompt" in low or "encode" in low or "text" in low:
            key = "encode_s"
        elif "load" in low:
            key = "load_s"
        elif "total" in low or "elapsed" in low:
            key = "reported_total_s"
        else:
            continue
        out[key] = max(out.get(key, 0.0), max(times))   # 用最大值，避免把单步耗时当成总耗时
    return out


def run_streaming(cmd, cwd):
    """实时转发 sd-cli 输出，同时收集完整日志（长任务期间不至于黑屏等待）。"""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    chunks = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        chunks.append(line)
    proc.stdout.close()
    proc.wait()
    return proc.returncode, time.perf_counter() - t0, "".join(chunks)


def build_sdcli_cmd(args, sdcli, depth, out_png):
    cmd = [str(sdcli),
           "-m", str(Path(args.model).resolve()),
           "-W", str(args.res), "-H", str(args.res),
           "--steps", str(args.steps),
           "--cfg-scale", str(args.cfg),
           "--seed", str(args.seed),
           "--threads", str(args.threads),
           "-p", args.prompt,
           "-n", args.negative,
           "-o", str(out_png)]
    if not args.no_control:
        cmd += ["--control-net", str(Path(args.control_net).resolve()),
                "--control-image", str(depth),          # 已是绝对路径（指南FAQ）
                "--control-strength", str(args.control_strength)]
    if args.sampling_method:
        cmd += [SAMPLING_FLAG, args.sampling_method]
    if args.init_img:
        cmd += ["-i", str(Path(args.init_img).resolve()),
                "--strength", str(args.strength)]       # 低强度 img2img：光色统一
    cmd += list(args.sd_extra or [])
    return cmd


# ---------------- 汇总（报告第五、六节直接抄表） ----------------

def summarize(args):
    outdir = Path(args.outdir).resolve()
    files = sorted(outdir.glob("*_timing.json"))
    if not files:
        print("[pipeline] %s 下没有 *_timing.json，先跑几次生成再汇总。" % outdir)
        return
    hdr = ("%-22s %-6s %-4s %-5s %-4s %-5s %-14s %-9s %-9s %-9s %s"
           % ("运行(name)", "分辨率", "步数", "CFG", "CN", "种子", "采样器",
              "Blender(s)", "采样(s)", "VAE(s)", "总耗时(s)"))
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print("[pipeline] 跳过损坏的 %s (%s)" % (f.name, e))
            continue
        print("%-22s %-6s %-4s %-5s %-4s %-5s %-14s %-9s %-9s %-9s %s" % (
            j.get("name", f.stem), j.get("res", "-"), j.get("steps", "-"),
            j.get("cfg", "-"), j.get("control_strength", "-"), j.get("seed", "-"),
            str(j.get("sampler", "-")),
            ("%.1f" % j["blender_export_s"]) if j.get("blender_export_s") else "-",
            ("%.1f" % j["sampling_s"]) if j.get("sampling_s") else "-",
            ("%.1f" % j["vae_s"]) if j.get("vae_s") else "-",
            ("%.1f" % j["wall_total_s"]) if j.get("wall_total_s") else "-"))
    print("[pipeline] 明细见各 %s/<name>_timing.json；采样/VAE 为 sd-cli 日志解析值，" % outdir.name)
    print("[pipeline] 若为 '-' 请到 <name>_sdcli.log 里手动查（不同版本 sd-cli 措辞不同）。")


# ---------------- 主流程 ----------------

def parse_args():
    p = argparse.ArgumentParser(
        description="白膜转写实一键流水线（Blender 导出 -> sd-cli 纯CPU深度控制生成）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None, help="输出文件名前缀，默认 test<res>，如 test512")
    p.add_argument("--res", type=int, default=512, choices=(512, 768), help="输出分辨率边长")
    p.add_argument("--steps", type=int, default=4, help="采样步数（Turbo 4~8）")
    p.add_argument("--cfg", type=float, default=1.2, help="CFG 强度（Turbo 保持 1.0~1.5）")
    p.add_argument("--control-strength", type=float, default=0.9, help="ControlNet 强度")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--threads", type=int, default=-1, help="-1 占满所有核心")
    p.add_argument("--control-image", default=None,
                   help="跳过 Blender，直接使用已有深度图（建议绝对路径）")
    p.add_argument("--sampling-method", default=None,
                   help="采样器，如 euler_a/euler/dpm++；不传则用 sd-cli 默认(euler a)")
    p.add_argument("--init-img", default=None, help="img2img 初始图（拓展：光色统一）")
    p.add_argument("--strength", type=float, default=0.35,
                   help="img2img 强度（需配合 --init-img，指南取 0.35）")
    p.add_argument("--blend", default=str(DEFAULT_BLEND), help="白模场景 .blend 文件")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="输出目录")
    p.add_argument("--samples", type=int, default=64, help="Blender Cycles 采样数")
    p.add_argument("--force-export", action="store_true", help="忽略已存在的深度图，强制重跑 Blender 导出")
    p.add_argument("--no-control", action="store_true", help="不挂 ControlNet（做无控制对照）")
    p.add_argument("--model", default=str(DEFAULT_MODEL), help="主模型 GGUF 路径")
    p.add_argument("--control-net", default=str(DEFAULT_CONTROLNET), help="深度 ControlNet GGUF 路径")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="正向提示词")
    p.add_argument("--negative", default=DEFAULT_NEGATIVE, help="负向提示词")
    p.add_argument("--sd-cli", default=None, help="sd-cli.exe 完整路径")
    p.add_argument("--sd-extra", action="append", default=[],
                   help="透传给 sd-cli 的额外参数，可多次出现（应对不同版本 flag 差异）")
    p.add_argument("--summarize", action="store_true", help="不生成，只汇总 output/*_timing.json")
    p.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不实际运行")
    return p.parse_args()


def main():
    args = parse_args()
    if args.summarize:
        summarize(args)
        return

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.name is None:
        args.name = "test%d" % args.res

    t_all0 = time.perf_counter()

    # ---------- 步骤 1：准备深度图 ----------
    blender_s = None
    depth = None
    if args.control_image:
        depth = Path(args.control_image)
        if not depth.is_file():
            sys.exit("[pipeline] --control-image 不存在: %s" % depth)
        depth = ensure_rgb(depth)
        print("[pipeline] 跳过 Blender，使用已有深度图: %s" % depth)
    else:
        old = find_depth(outdir, args.name)
        if old and not args.force_export and depth_matches_res(old, args.res):
            depth = ensure_rgb(old)
            print("[pipeline] 复用已有深度图 %s（加 --force-export 可强制重导出）" % depth.name)
        else:
            blend = Path(args.blend)
            if not blend.is_file():
                sys.exit("[pipeline] 找不到白模场景 %s。先运行步骤1: python blender/make_demo_scene.py，"
                         "或用 --blend 指定你的 .blend，或用 --control-image 直接给深度图。" % blend)
            if args.dry_run:
                blender = find_blender(soft=True) or "blender"
                print("[pipeline] [dry-run] 将执行 Blender 导出（--force-export 可强制重导出）：")
                print("[pipeline]   " + fmt_cmd(
                    [str(blender), "-b", str(blend), "-P", str(BLENDER_EXPORT_SCRIPT),
                     "--", "--outdir", str(outdir), "--name", args.name,
                     "--res", str(args.res), "--samples", str(args.samples)]))
                depth = outdir / ("%s_depth_0001.png" % args.name)   # 预期产物，仅用于展示命令
            else:
                blender = find_blender()
                blender_s = blender_export(blender, blend, outdir, args.name, args.res, args.samples)
                depth = find_depth(outdir, args.name)
                if depth is None:
                    sys.exit("[pipeline] Blender 结束但没找到 %s_depth_*.png，请检查导出日志" % args.name)
                print("[pipeline] Blender 导出耗时 %.1f s，深度图: %s" % (blender_s, depth.name))

    out_png = outdir / ("%s_photoreal.png" % args.name)
    log_path = outdir / ("%s_sdcli.log" % args.name)
    json_path = outdir / ("%s_timing.json" % args.name)

    # ---------- 步骤 2：sd-cli 生成 ----------
    sdcli = find_sdcli(args.sd_cli, soft=args.dry_run)
    if sdcli is None:
        sdcli = "sd-cli"
        print("[pipeline] [dry-run] 找不到 sd-cli.exe，实际运行前请放到 vendor/sdcpp-ext/。")
    cmd = build_sdcli_cmd(args, sdcli, depth, out_png)
    mode = "img2img" if args.init_img else "txt2img"

    print("[pipeline] 步骤2/2 sd-cli 生成写实图（%s, %s）：" % (mode, "ControlNet=%.2f" % args.control_strength
                                                  if not args.no_control else "无ControlNet"))
    print("[pipeline]   " + fmt_cmd(cmd))

    check_model(args.model, "主模型(SD-Turbo Q4_0 GGUF，约1.70GB)", dry_run=args.dry_run)
    if not args.no_control:
        check_model(args.control_net, "ControlNet(SD2.1 depth GGUF，约695MB；不能换 SD1.5 版本："
                                      "交叉注意力 768 vs 1024 会报形状不匹配)", dry_run=args.dry_run)

    if args.dry_run:
        print("[pipeline] [dry-run] 预期输出: %s" % out_png)
        print("[pipeline] [dry-run] 结束：未执行任何命令。")
        return

    if not Path(depth).is_file():
        sys.exit("[pipeline] 深度图不存在: %s" % depth)

    rc, sdcli_wall, raw = run_streaming(cmd, cwd=str(ROOT))
    log_path.write_text(raw, encoding="utf-8")
    if rc != 0:
        sys.exit("[pipeline] sd-cli 失败，返回码 %d；完整日志: %s" % (rc, log_path))
    if not out_png.is_file():
        sys.exit("[pipeline] sd-cli 正常退出但没有生成 %s，请查看日志 %s" % (out_png, log_path))

    parsed = parse_sdcli_timing(raw)
    wall_total = time.perf_counter() - t_all0

    record = {
        "name": args.name, "mode": mode,
        "res": args.res, "steps": args.steps, "cfg": args.cfg,
        "control_strength": None if args.no_control else args.control_strength,
        "seed": args.seed, "threads": args.threads,
        "sampler": args.sampling_method or "euler a(sd-cli 默认)",
        "prompt": args.prompt, "negative": args.negative,
        "model": str(Path(args.model).resolve()), "model_mb": round(file_mb(args.model), 1),
        "control_net": None if args.no_control else str(Path(args.control_net).resolve()),
        "controlnet_mb": None if args.no_control else round(file_mb(args.control_net), 1),
        "blender_export_s": round(blender_s, 1) if blender_s is not None else None,
        "sampling_s": round(parsed["sampling_s"], 1) if parsed.get("sampling_s") else None,
        "vae_s": round(parsed["vae_s"], 1) if parsed.get("vae_s") else None,
        "encode_s": round(parsed["encode_s"], 1) if parsed.get("encode_s") else None,
        "sdcli_wall_s": round(sdcli_wall, 1),
        "wall_total_s": round(wall_total, 1),
        "sdcli_reported": parsed,
        "depth_image": str(depth),
        "output": str(out_png),
        "log": str(log_path),
        "commands": {"blender": None if args.control_image else fmt_cmd(
            ["blender", "-b", str(Path(args.blend).resolve()), "-P", str(BLENDER_EXPORT_SCRIPT),
             "--", "--outdir", str(outdir), "--name", args.name,
             "--res", str(args.res), "--samples", str(args.samples)]),
            "sd_cli": fmt_cmd(cmd)},
        "note": "sampling/vae 来自 sd-cli 日志解析，解析不到时看 sdcli_reported 与 <name>_sdcli.log",
    }
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("[pipeline] ================== 运行摘要（报告第五、六节可直接引用） ==================")
    print("[pipeline] 运行        : %s (%s)" % (args.name, mode))
    print("[pipeline] 参数       : %dx%d, steps=%d, cfg=%.2f, control=%.2f, seed=%d, sampler=%s"
          % (args.res, args.res, args.steps, args.cfg,
             -1 if args.no_control else args.control_strength, args.seed, record["sampler"]))
    print("[pipeline] Blender导出 : %s s" % (("%.1f" % blender_s) if blender_s is not None else "跳过/复用"))
    print("[pipeline] 采样耗时   : %s s%s" % (record["sampling_s"] or "-",
          "" if record["sampling_s"] else "（日志未解析到，请查 %s）" % log_path.name))
    print("[pipeline] VAE 解码   : %s s" % (record["vae_s"] or "-"))
    print("[pipeline] sd-cli 墙钟: %.1f s" % sdcli_wall)
    print("[pipeline] 全程总耗时: %.1f s" % wall_total)
    print("[pipeline] 输出      : %s (%.2f MB)" % (out_png, file_mb(out_png)))
    print("[pipeline] 计时记录  : %s" % json_path)
    print("[pipeline] 提示      : python scripts/run_pipeline.py --summarize 可汇总多次运行做参数对比")


if __name__ == "__main__":
    main()
