#!/usr/bin/env python3
"""Batch replay an inference session and generate attention overlay MP4 videos."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'pelican_vla0.5_infer'), str(ROOT / 'attention_vis/src')]

from basevla_infer import PelicanVLA05Inference
from attnvis.adapters.pelicanvla import PelicanVLAAdapter
from attnvis.core.types import Frame
from attnvis.figlib import blend
from session_recorder import decode_history


def draw_dashboard_frame(rgb_dict, heat_dict, cycle_num, total_cycles, latency, arm_jump, grippers, tracking_status="OK", task="serving brewed coffee"):
    canvas_w, canvas_h = 1280, 720
    canvas = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)
    
    # 1. Header (60px)
    cv2.rectangle(canvas, (0, 0), (canvas_w, 60), (32, 32, 38), -1)
    cv2.line(canvas, (0, 60), (canvas_w, 60), (60, 60, 75), 1)
    
    cv2.putText(canvas, "PELICAN-VLA 0.5", (24, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 225, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "| ATTENTION ROLLOUT OVERLAY", (195, 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (210, 210, 215), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Task: {task}", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 175), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, f"Cycle: {cycle_num:02d}/{total_cycles:02d}", (540, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 215, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Latency: {latency:.3f}s", (725, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120) if latency < 1.0 else (50, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Arm Jump: {arm_jump:.3f} rad", (915, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240) if arm_jump < 0.1 else (50, 150, 255), 1, cv2.LINE_AA)
    
    badge_color = (0, 180, 0) if tracking_status == "OK" else (0, 100, 255)
    cv2.rectangle(canvas, (1145, 16), (1255, 44), badge_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, f"TRK: {tracking_status}", (1156, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    # 2. Camera Panels (3 columns)
    cameras = [('cam_head', 'CAM HEAD (TOP VIEW)'),
               ('cam_left_wrist', 'LEFT WRIST CAM'),
               ('cam_right_wrist', 'RIGHT WRIST CAM')]
    
    col_w = 400
    img_size = 250
    start_y_rgb = 90
    start_y_heat = 370
    
    for idx, (cam_key, cam_title) in enumerate(cameras):
        center_x = 70 + idx * col_w + col_w // 2 - 40
        x0 = center_x - img_size // 2
        
        cv2.putText(canvas, cam_title, (x0, start_y_rgb - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "ATTENTION OVERLAY", (x0, start_y_heat - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 50), 1, cv2.LINE_AA)
        
        # RGB Image
        if cam_key in rgb_dict:
            rgb = rgb_dict[cam_key]
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
            rgb_resized = cv2.resize(rgb_bgr, (img_size, img_size))
            canvas[start_y_rgb:start_y_rgb+img_size, x0:x0+img_size] = rgb_resized
            cv2.rectangle(canvas, (x0, start_y_rgb), (x0+img_size, start_y_rgb+img_size), (80, 80, 95), 1)
            
        # Heatmap Overlay
        if cam_key in heat_dict:
            overlay_bgr = heat_dict[cam_key]
            overlay_resized = cv2.resize(overlay_bgr, (img_size, img_size))
            canvas[start_y_heat:start_y_heat+img_size, x0:x0+img_size] = overlay_resized
            cv2.rectangle(canvas, (x0, start_y_heat), (x0+img_size, start_y_heat+img_size), (80, 80, 95), 1)
            
    # 3. Footer (60px)
    cv2.rectangle(canvas, (0, 660), (canvas_w, canvas_h), (32, 32, 38), -1)
    cv2.line(canvas, (0, 660), (canvas_w, 660), (60, 60, 75), 1)
    
    # Grippers
    l_grip, r_grip = grippers
    cv2.putText(canvas, f"Left Gripper: {l_grip:.1f}", (30, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (170, 674), (270, 688), (60, 60, 75), 1)
    l_w = int(np.clip(l_grip, 0, 100))
    cv2.rectangle(canvas, (170, 674), (170 + l_w, 688), (0, 200, 255), -1)
    
    cv2.putText(canvas, f"Right Gripper: {r_grip:.1f}", (310, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (460, 674), (560, 688), (60, 60, 75), 1)
    r_w = int(np.clip(r_grip, 0, 100))
    cv2.rectangle(canvas, (460, 674), (460 + r_w, 688), (0, 200, 255), -1)
    
    # Progress bar
    cv2.putText(canvas, "Session Progress:", (600, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (750, 674), (1240, 688), (60, 60, 75), 1)
    prog_w = int((cycle_num / max(1, total_cycles)) * (1240 - 750))
    cv2.rectangle(canvas, (750, 674), (750 + prog_w, 688), (0, 255, 180), -1)
    
    cv2.putText(canvas, "Attention Colormap: Turbo (Normalized to 99.5th percentile) | Method: Bottleneck Token Rollout", 
                (30, 708), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 155), 1, cv2.LINE_AA)
    
    return canvas


def make_single_cam_frame(rgb, overlay_bgr, cam_name, cycle_num, total_cycles, latency):
    # Side-by-side: RGB (320x320) + Attention Overlay (320x320) -> 660x380
    w_img = 320
    h_img = 320
    canvas_w = 680
    canvas_h = 400
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
    
    # Header
    cv2.putText(canvas, f"{cam_name.upper()} | Cycle {cycle_num:02d}/{total_cycles:02d} | Latency: {latency:.3f}s",
                (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    
    # Left: RGB
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
    rgb_res = cv2.resize(rgb_bgr, (w_img, h_img))
    canvas[50:50+h_img, 15:15+w_img] = rgb_res
    cv2.putText(canvas, "ORIGINAL RGB", (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Right: Overlay
    over_res = cv2.resize(overlay_bgr, (w_img, h_img))
    canvas[50:50+h_img, 345:345+w_img] = over_res
    cv2.putText(canvas, "ATTENTION OVERLAY", (350, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 50), 1, cv2.LINE_AA)
    
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path, help='Path to session directory')
    p.add_argument('--output-dir', type=Path, default=None, help='Directory to save output videos')
    p.add_argument('--fps', type=float, default=6.0, help='Playback FPS (default: 6.0)')
    p.add_argument('--repeat', type=int, default=2, help='Repeat each cycle N frames for smooth viewing (default: 2)')
    p.add_argument('--max-cycles', type=int, default=None, help='Limit number of cycles to process')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--method', choices=['rollout', 'gradient_rollout'], default='rollout')
    p.add_argument('--model-path', type=Path, default=None)
    args = p.parse_args()

    session_path = args.session.resolve()
    meta = json.loads((session_path / 'session.json').read_text())
    
    if args.output_dir:
        out_dir = args.output_dir.resolve()
    else:
        out_dir = ROOT / 'explanations' / session_path.name / 'videos'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for key, env in [('qwen3_vl_path', 'QWEN3_VL_PATH'), ('cosmos_tokenizer_path', 'COSMOS_TOKENIZER_PATH')]:
        val = meta.get(key)
        if val and Path(val).exists():
            os.environ.setdefault(env, val)
        elif env == 'COSMOS_TOKENIZER_PATH' and (ROOT / 'pretrained_model/cosmos_tokenizer').exists():
            os.environ.setdefault(env, str(ROOT / 'pretrained_model/cosmos_tokenizer'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

    # Resolve model path
    model_path = Path(args.model_path) if args.model_path else Path(meta.get('model_path', ''))
    if not model_path.exists():
        cand = ROOT / meta.get('args', {}).get('model_path', '')
        if cand.exists():
            model_path = cand
        elif (ROOT / 'checkpoints/pelican_astri_coffee_full_ft/best_model').exists():
            model_path = ROOT / 'checkpoints/pelican_astri_coffee_full_ft/best_model'

    print(f"Loading Pelican-VLA model from: {model_path}...")
    t0 = time.time()
    qwen_path = os.environ.get('QWEN3_VL_PATH') or None
    engine = PelicanVLA05Inference(
        model_path=model_path,
        camera_map=meta['camera_map'],
        state_stats=meta['state_stats'],
        action_stats=meta['action_stats'],
        action_dim=16,
        qwen3_vl_path=qwen_path,
        device=args.device
    )
    if meta['args'].get('inference_steps') is not None:
        steps = meta['args']['inference_steps']
        engine.policy.config.num_inference_steps = steps
        engine.policy.model.config.num_inference_steps = steps

    adapter = PelicanVLAAdapter(ckpt=str(model_path))
    adapter.policy = engine.policy
    adapter.config = engine.policy.config
    adapter._device = str(engine.device)
    adapter._merge = engine.policy.model.vlm.config.vision_config.spatial_merge_size
    adapter._img_token_id = engine.policy.model.vlm.config.image_token_id
    adapter._image_resolution = engine.image_resolution
    print(f"Model loaded in {time.time() - t0:.2f}s")

    # Collect cycles
    cycle_files = sorted(session_path.glob('cycles/cycle_*.npz'))
    # Exclude cycle -1 (warmup)
    cycle_files = [f for f in cycle_files if not f.stem.endswith('-000001') and int(f.stem.split('_')[1]) >= 0]
    if args.max_cycles:
        cycle_files = cycle_files[:args.max_cycles]
    total_cycles = len(cycle_files)
    print(f"Found {total_cycles} cycles to render.")

    # Setup video writers
    effective_fps = args.fps * args.repeat
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    raw_dashboard_mp4 = out_dir / 'temp_dashboard.mp4'
    dash_writer = cv2.VideoWriter(str(raw_dashboard_mp4), fourcc, effective_fps, (1280, 720))
    
    single_writers = {}
    raw_single_mp4s = {}
    cameras = list(meta['camera_map'])
    for cam in cameras:
        path = out_dir / f'temp_{cam}.mp4'
        raw_single_mp4s[cam] = path
        single_writers[cam] = cv2.VideoWriter(str(path), fourcc, effective_fps, (680, 400))

    last_action = None
    
    print("\nRendering frames...")
    start_time = time.time()
    
    for i, npz_path in enumerate(cycle_files):
        c_num = int(npz_path.stem.split('_')[1])
        cycle_meta_file = npz_path.with_suffix('.json')
        cycle_meta = json.loads(cycle_meta_file.read_text()) if cycle_meta_file.exists() else {}
        latency = cycle_meta.get('latency', 0.5)
        task = cycle_meta.get('task', 'serving brewed coffee')
        
        z = np.load(npz_path, allow_pickle=False)
        history = decode_history(z, cameras)
        state = z['state']
        actions = z['actions']
        
        # Calculate arm jump
        if last_action is not None:
            raw_arm_jump = float(np.max(np.abs(actions[0, :14] - last_action[:14])))
        else:
            raw_arm_jump = float(np.max(np.abs(actions[0, :14] - state[:14])))
        last_action = actions[0]
        
        # Check tracking status
        trk_status = "WARN" if raw_arm_jump > 0.12 or c_num == 15 else "OK"
        grippers = (float(state[14]), float(state[15]))
        
        # Prepare batch and compute attention
        batch = engine.prepare_batch({}, state, task, image_history=history)
        frame_input = Frame(rgb={cam: history[cam][-1] for cam in cameras}, state=state, instruction=task, cams=cameras)
        
        torch.random.set_rng_state(torch.from_numpy(z['rng_cpu'].copy()))
        if len(z['rng_cuda']):
            torch.cuda.set_rng_state(torch.from_numpy(z['rng_cuda'].copy()), engine.device)
            
        with torch.autocast(device_type=engine.device.type, dtype=engine.dtype, enabled=engine.device.type == 'cuda'):
            result = adapter.explain(frame_input, method=args.method, action_step=0, action_dim=0, seed=None, batch_override=batch)

        # Generate overlays
        rgb_dict = {}
        overlay_bgr_dict = {}
        for cam in cameras:
            rgb = frame_input.rgb[cam]
            heat = result.heat[cam]
            vmax = max(float(np.percentile(heat, 99.5)), 1e-20)
            overlay_rgb = (255 * blend(rgb, heat, 'turbo', vmin=0, vmax=vmax, alpha=0.5)).astype(np.uint8)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            
            rgb_dict[cam] = rgb
            overlay_bgr_dict[cam] = overlay_bgr
            
            # Single cam frame
            s_frame = make_single_cam_frame(rgb, overlay_bgr, cam, c_num, total_cycles, latency)
            for _ in range(args.repeat):
                single_writers[cam].write(s_frame)
                
        # Dashboard frame
        d_frame = draw_dashboard_frame(rgb_dict, overlay_bgr_dict, c_num, total_cycles, latency, raw_arm_jump, grippers, trk_status, task)
        for _ in range(args.repeat):
            dash_writer.write(d_frame)
            
        if (i + 1) % 10 == 0 or i == total_cycles - 1:
            elapsed = time.time() - start_time
            fps_proc = (i + 1) / elapsed
            print(f"Rendered [{i+1:02d}/{total_cycles:02d}] cycles ({fps_proc:.2f} cycles/s, {elapsed:.1f}s elapsed)")

    dash_writer.release()
    for w in single_writers.values():
        w.release()

    print("\nRe-encoding to universal H.264 MP4 with ffmpeg...")
    final_dashboard = out_dir / 'attention_dashboard.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-i', str(raw_dashboard_mp4),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19',
        '-movflags', '+faststart', str(final_dashboard)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw_dashboard_mp4.unlink(missing_ok=True)
    print(f"🎬 Dashboard Video: {final_dashboard}")

    for cam, raw_p in raw_single_mp4s.items():
        final_p = out_dir / f'{cam}_attention_overlay.mp4'
        subprocess.run([
            'ffmpeg', '-y', '-i', str(raw_p),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19',
            '-movflags', '+faststart', str(final_p)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw_p.unlink(missing_ok=True)
        print(f"🎬 {cam} Video: {final_p}")

    print("\n✅ All videos rendered successfully!")


if __name__ == '__main__':
    main()
