#!/usr/bin/env python3
"""Replay saved robot observations through three checkpoints and render attention."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'pelican_vla0.5_infer'), str(ROOT/'attention_vis/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('COSMOS_TOKENIZER_PATH', str(ROOT/'pretrained_model/cosmos_tokenizer'))
import cv2
import numpy as np
import torch
from basevla_infer import PelicanVLA05Inference
from session_recorder import decode_history
from attnvis.adapters.pelicanvla import PelicanVLAAdapter
from attnvis.core.types import Frame
from attnvis.figlib import blend


def caption(canvas, text, x, y, scale=.5):
    cv2.putText(canvas, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240,240,240), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    meta = json.loads((args.session/'session.json').read_text())
    paths = [p for p in sorted(args.session.glob('cycles/*.npz')) if int(p.stem.split('_')[1]) >= 0]
    if not paths:
        raise ValueError('No inference cycles')
    cycles = [json.loads(p.with_suffix('.json').read_text()) for p in paths]
    times = np.array([c['started_monotonic'] for c in cycles])
    if not np.all(np.diff(times) > 0):
        raise ValueError('Non-monotonic cycle timestamps')
    times -= times[0]
    cams = list(meta['camera_map'])
    checkpoints = [ROOT/'checkpoints/pelican_astri_coffee_full_ft/best_model',
                   ROOT/'checkpoints/coffee_aug_20261020_2gpu/best_model',
                   ROOT/'checkpoints/coffee_aug_20261020_2gpu/checkpoint-076250']
    names = ['legacy_best', 'augmented_best_038750', 'augmented_076250']
    labels = ['Legacy best', 'Augmented best | step 38750', 'Augmented | step 76250']
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.cuda.set_device(torch.device(args.device))
    maps, rgbs, reports = [], [], []
    for model_index, checkpoint in enumerate(checkpoints):
        stats = json.loads((checkpoint/'stats.json').read_text())
        state_stats = meta['state_stats'] if model_index == 0 else stats['state']
        action_stats = meta['action_stats'] if model_index == 0 else stats['action']
        engine = PelicanVLA05Inference(model_path=checkpoint, camera_map=meta['camera_map'],
            state_stats=state_stats, action_stats=action_stats, action_dim=16, device=args.device)
        for param in engine.policy.parameters():
            param.requires_grad_(False)
        steps = meta['args'].get('inference_steps')
        if steps is not None:
            engine.policy.config.num_inference_steps = steps
            engine.policy.model.config.num_inference_steps = steps
        adapter = PelicanVLAAdapter(ckpt=str(checkpoint))
        adapter.policy = engine.policy
        adapter.config = engine.policy.config
        adapter._device = str(engine.device)
        adapter._merge = engine.policy.model.vlm.config.vision_config.spatial_merge_size
        adapter._img_token_id = engine.policy.model.vlm.config.image_token_id
        adapter._image_resolution = engine.image_resolution
        heats, tokens, replay_errors = [], [], []
        start = time.monotonic()
        for n, (path, cycle) in enumerate(zip(paths, cycles)):
            with np.load(path, allow_pickle=False) as archive:
                history = decode_history(archive, cams)
                rgb = np.stack([history[c][-1] for c in cams])
                if model_index == 0:
                    rgbs.append(rgb)
                else:
                    np.testing.assert_array_equal(rgb, rgbs[n])
                def restore_rng():
                    torch.random.set_rng_state(torch.from_numpy(archive['rng_cpu'].copy()))
                    if len(archive['rng_cuda']):
                        torch.cuda.set_rng_state(torch.from_numpy(archive['rng_cuda'].copy()), engine.device)
                if model_index == 0:
                    restore_rng()
                    predicted = engine.infer({}, archive['state'], cycle['task'], image_history=history)
                    error = float(np.max(np.abs(predicted-archive['actions']) / engine._safe_std(engine.action_std[:16])))
                    tolerance = max(1e-5, 4*torch.finfo(engine.dtype).eps)
                    if not np.isfinite(error) or error > tolerance:
                        raise ValueError(f'Legacy replay mismatch at {path.stem}: {error} > {tolerance}')
                    replay_errors.append(error)
                batch = engine.prepare_batch({}, archive['state'], cycle['task'], image_history=history)
                frame = Frame(rgb=dict(zip(cams, rgb)), state=archive['state'], instruction=cycle['task'], cams=cams)
                restore_rng()
                with torch.autocast(device_type=engine.device.type, dtype=engine.dtype):
                    result = adapter.explain(frame, method='rollout', action_step=0, seed=None, batch_override=batch)
                heat = np.stack([result.heat[c] for c in cams])
                if not np.isfinite(heat).all() or heat.max() <= 0:
                    raise ValueError(f'Invalid attention at {path.stem}')
                heats.append(heat)
                tokens.append(np.stack([result.raw['token_maps'][c] for c in cams]))
            if (n+1)%16 == 0:
                print(f'{names[model_index]}: {n+1}/{len(paths)} cycles, {time.monotonic()-start:.1f}s', flush=True)
        heat = np.stack(heats)
        np.savez_compressed(args.output/f'{names[model_index]}_attention.npz', heat=heat, tokens=np.stack(tokens), cycle_ids=[c['cycle'] for c in cycles])
        maps.append(heat)
        reports.append(dict(checkpoint=str(checkpoint), state_stats=state_stats,
            normalization_source='recorded session' if model_index==0 else 'checkpoint stats.json',
            inference_steps=engine.policy.config.num_inference_steps,
            replay_errors=replay_errors, replay_tolerance=tolerance if model_index==0 else None))
        del adapter, engine, batch, result
        gc.collect()
        torch.cuda.empty_cache()
    rgbs = np.stack(rgbs)
    np.savez_compressed(args.output/'inputs.npz', rgb=rgbs, cycle_ids=[c['cycle'] for c in cycles], times=times)
    limits = [max(float(np.percentile(np.concatenate([m[:,c].ravel() for m in maps]),99.5)),1e-20) for c in range(len(cams))]
    # Sample-and-hold cycle observations at recorded timestamps, not invented camera frames.
    fps = 10
    final_hold = float(np.median(np.diff(times)))
    playback = np.arange(0, times[-1]+final_hold, 1/fps)
    indices = np.searchsorted(times, playback, side='right')-1
    specs = {'comparison_three_checkpoints': [None,0,1,2], **{name:[None,i] for i,name in enumerate(names)}}
    probes = {}
    for name, rows in specs.items():
        size = (1060, 110+245*len(rows)+30)
        tmp = args.output/f'{name}.temporary.mp4'
        out = args.output/f'{name}.mp4'
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            raise RuntimeError('Video writer failed')
        try:
            for k,n in enumerate(indices):
                canvas = np.full((size[1],size[0],3),22,np.uint8)
                caption(canvas, 'RECORDED ROBOT SESSION | THREE CHECKPOINTS',20,27,.65)
                caption(canvas, f'Cycle {cycles[n]["cycle"]} / {cycles[-1]["cycle"]} | t={playback[k]:.1f}s | {cycles[n]["task"]}',20,51)
                caption(canvas, 'Same recorded history, state and RNG | cycle images held until next inference',20,75,.48)
                for row,m in enumerate(rows):
                    y = 110+245*row
                    for c,cam in enumerate(cams):
                        x = 20+345*c
                        caption(canvas, cam+' | '+('RGB' if m is None else labels[m]), x,y-9,.39)
                        rgb = rgbs[n,c] if m is None else (255*blend(rgbs[n,c], maps[m][n,c], 'turbo', vmin=0, vmax=limits[c], alpha=.5)).astype(np.uint8)
                        canvas[y:y+224,x+40:x+264] = cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(224,224))
                caption(canvas,'Offline attention on recorded observations; new models did not drive this robot run.',20,size[1]-12,.46)
                writer.write(canvas)
                if name.startswith('comparison') and k == len(indices)//2:
                    cv2.imwrite(str(args.output/'preview.jpg'),canvas)
        finally:
            writer.release()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(tmp),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(out)],text=True))['streams'][0]
        capture = cv2.VideoCapture(str(out))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        if count != len(indices) or int(probe['nb_frames']) != count:
            raise ValueError('Video decode frame count mismatch')
        probes[name] = probe
        tmp.unlink()
        print(f'Validated {out.name}: {count} frames',flush=True)
    report = dict(session=str(args.session.resolve()), cycles=[c['cycle'] for c in cycles],
        models=reports, video_streams=probes, color_limits=limits, method='decoder rollout, action token 0, last denoising call',
        rng='Recorded CPU/CUDA RNG restored independently for each model and cycle',
        timing=dict(fps=fps, cycle_times=times.tolist(), frame_cycle_indices=indices.tolist(), final_hold_seconds=final_hold),
        caveats=['Offline comparison on legacy robot observations, not closed-loop evaluation of new models.',
                 'Each cycle image/attention is held until the next recorded cycle timestamp; final hold is median cycle interval.',
                 'Shared color limits per camera across all models and cycles. Attention is not causal attribution.'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE',args.output,flush=True)


if __name__ == '__main__':
    main()
