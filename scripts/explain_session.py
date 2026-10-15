#!/usr/bin/env python3
"""Replay a recorded Pelican cycle and generate decoder attention offline."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pelican_vla0.5_infer'),str(ROOT/'attention_vis/src')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('session',type=Path)
    p.add_argument('--cycle',type=int,default=0)
    p.add_argument('--method',choices=['rollout','gradient_rollout'],default='rollout')
    p.add_argument('--action-step',type=int,default=0)
    p.add_argument('--action-dim',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--replay-atol',type=float,default=None,
        help='Maximum action error in training-standard-deviation units (default: four dtype eps, floor 1e-5)')
    a=p.parse_args()
    meta=json.loads((a.session/'session.json').read_text())
    name=f'cycle_{a.cycle:06d}'
    cycle=json.loads((a.session/'cycles'/f'{name}.json').read_text())
    for key,env in [('qwen3_vl_path','QWEN3_VL_PATH'),('cosmos_tokenizer_path','COSMOS_TOKENIZER_PATH')]:
        if meta.get(key): os.environ.setdefault(env,meta[key])
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    import numpy as np
    import torch
    from basevla_infer import PelicanVLA05Inference
    from attnvis.adapters.pelicanvla import PelicanVLAAdapter
    from attnvis.core.types import Frame
    from PIL import Image
    from attnvis.figlib import blend
    # Fail rather than silently replay with replaced checkpoint files.
    model_path=Path(meta['model_path'])
    for file,info in meta['checkpoint_files'].items():
        stat=(model_path/file).stat()
        if stat.st_size!=info['size'] or stat.st_mtime_ns!=info['mtime_ns']:
            raise ValueError(f'Checkpoint changed since recording: {file}')
    engine=PelicanVLA05Inference(model_path=model_path,camera_map=meta['camera_map'],
        state_stats=meta['state_stats'],action_stats=meta['action_stats'],action_dim=16,
        qwen3_vl_path=os.environ.get('QWEN3_VL_PATH'),device=a.device)
    steps=meta['args'].get('inference_steps')
    if steps is not None:
        engine.policy.config.num_inference_steps=steps
        engine.policy.model.config.num_inference_steps=steps
    z=np.load(a.session/'cycles'/f'{name}.npz',allow_pickle=False)
    cams=list(meta['camera_map'])
    from session_recorder import decode_history
    history=decode_history(z,cams)
    def restore_rng():
        torch.random.set_rng_state(torch.from_numpy(z['rng_cpu'].copy()))
        if len(z['rng_cuda']):
            if engine.device.type!='cuda': raise ValueError('Recorded CUDA noise requires CUDA replay')
            torch.cuda.set_rng_state(torch.from_numpy(z['rng_cuda'].copy()),engine.device)
    restore_rng()
    actions=engine.infer({},z['state'],cycle['task'],image_history=history)
    difference=float(np.max(np.abs(actions-z['actions'])))
    std=engine._safe_std(engine.action_std[:16])
    normalized_error=float(np.max(np.abs(actions-z['actions'])/std))
    tolerance=max(1e-5,4*torch.finfo(engine.dtype).eps) if a.replay_atol is None else a.replay_atol
    if tolerance < 0: raise ValueError('replay-atol must be nonnegative')
    print(f'Replay error: raw={difference:.8g}, normalized={normalized_error:.8g}, tolerance={tolerance:.8g}')
    if not np.isfinite(normalized_error) or normalized_error > tolerance:
        raise RuntimeError('Replay exceeds normalized action tolerance; refusing misleading explanation')
    adapter=PelicanVLAAdapter(ckpt=str(model_path))
    adapter.policy=engine.policy; adapter.config=engine.policy.config
    adapter._device=str(engine.device); adapter._merge=engine.policy.model.vlm.config.vision_config.spatial_merge_size
    adapter._img_token_id=engine.policy.model.vlm.config.image_token_id
    adapter._image_resolution=engine.image_resolution
    batch=engine.prepare_batch({},z['state'],cycle['task'],image_history=history)
    frame=Frame(rgb={c:history[c][-1] for c in cams},state=z['state'],instruction=cycle['task'],cams=cams)
    restore_rng()
    with torch.autocast(device_type=engine.device.type,dtype=engine.dtype,enabled=engine.device.type=='cuda'):
        result=adapter.explain(frame,method=a.method,action_step=a.action_step,
            action_dim=a.action_dim,seed=None,batch_override=batch)
    out=a.session/'explanations'/f'{name}_{a.method}_step{a.action_step}_dim{a.action_dim}'
    out.mkdir(parents=True,exist_ok=True)
    for cam,heat in result.heat.items():
        np.savez(out/f'{cam}.npz',heat=heat,rgb=frame.rgb[cam],tokens=result.raw['token_maps'][cam])
        Image.fromarray((255*blend(frame.rgb[cam],heat,'turbo',vmin=0,
            vmax=max(float(np.percentile(heat,99.5)),1e-20),alpha=.5)).astype(np.uint8)).save(out/f'{cam}.png')
    result.raw['metadata'].update(replay_max_absolute_error=difference,
        replay_max_normalized_error=normalized_error,replay_normalized_tolerance=tolerance,
        replay_bitwise_equal=bool(np.array_equal(actions,z['actions'])),source_cycle=a.cycle)
    (out/'metadata.json').write_text(json.dumps(result.raw['metadata'],indent=2))
    print(out)

if __name__=='__main__': main()
