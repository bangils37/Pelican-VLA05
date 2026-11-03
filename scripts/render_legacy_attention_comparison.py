#!/usr/bin/env python3
"""Compare a legacy checkpoint to cached augmented-model attention on the same episode."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

from render_training_attention_video import caption
from explain_training_sample import ROOT, np, torch, LeRobotPelicanDataset, PelicanVLA05Policy, PelicanVLAAdapter, Frame, blend
import cv2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=ROOT/'explanations/coffee_aug_20261022_training_sample/videos')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/pelican_astri_coffee_full_ft/best_model')
    p.add_argument('--output',type=Path,default=ROOT/'explanations/20261015T162257+0700_797af814/dataset_episode_001_comparison')
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.reference/'manifest.json').read_text())
    source=manifest['source_sample'];episode=manifest['episode'];frames=manifest['frame_indices'];fps=manifest['video_fps']
    previous=[np.load(a.reference/f'{name}_attention.npz') for name in ['best_model','checkpoint-076250']]
    inputs=np.load(a.reference/'inputs.npz');rgbs=inputs['rgb'];subtasks=inputs['subtasks']
    for z in [inputs,*previous]:np.testing.assert_array_equal(z['frames'],frames)
    config=json.loads((Path(source['checkpoint_paths'][0])/'instruction_augmentation.json').read_text())
    torch.set_num_threads(2);torch.cuda.set_device(torch.device(a.device))
    ds=LeRobotPelicanDataset(source['dataset'],split='train',seed=config['seed'],val_ratio=config['val_ratio'],use_subtasks_prob=0)
    lookup={f:i for i,(_,ep,f,_,_) in enumerate(ds.samples) if ep==episode}
    if any(f not in lookup for f in frames):raise ValueError('Episode/frame mismatch')
    stats=json.loads((a.checkpoint/'stats.json').read_text())
    mean=np.asarray(stats['state']['mean'],dtype=np.float32);std=np.asarray(stats['state']['std'],dtype=np.float32)
    std=np.where(std<1e-4,1.,std)
    a.output.mkdir(parents=True)
    policy=PelicanVLA05Policy.from_pretrained(str(a.checkpoint),strict=True).to(a.device).eval()
    for param in policy.parameters():param.requires_grad_(False)
    adapter=PelicanVLAAdapter(ckpt=str(a.checkpoint));adapter.policy=policy;adapter.config=policy.config;adapter._device=a.device
    adapter._image_resolution=tuple(policy.config.image_resolution)
    adapter._merge=policy.model.vlm.config.vision_config.spatial_merge_size
    adapter._img_token_id=policy.model.vlm.config.image_token_id
    cams=['cam_head','cam_left_wrist','cam_right_wrist'];heats=[];tokens=[];start=time.monotonic()
    for n,f in enumerate(frames):
        i=lookup[f];row=ds.samples[i][0];sample=ds[i]
        rgb=np.stack([np.rint(sample[f'observation.images.image{k}'][1].permute(1,2,0).numpy()*255).astype(np.uint8) for k in range(3)])
        np.testing.assert_array_equal(rgb,rgbs[n])
        if sample['task']!=source['instruction']:raise ValueError('Instruction mismatch')
        raw_state=ds.raw_states[row]
        normalized=np.zeros(policy.config.max_state_dim,dtype=np.float32)
        normalized[:len(mean)]=(raw_state-mean)/std
        sample['observation.state']=torch.from_numpy(normalized)
        batch={k:v.unsqueeze(0).to(a.device) for k,v in sample.items() if isinstance(v,torch.Tensor) and k!='action'}
        frame=Frame(rgb=dict(zip(cams,rgb)),state=raw_state,instruction=source['instruction'],cams=cams)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            result=adapter.explain(frame,method='rollout',action_step=0,seed=source['seed'],batch_override=batch)
        heat=np.stack([result.heat[c] for c in cams])
        if not np.isfinite(heat).all() or heat.max()<=0:raise ValueError('Invalid attention')
        heats.append(heat);tokens.append(np.stack([result.raw['token_maps'][c] for c in cams]))
        if (n+1)%20==0 or n+1==len(frames):print(f'Legacy: {n+1}/{len(frames)} frames, {time.monotonic()-start:.1f}s',flush=True)
    legacy=np.stack(heats)
    np.savez_compressed(a.output/'legacy_best_attention.npz',heat=legacy,tokens=np.stack(tokens),frames=frames)
    del policy,adapter,batch,result
    gc.collect();torch.cuda.empty_cache()
    maps=[legacy,previous[0]['heat'],previous[1]['heat']]
    labels=['legacy best (old coffee)','augmented best (step 38750)','augmented step 76250']
    limits=[max(float(np.percentile(np.concatenate([m[:,c].ravel() for m in maps]),99.5)),1e-20) for c in range(3)]
    specs={'comparison_three_checkpoints':(1060,1110),'legacy_best':(1060,620)};writers={}
    for name,size in specs.items():
        w=cv2.VideoWriter(str(a.output/f'{name}.temporary.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
        if not w.isOpened():raise RuntimeError('VideoWriter failed')
        writers[name]=w
    def render(n,name):
        width,height=specs[name];canvas=np.full((height,width,3),22,dtype=np.uint8)
        caption(canvas,'COFFEE ATTENTION | LEGACY vs AUGMENTED CHECKPOINTS',20,27,.65)
        caption(canvas,f'Episode {episode} | frame {frames[n]} | {frames[n]/manifest["source_fps"]:.2f}s | subtask {subtasks[n]} | {fps:g} FPS',20,52)
        caption(canvas,'Same raw input + seed; each model uses its own state normalization.',20,77,.5)
        rows=[None,0,1,2] if name.startswith('comparison') else [None,0]
        for row,m in enumerate(rows):
            y=110+row*245
            for c,cam in enumerate(cams):
                x=20+c*345
                caption(canvas,f'{cam} | '+('RGB' if m is None else labels[m]),x,y-9,.4)
                rgb=rgbs[n,c] if m is None else (255*blend(rgbs[n,c],maps[m][n,c],'turbo',vmin=0,vmax=limits[c],alpha=.5)).astype(np.uint8)
                canvas[y:y+224,x+40:x+264]=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        caption(canvas,'Fixed shared color range per camera/time/model. Decoder rollout only; not causal attribution.',20,height-15,.46)
        return canvas
    try:
        for n in range(len(frames)):
            for name,w in writers.items():w.write(render(n,name))
            if frames[n]==1002:cv2.imwrite(str(a.output/'preview.jpg'),render(n,'comparison_three_checkpoints'))
    finally:
        for w in writers.values():w.release()
    probes={}
    for name in specs:
        tmp=a.output/f'{name}.temporary.mp4';out=a.output/f'{name}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(tmp),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(out)],text=True))['streams'][0]
        capture=cv2.VideoCapture(str(out));count=0
        while True:
            ok,_=capture.read()
            if not ok:break
            count+=1
        capture.release()
        if count!=len(frames) or int(probe['nb_frames'])!=len(frames):raise ValueError('Decode/frame count mismatch')
        probes[name]=probe;tmp.unlink()
    info=dict(legacy_checkpoint=str(a.checkpoint),reference_videos=str(a.reference),source_sample=source,
              episode=episode,frame_indices=frames,video_streams=probes,color_limits=limits,
              seed=source['seed'],normalization='Each checkpoint uses its own saved state statistics; shared raw state and images.',
              legacy_training_split='Unknown: legacy checkpoint has no episode-split manifest.',
              source_session_reference='20261015T162257+0700_797af814 identifies the old checkpoint; video uses dataset episode 1, not that recorded robot session.',
              caveat='Attention rollout is not causal attribution or a robot success metric.')
    (a.output/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print('DONE',a.output,flush=True)


if __name__=='__main__':main()
