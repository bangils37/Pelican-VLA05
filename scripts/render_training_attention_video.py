#!/usr/bin/env python3
"""Render synchronized training-episode attention for two checkpoints (no robot execution)."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

# Reuse the dataset explanation environment and model imports.
from explain_training_sample import ROOT, np, torch, LeRobotPelicanDataset, PelicanVLA05Policy, PelicanVLAAdapter, Frame, blend
import cv2


def caption(canvas, text, x, y, scale=.55):
    cv2.putText(canvas,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(230,230,230),1,cv2.LINE_AA)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-report',type=Path,default=ROOT/'explanations/coffee_aug_20261022_training_sample/sample.json')
    p.add_argument('--output',type=Path,default=ROOT/'explanations/coffee_aug_20261022_training_sample/videos')
    p.add_argument('--stride',type=int,default=6)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    if a.stride<1:raise ValueError('stride must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    source=json.loads(a.sample_report.read_text());cks=[Path(s) for s in source['checkpoint_paths']]
    config=json.loads((cks[0]/'instruction_augmentation.json').read_text())
    torch.set_num_threads(2);torch.cuda.set_device(torch.device(a.device))
    ds=LeRobotPelicanDataset(source['dataset'],split='train',seed=config['seed'],val_ratio=config['val_ratio'],use_subtasks_prob=0)
    episode=source['episode']
    for ck in cks:
        if episode not in json.loads((ck/'instruction_augmentation.json').read_text())['active_episode_ids']:
            raise ValueError('Episode not in checkpoint training split')
        stats=json.loads((ck/'stats.json').read_text())
        for key,mean,std in [('state',ds.state_mean,ds.state_std),('action',ds.action_mean,ds.action_std)]:
            np.testing.assert_allclose(stats[key]['mean'],mean,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(stats[key]['std'],std,rtol=1e-6,atol=1e-6)
    selected=[i for i,(_,ep,f,_,_) in enumerate(ds.samples) if ep==episode and f%a.stride==0]
    frames=[ds.samples[i][2] for i in selected];fps=ds.fps/a.stride
    if not selected:raise ValueError('No frames selected')
    cams=['cam_head','cam_left_wrist','cam_right_wrist']
    for cam in cams:
        vr,start=ds._get_video_reader(cam,ds.episodes_by_id[episode])
        if vr is None or start+ds.episodes_by_id[episode]['length']>len(vr):raise ValueError(f'Invalid video: {cam}')
    a.output.mkdir(parents=True)
    rgbs=[];subtasks=[];all_heats=[];metadata=[]
    for ck_index,ck in enumerate(cks):
        policy=PelicanVLA05Policy.from_pretrained(str(ck),strict=True).to(a.device).eval()
        for parameter in policy.parameters():parameter.requires_grad_(False)
        adapter=PelicanVLAAdapter(ckpt=str(ck));adapter.policy=policy;adapter.config=policy.config
        adapter._device=a.device;adapter._image_resolution=tuple(policy.config.image_resolution)
        adapter._merge=policy.model.vlm.config.vision_config.spatial_merge_size
        adapter._img_token_id=policy.model.vlm.config.image_token_id
        heats=[];tokens=[];started=time.monotonic()
        for n,i in enumerate(selected):
            row,_,frame,_,_=ds.samples[i];sample=ds[i]
            rgb=np.stack([np.rint(sample[f'observation.images.image{k}'][1].permute(1,2,0).numpy()*255).astype(np.uint8) for k in range(3)])
            if ck_index==0:
                rgbs.append(rgb);subtasks.append(int(ds.subtask_indices[row]))
            elif not np.array_equal(rgb,rgbs[n]):raise RuntimeError('Checkpoint inputs differ')
            obs=Frame(rgb=dict(zip(cams,rgb)),state=ds.raw_states[row],instruction=sample['task'],cams=cams)
            batch={key:value.unsqueeze(0).to(a.device) for key,value in sample.items() if isinstance(value,torch.Tensor) and key!='action'}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=adapter.explain(obs,method='rollout',action_step=0,seed=source['seed'],batch_override=batch)
            heat=np.stack([result.heat[cam] for cam in cams])
            if not np.isfinite(heat).all() or heat.max()<=0:raise RuntimeError('Invalid heatmaps')
            heats.append(heat);tokens.append(np.stack([result.raw['token_maps'][cam] for cam in cams]))
            if n==0:
                meta=dict(result.raw['metadata']);meta['temporal_input']='actual training frames at [-15, 0, 15], clamped at episode ends'
                metadata.append(meta)
            if (n+1)%20==0 or n+1==len(selected):
                print(f'{ck.name}: {n+1}/{len(selected)} frames, {time.monotonic()-started:.1f}s',flush=True)
        all_heats.append(np.stack(heats))
        np.savez_compressed(a.output/f'{ck.name}_attention.npz',heat=all_heats[-1],tokens=np.stack(tokens),frames=frames)
        del policy,adapter,result,batch
        gc.collect();torch.cuda.empty_cache()
    # A single color range per camera across time AND both checkpoints.
    limits=[max(float(np.percentile(np.concatenate([h[:,c].ravel() for h in all_heats]),99.5)),1e-20) for c in range(3)]
    np.savez_compressed(a.output/'inputs.npz',rgb=np.stack(rgbs),frames=frames,subtasks=subtasks)
    outputs={'comparison':(1060,860),'best_model':(1060,610),'checkpoint-076250':(1060,610)}
    writers={}
    for name,(w,h) in outputs.items():
        writer=cv2.VideoWriter(str(a.output/f'{name}.temporary.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened():raise RuntimeError('VideoWriter failed')
        writers[name]=writer
    def render(n,which):
        w,h=outputs[which];canvas=np.full((h,w,3),22,dtype=np.uint8)
        caption(canvas,'PELICAN-VLA | TRAIN EPISODE ATTENTION ROLLOUT',20,27,.7)
        caption(canvas,f'Episode {episode} | frame {frames[n]} | t={frames[n]/ds.fps:.2f}s | subtask {subtasks[n]} | {fps:g} FPS',20,52)
        caption(canvas,'Instruction: '+source['instruction'],20,76)
        models=[0,1] if which=='comparison' else [0 if which=='best_model' else 1]
        rows=[None]+models
        for row,model in enumerate(rows):
            y=110+row*245
            for c,cam in enumerate(cams):
                x=20+c*345
                label='RGB' if model is None else cks[model].name
                caption(canvas,f'{cam} | {label}',x,y-9,.47)
                rgb=rgbs[n][c] if model is None else (255*blend(rgbs[n][c],all_heats[model][n,c],'turbo',vmin=0,vmax=limits[c],alpha=.5)).astype(np.uint8)
                panel=cv2.cvtColor(cv2.resize(rgb,(224,224)),cv2.COLOR_RGB2BGR)
                canvas[y:y+224,x+40:x+264]=panel
        caption(canvas,'Same seed; fixed color range per camera across time/models. Action step 0, final denoising call.',20,h-17,.46)
        return canvas
    try:
        for n in range(len(frames)):
            for name,writer in writers.items():writer.write(render(n,name))
            if n==len(frames)//2:cv2.imwrite(str(a.output/'preview.jpg'),render(n,'comparison'))
    finally:
        for writer in writers.values():writer.release()
    probes={}
    for name in outputs:
        temporary=a.output/f'{name}.temporary.mp4';final=a.output/f'{name}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(temporary),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate,duration','-of','json',str(final)],text=True))['streams'][0]
        if int(probe['nb_frames'])!=len(frames):raise RuntimeError('Encoded frame count mismatch')
        capture=cv2.VideoCapture(str(final));decoded=0
        while True:
            ok,_=capture.read()
            if not ok:break
            decoded+=1
        capture.release()
        if decoded!=len(frames):raise RuntimeError('Video decode validation failed')
        probes[name]=probe;temporary.unlink()
    report=dict(source_sample=source,episode=episode,frame_indices=frames,source_fps=ds.fps,video_fps=fps,stride=a.stride,
                color_limits=limits,color_scale='Shared per camera across all frames and both checkpoints; p99.5',
                methods=metadata,video_streams=probes,
                caveats=['Attention computed on sampled frames, not interpolated.', 'Offline decoder rollout, not causal attribution or robot rollout success.', 'Dataset episode boundaries use training loader clamping/padding.'])
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('DONE:',a.output,flush=True)


if __name__=='__main__':main()
