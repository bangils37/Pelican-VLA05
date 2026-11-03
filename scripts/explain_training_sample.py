#!/usr/bin/env python3
"""Compare two coffee checkpoints on one verified training sample, using decoder rollout."""
import argparse
import base64
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'pelican_vla0.5_infer/src'), str(ROOT/'attention_vis/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('COSMOS_TOKENIZER_PATH', str(ROOT/'pretrained_model/cosmos_tokenizer'))
os.environ.setdefault('ATTNVIS_PELICANVLA_SRC', str(ROOT/'pelican_vla0.5_infer'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from training.dataset import LeRobotPelicanDataset
from lerobot.policies.pelicanvla05 import PelicanVLA05Policy
from attnvis.adapters.pelicanvla import PelicanVLAAdapter
from attnvis.core.types import Frame
from attnvis.figlib import blend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='/home/aitt/data/serving_brewed_coffee_clean_374_episodes')
    p.add_argument('--checkpoints', nargs=2, type=Path, default=[ROOT/'checkpoints/coffee_aug_20261020_2gpu/best_model',ROOT/'checkpoints/coffee_aug_20261020_2gpu/checkpoint-076250'])
    p.add_argument('--output', type=Path, default=ROOT/'explanations/coffee_aug_20261022_training_sample')
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(2)
    torch.cuda.set_device(torch.device(a.device))
    configs=[json.loads((ck/'instruction_augmentation.json').read_text()) for ck in a.checkpoints]
    ds=LeRobotPelicanDataset(a.dataset, split='train', seed=configs[0]['seed'], val_ratio=configs[0]['val_ratio'],
                            use_subtasks_prob=0, instruction_augmentation=False)
    if ds.active_episodes_set != set(configs[0]['active_episode_ids']): raise ValueError('Training split mismatch')
    # Choose an interior point in the pouring stage in the first eligible train episode.
    selected=None
    for episode in sorted(ds.active_episodes_set):
        eligible=[]
        for i,(row,ep,frame,length,_) in enumerate(ds.samples):
            if ep!=episode or frame<15 or frame+max(49,15)>=length: continue
            if np.all(ds.subtask_indices[row-15:row+50]==2):eligible.append(i)
        if eligible:
            selected=eligible[len(eligible)//2];break
    if selected is None:raise RuntimeError('No valid interior pouring sample')
    row,episode,frame_idx,ep_len,_=ds.samples[selected]
    for config in configs:
        if episode not in config['active_episode_ids']:raise ValueError('Sample is not in both training splits')
    sample=ds[selected]
    a.output.mkdir(parents=True)
    cams=['cam_head','cam_left_wrist','cam_right_wrist']
    rgb={cam:np.rint(sample[f'observation.images.image{i}'][1].permute(1,2,0).numpy()*255).astype(np.uint8) for i,cam in enumerate(cams)}
    for i,cam in enumerate(cams):
        vr,start=ds._get_video_reader(cam,ds.episodes_by_id[episode])
        if vr is None or not 0<=start+frame_idx-15<start+frame_idx+15<len(vr):raise RuntimeError('Missing/out-of-range camera frames')
        if not np.any(rgb[cam]):raise RuntimeError('Empty camera frame')
        Image.fromarray(rgb[cam]).save(a.output/f'input_{cam}.png')
    raw=Frame(rgb=rgb,state=ds.raw_states[row],instruction=sample['task'],cams=cams)
    gt=sample['action'].numpy()[:,:16]
    state=sample['observation.state'].numpy()
    batch={k:v.unsqueeze(0).to(a.device) for k,v in sample.items() if isinstance(v,torch.Tensor) and k!='action'}
    fingerprint=hashlib.sha256()
    for k,v in sorted(batch.items()):
        fingerprint.update(k.encode());fingerprint.update(v.cpu().contiguous().numpy().tobytes())
    metadata=dict(dataset=str(Path(a.dataset).resolve()),split='train',dataset_index=selected,global_row=row,
                  episode=episode,frame=frame_idx,episode_length=ep_len,subtask_index=2,instruction=sample['task'],
                  temporal_offsets=[-15,0,15],chunk_size=50,seed=a.seed,action_step=0,
                  preprocessing='Exact training Dataset tensors, 224x224 direct resize; no repeated-frame or letterbox substitution',
                  batch_sha256=fingerprint.hexdigest(),checkpoint_paths=[str(c.resolve()) for c in a.checkpoints])
    (a.output/'sample.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savez_compressed(a.output/'sample.npz',ground_truth_normalized=gt,state_normalized=state,
                        **{cam:sample[f'observation.images.image{i}'].numpy() for i,cam in enumerate(cams)})
    ds._video_readers.clear()
    print('Selected sample:',metadata,flush=True)
    results=[]
    for ck in a.checkpoints:
        label=ck.name; out=a.output/label;out.mkdir()
        stats=json.loads((ck/'stats.json').read_text())
        for key,mean,std in [('state',ds.state_mean,ds.state_std),('action',ds.action_mean,ds.action_std)]:
            np.testing.assert_allclose(stats[key]['mean'],mean,rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(stats[key]['std'],std,rtol=1e-6,atol=1e-6)
        policy=PelicanVLA05Policy.from_pretrained(str(ck),strict=True).to(a.device).eval()
        for parameter in policy.parameters():parameter.requires_grad_(False)
        adapter=PelicanVLAAdapter(ckpt=str(ck))
        adapter.policy=policy;adapter.config=policy.config;adapter._device=a.device
        adapter._image_resolution=tuple(policy.config.image_resolution)
        adapter._merge=policy.model.vlm.config.vision_config.spatial_merge_size
        adapter._img_token_id=policy.model.vlm.config.image_token_id
        with torch.random.fork_rng(devices=[torch.device(a.device).index]):
            torch.manual_seed(a.seed);torch.cuda.manual_seed(a.seed)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                predicted,_=policy.predict_action_chunk(batch,decode_image=False)
        prediction=predicted[0].float().cpu().numpy()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            explanation=adapter.explain(raw,method='rollout',action_step=0,seed=a.seed,batch_override=batch)
        heat=explanation.heat
        for cam in cams:
            if not np.isfinite(heat[cam]).all() or heat[cam].max()<=0:raise RuntimeError('Invalid heatmap')
            np.savez_compressed(out/f'{cam}.npz',heat=heat[cam],rgb=rgb[cam],tokens=explanation.raw['token_maps'][cam])
        if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite actions')
        np.savez_compressed(out/'rollout.npz',action_to_image=explanation.raw['action_to_image'],
                           image_columns=explanation.raw['image_columns'],grid_thw=explanation.raw['grid_thw'])
        mean,std=ds.action_mean,ds.action_std
        physical=prediction*std+mean;gt_physical=gt*std+mean
        np.savez_compressed(out/'actions.npz',predicted_normalized=prediction,ground_truth_normalized=gt,
                            predicted=physical,ground_truth=gt_physical)
        metrics=dict(normalized_action_mse=float(np.mean((prediction-gt)**2)),
                     normalized_action_mae=float(np.mean(abs(prediction-gt))),
                     normalized_mse_per_joint=np.mean((prediction-gt)**2,axis=0).tolist())
        explanation.raw['metadata'].update(temporal_input='dataset_training_t_minus15_t_t_plus15',
            preprocessing=metadata['preprocessing'],sample=metadata,metrics=metrics,
            caveat='Decoder image-token attention at action step 0, last denoising call; not causal attribution or a full temporal/vision-encoder explanation.')
        (out/'metadata.json').write_text(json.dumps(explanation.raw['metadata'],indent=2)+'\n')
        results.append(dict(label=label,heat=heat,physical=physical,metrics=metrics))
        print(label,metrics,flush=True)
        del predicted,policy,adapter,explanation
        gc.collect();torch.cuda.empty_cache()
    # Shared per-camera color limits across checkpoints; no independently rescaled overlays.
    fig,axes=plt.subplots(3,3,figsize=(12,12))
    for j,cam in enumerate(cams):
        vmax=max(float(np.percentile(np.concatenate([r['heat'][cam].ravel() for r in results]),99.5)),1e-20)
        axes[0,j].imshow(rgb[cam]);axes[0,j].set_title(cam+' | input')
        for i,r in enumerate(results):
            overlay=(255*blend(rgb[cam],r['heat'][cam],'turbo',vmin=0,vmax=vmax,alpha=.5)).astype(np.uint8)
            Image.fromarray(overlay).save(a.output/r['label']/f'{cam}.png')
            axes[i+1,j].imshow(overlay);axes[i+1,j].set_title(r['label']+' | rollout')
        for i in range(3):axes[i,j].axis('off')
    fig.suptitle(f'Train episode {episode}, frame {frame_idx} | {sample["task"]}\nSame sample / seed; shared color scale within each camera')
    fig.tight_layout();fig.savefig(a.output/'attention_comparison.png',dpi=150);plt.close(fig)
    names=ds.info['features']['action']['names']
    if len(names)==1 and isinstance(names[0],list):names=names[0]
    fig,axes=plt.subplots(4,4,figsize=(16,12));time=np.arange(50)/ds.fps
    for d,ax in enumerate(axes.flat):
        ax.plot(time,gt_physical[:,d],color='black',label='Ground truth')
        for r in results:ax.plot(time,r['physical'][:,d],label=r['label'],alpha=.85)
        ax.set_title(names[d]);ax.set_xlabel('seconds');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('50-step open-loop action prediction | original dataset units (joint and gripper units differ)')
    fig.tight_layout();fig.savefig(a.output/'action_comparison.png',dpi=150);plt.close(fig)
    report={'sample':metadata,'metrics':{r['label']:r['metrics'] for r in results},
            'limitations':['One training sample; not validation or rollout success evaluation.','Attention rollout is not causal attribution.','Qwen image tokens here encode the current frame; t+15 is a training target, not a claim of future visual attribution.']}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Coffee checkpoint comparison\n',f'Train episode **{episode}**, frame **{frame_idx}**, subtask **2**. Instruction: `{sample["task"]}`.\n',f'Same input tensors, seed {a.seed}, rollout action step 0.\n','\n'.join('- '+s for s in report['limitations'])]}]
    for filename in ('attention_comparison.png','action_comparison.png'):
        cells.append({'cell_type':'code','metadata':{},'execution_count':len(cells),'source':[f'from IPython.display import Image, display\ndisplay(Image(filename="{filename}"))'],
                      'outputs':[{'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode((a.output/filename).read_bytes()).decode(),'text/plain':[filename]}}]})
    cells.append({'cell_type':'markdown','metadata':{},'source':['## Metrics (normalized action space)\n','\n'.join(f'- {r["label"]}: MSE={r["metrics"]["normalized_action_mse"]:.6f}, MAE={r["metrics"]["normalized_action_mae"]:.6f}' for r in results)]})
    (a.output/'comparison.ipynb').write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':4}))
    print('DONE',a.output,flush=True)


if __name__=='__main__':main()
