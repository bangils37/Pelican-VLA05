"""Paired checkpoint benchmark: inference latency with/without recording."""
import argparse
import os,sys,json,time,threading
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'pelican_vla0.5_infer'),str(root/'attention_vis/src')]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True,help='Zarr replay dataset (data/state and three cameras)')
    parser.add_argument('--output',type=Path,default=Path('/tmp/pelican-benchmarks'))
    parser.add_argument('--pairs',type=int,default=20)
    parser.add_argument('--task',default='serving brewed coffee')
    args=parser.parse_args()
    if args.pairs<2: parser.error('--pairs must be >= 2')
    import numpy as np,torch,zarr,transformers
    from basevla_infer import PelicanVLA05Inference
    from infer_astribot import _load_stats
    from session_recorder import SessionRecorder, encode_history
    ckpt=args.model_path.resolve()
    ss,ac=_load_stats(ckpt/'stats.json');cams=['cam_head','cam_left_wrist','cam_right_wrist']
    mapping={c:f'image{i}' for i,c in enumerate(cams)}
    engine=PelicanVLA05Inference(model_path=ckpt,camera_map=mapping,state_stats=ss,action_stats=ac,
        action_dim=16,qwen3_vl_path=os.environ['QWEN3_VL_PATH'])
    data=zarr.open(str(args.dataset),mode='r')['data']
    history={c:np.asarray(data[c][15:31]) for c in cams}; state=np.asarray(data['state'][30]);task=args.task
    for _ in range(3):engine.infer({},state,task,image_history=history)
    rec=SessionRecorder(args.output,dict(
        args={'inference_steps':None},model_path=str(ckpt),camera_map=mapping,state_stats=ss,action_stats=ac,
        qwen3_vl_path=os.environ['QWEN3_VL_PATH'],cosmos_tokenizer_path=os.environ['COSMOS_TOKENIZER_PATH'],
        checkpoint_files={p.name:dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in ckpt.iterdir() if p.is_file()}))
    active=threading.Event();stop=threading.Event()
    def sensors():
        while not stop.wait(1/30):
            if active.is_set():
                rec.frames({c:history[c][-1] for c in cams},time.monotonic())
                for part in ['left_arm','right_arm','head','torso']:
                    rec.event('joint_state',part=part,position=state)
    thread=threading.Thread(target=sensors);thread.start()
    elapsed={False:[],True:[]};submit=[]
    try:
        for i in range(args.pairs*2):
            recording=(i%4 in (1,2))  # ABBA order limits monotonic warmup/drift bias
            if recording:active.set()
            else:active.clear()
            torch.cuda.synchronize();start=time.perf_counter()
            if recording:
                cpu=torch.random.get_rng_state().numpy().copy()
                gpu=torch.cuda.get_rng_state(engine.device).cpu().numpy().copy()
            actions=engine.infer({},state,task,image_history=history)
            assert actions.shape==(50,16) and np.isfinite(actions).all()
            if recording:
                t=time.perf_counter()
                rec.trace(f'cycle_{i:06d}',dict(state=state,states=np.stack([state]*16),actions=actions,
                    rng_cpu=cpu,rng_cuda=gpu,**encode_history(history,engine.image_delta_indices)),
                    dict(cycle=i,task=task))
                submit.append(time.perf_counter()-t)
            torch.cuda.synchronize();elapsed[recording].append(time.perf_counter()-start)
    finally:
        stop.set();thread.join();rec.close('completed')
    result=dict(session=str(rec.path),baseline_ms=[x*1000 for x in elapsed[False]],
        recording_ms=[x*1000 for x in elapsed[True]],submit_ms=[x*1000 for x in submit],
        baseline_median_ms=float(np.median(elapsed[False])*1000),recording_median_ms=float(np.median(elapsed[True])*1000))
    result['percentiles_ms']={str(k):dict(zip(['p50','p95','p99'],np.percentile(v,[50,95,99])*1000)) for k,v in elapsed.items()}
    result['median_overhead_percent']=100*(np.median(elapsed[True])/np.median(elapsed[False])-1)
    result['complete']=rec.metadata['complete']
    result['note']='Measured on this workload; not a guarantee of zero overhead or worst-case latency.'
    (rec.path/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if not rec.metadata['complete']: raise RuntimeError('Benchmark lost records; see session.json')
if __name__=='__main__':main()
