"""Bounded, non-blocking producers; disk/video work in a spawned CPU process.

Overload drops records instead of stalling control. Counters and worker errors
are explicit: a session with drops/errors is not an exact replay dataset.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, default=json_default, indent=2))
    temporary.replace(path)


def _writer(directory, queues, stop, ready, status, fps, video):
    import cv2
    cv2.setNumThreads(1)
    os.nice(10)
    root = Path(directory)
    videos = {}
    counts = Counter()
    error = None
    events = None
    console = None
    try:
        (root/'cycles').mkdir()
        (root/'videos').mkdir()
        events = (root/'events.jsonl').open('a', buffering=1)
        console = (root/'console.log').open('a', buffering=1)
        ready.set()
        done = set()
        while len(done) < len(queues):
            worked = False
            for kind, q in queues.items():
                if kind in done:
                    continue
                try:
                    item = q.get(timeout=.01)
                except queue.Empty:
                    continue
                if item is None:
                    done.add(kind)
                    continue
                worked = True
                if kind == 'trace':
                    name, arrays, meta = item
                    path = root/'cycles'/name
                    with path.with_suffix('.tmp').open('wb') as f:
                        np.savez(f, **arrays)  # lossless, no CPU-heavy compression
                    path.with_suffix('.tmp').replace(path.with_suffix('.npz'))
                    _write_json(path.with_suffix('.json'),meta)
                elif kind == 'video':
                    images, stamp, phase = item
                    if video:
                        for cam, rgb in images.items():
                            if cam not in videos:
                                h,w = rgb.shape[:2]
                                writer = cv2.VideoWriter(str(root/'videos'/f'{cam}.mp4'),
                                    cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                                if not writer.isOpened():
                                    raise RuntimeError(f'Cannot open video writer for {cam}')
                                videos[cam] = writer
                            videos[cam].write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    events.write(json.dumps(dict(kind='video_frame', monotonic=stamp,
                        phase=phase,index=counts['video']),default=json_default)+'\n')
                else:
                    if item.get('kind') == 'console':
                        console.write(item['text']+'\n')
                    events.write(json.dumps(item,default=json_default)+'\n')
                counts[kind] += 1
            if not worked:
                time.sleep(.001)
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        status.put(error)
        ready.set()
    finally:
        for writer in videos.values():
            writer.release()
        if events:
            events.close()
        if console:
            console.close()
        _write_json(root/'writer.json', dict(written=dict(counts),error=error,closed=True))


class SessionRecorder:
    def __init__(self, directory, metadata, *, fps=30., video=True, trace_queue=2,
                 event_queue=4096, video_queue=8):
        if fps <= 0 or min(trace_queue,event_queue,video_queue) <= 0:
            raise ValueError('Recording FPS and queue capacities must be positive')
        self.path = Path(directory).expanduser()/(
            datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')+'_'+uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True,exist_ok=False)
        self.metadata = dict(schema_version=1,started_at=datetime.now().astimezone().isoformat(),
            started_monotonic=time.monotonic(), **metadata)
        _write_json(self.path/'session.json',self.metadata)
        ctx = mp.get_context('spawn')
        self.queues = {k:ctx.Queue(n) for k,n in
            [('trace',trace_queue),('event',event_queue),('video',video_queue)]}
        self.status = ctx.Queue()
        self.ready = ctx.Event()
        self.stop = ctx.Event()
        self.dropped = Counter()
        self.accepted = Counter()
        self._lock = threading.Lock()
        self._reported_drops = 0
        self.closed = False
        self.phase = 'STARTUP'
        self._next_video = 0.
        self.fps = fps
        self.video_enabled = video
        self.process = ctx.Process(target=_writer,args=(str(self.path),self.queues,
            self.stop,self.ready,self.status,fps,video),daemon=True)
        self.process.start()
        if not self.ready.wait(15):
            self.process.terminate(); self.process.join(2)
            raise RuntimeError('Recorder startup timed out')
        self.check()

    def check(self):
        total = sum(self.dropped.values())
        if total != self._reported_drops:
            self._reported_drops = total
            print(f"WARNING: recorder overload, dropped records: {dict(self.dropped)}",file=sys.stderr)
        try:
            error = self.status.get_nowait()
        except queue.Empty:
            if not self.process.is_alive() and not self.closed:
                raise RuntimeError('Recorder worker exited unexpectedly')
        else:
            raise RuntimeError(f'Recorder failed: {error}')

    def submit(self,kind,payload):
        # Never encode, compress, stack images or wait for disk here.
        with self._lock:
            if self.closed:
                return False
            try:
                self.queues[kind].put_nowait(payload)
                self.accepted[kind] += 1
                return True
            except queue.Full:
                self.dropped[kind] += 1
                return False

    def event(self,kind,**fields):
        return self.submit('event',dict(kind=kind,monotonic=time.monotonic(),phase=self.phase,**fields))

    def frames(self,images,stamp):
        if not self.video_enabled:
            return
        with self._lock:
            if stamp < self._next_video:
                return
            self._next_video = stamp+1/self.fps
        self.submit('video',(images,stamp,self.phase))

    def trace(self,name,arrays,metadata):
        return self.submit('trace',(name,arrays,metadata))

    def close(self,outcome,timeout=20.):
        with self._lock:
            if self.closed:
                return
            self.closed = True
        deadline = time.monotonic()+timeout
        # Sentinels drain accepted records. Blocking is allowed only after hold.
        for q in self.queues.values():
            try:
                q.put(None,timeout=max(.01,deadline-time.monotonic()))
            except queue.Full:
                break
        self.process.join(max(0.,deadline-time.monotonic()))
        timed_out = self.process.is_alive()
        if timed_out:
            self.process.terminate(); self.process.join(2)
        for q in self.queues.values():
            q.cancel_join_thread(); q.close()
        writer_file = self.path/'writer.json'
        worker = json.loads(writer_file.read_text()) if writer_file.exists() else {}
        self.metadata.update(outcome=outcome,accepted=dict(self.accepted),dropped=dict(self.dropped),
            writer=worker,flush_timed_out=timed_out,complete=(not timed_out and
                not worker.get('error') and bool(worker) and not any(self.dropped.values())))
        _write_json(self.path/'session.json',self.metadata)
        print(f'Session saved: {self.path} complete={self.metadata["complete"]} drops={dict(self.dropped)}')


class ConsoleCapture:
    """Keep terminal behavior; mirror text into the asynchronous event log."""
    def __init__(self,stream,recorder):
        self.stream,self.recorder=stream,recorder
    def write(self,text):
        result=self.stream.write(text)
        if text.strip(): self.recorder.event('console',text=text)
        return result
    def flush(self): self.stream.flush()
    def __getattr__(self,key): return getattr(self.stream,key)
