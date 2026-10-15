import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pelican_vla0.5_infer'))
from session_recorder import SessionRecorder
from astribot_runtime import should_return_home, S1_HOME_POSE


class RecorderTests(unittest.TestCase):
    def test_roundtrip_and_video(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            rec=SessionRecorder(tmp,{'test':True},fps=30)
            source=np.arange(48,dtype=np.uint8).reshape(4,4,3)
            rec.event('state',position=np.zeros(16))
            rec.trace('cycle_000000',{'history_cam':[source,source], 'actions':np.ones((50,16))},{'task':'test'})
            rec.frames({'cam':np.full((32,32,3),100,dtype=np.uint8)},time.monotonic())
            rec.close('completed')
            manifest=json.loads((rec.path/'session.json').read_text())
            self.assertTrue(manifest['complete'])
            data=np.load(rec.path/'cycles/cycle_000000.npz',allow_pickle=False)
            np.testing.assert_array_equal(data['history_cam'][0],source)
            video=cv2.VideoCapture(str(rec.path/'videos/cam.mp4'))
            ok,_=video.read();video.release();self.assertTrue(ok)
            rec.close('duplicate')
            self.assertEqual(json.loads((rec.path/'session.json').read_text())['outcome'],'completed')

    def test_backpressure_nonblocking_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=SessionRecorder(tmp,{},video=False,event_queue=1)
            start=time.monotonic()
            for i in range(1000): rec.event('load',i=i)
            elapsed=time.monotonic()-start
            rec.close('completed')
            self.assertLess(elapsed,1.)
            self.assertGreater(rec.dropped['event'],0)
            self.assertFalse(json.loads((rec.path/'session.json').read_text())['complete'])

    def test_worker_error_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=SessionRecorder(tmp,{},video=False)
            rec.event('bad',unsupported=object())
            limit=time.monotonic()+5
            while rec.process.is_alive() and time.monotonic()<limit: time.sleep(.01)
            with self.assertRaises(RuntimeError): rec.check()
            rec.close('error')
            self.assertFalse(json.loads((rec.path/'session.json').read_text())['complete'])

    def test_home_only_on_graceful_execute(self):
        base=dict(execute=True,commands_started=True,outcome='completed',enabled=True,ros_ok=True)
        self.assertTrue(should_return_home(**base))
        for field,value in [('execute',False),('commands_started',False),('outcome','interrupted'),
                ('outcome','error'),('enabled',False),('ros_ok',False)]:
            self.assertFalse(should_return_home(**dict(base,**{field:value})))
        self.assertEqual(set(S1_HOME_POSE),{'torso','head','left_arm','right_arm','left_gripper','right_gripper'})

if __name__=='__main__': unittest.main()
