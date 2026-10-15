"""Exercise production motion methods with transport replaced, without ROS/robot."""
import ast
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace, MethodType
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pelican_vla0.5_infer'))
from astribot_runtime import PART_DOF, PART_KEYS, S1_HOME_POSE, pack_vector


def method(name):
    tree=ast.parse((Path(__file__).resolve().parents[1]/'pelican_vla0.5_infer/astribot_ros2.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name)
    code=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    scope=dict(np=np,time=time,PART_DOF=PART_DOF,PART_KEYS=PART_KEYS,pack_vector=pack_vector,
        rclpy=SimpleNamespace(ok=lambda:True),Mapping=dict,Sequence=list)
    exec(compile(code,'astribot_ros2.py','exec'),scope)
    return scope[name]


class MotionTests(unittest.TestCase):
    def node(self):
        node=SimpleNamespace(_lock=threading.RLock(),_parts={k:v.copy() for k,v in S1_HOME_POSE.items()},
            _part_times={k:time.monotonic() for k in S1_HOME_POSE},_target='old',_last_commands={})
        node.sent=[]
        def publish(k,v):
            node.sent.append((k,np.asarray(v).copy())); node._last_commands[k]=np.asarray(v).copy()
        node._publish_part=publish
        node.set_action_target=lambda target:setattr(node,'_target',target)
        node.publish_hold=MethodType(method('publish_hold'),node)
        node.move_to_pose=MethodType(method('move_to_pose'),node)
        return node

    def test_home_publishes_all_parts_and_hold_disables_policy(self):
        n=self.node(); n.move_to_pose(S1_HOME_POSE,.001,100.,1.)
        self.assertEqual({k for k,v in n.sent},set(S1_HOME_POSE))
        n.publish_hold()
        self.assertIsNone(n._target)
        self.assertEqual(len(n.sent),12)

    def test_stale_head_prevents_motion(self):
        n=self.node();n._part_times['head']=0.
        with self.assertRaisesRegex(RuntimeError,'stale'):n.move_to_pose(S1_HOME_POSE,.001,100.,1.)
        self.assertEqual(n.sent,[])
        self.assertIsNone(n._target)

    def test_joint_trace_keeps_all_source_measurements(self):
        n=self.node(); records=[]
        n.recorder=SimpleNamespace(event=lambda kind,**value:records.append((kind,value)))
        callback=MethodType(method('_on_joint'),n)
        msg=SimpleNamespace(position=[.1,.2],velocity=[.3,.4],acceleration=[.5,.6],
            torque=[.7,.8],mode=1,name=['h0','h1'],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=12,nanosec=34)))
        callback(msg,'head')
        self.assertEqual(records[0][1]['source']['velocity'],[.3,.4])
        self.assertEqual(records[0][1]['source']['torque'],[.7,.8])
        self.assertEqual(records[0][1]['source_stamp'],{'sec':12,'nanosec':34})

    def test_invalid_pose_rejected(self):
        n=self.node();pose=dict(S1_HOME_POSE,head=np.array([np.nan,0.]))
        with self.assertRaises(ValueError):n.move_to_pose(pose,1.,100.,1.)
        self.assertEqual(n.sent,[])

if __name__=='__main__':unittest.main()
