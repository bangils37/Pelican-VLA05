"""Standalone HOME/READY motion; no checkpoint or camera required."""
import argparse
import threading
import time
from pathlib import Path
import yaml
from astribot_runtime import S1_HOME_POSE, S1_VMO_READY_POSE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose',choices=['home','ready'],default='home')
    p.add_argument('--ros-config',default=str(Path(__file__).parent/'configs/astribot_s1_ros2.yaml'))
    p.add_argument('--duration',type=float,default=4.)
    p.add_argument('--rate',type=float,default=100.)
    p.add_argument('--max-sensor-age',type=float,default=1.)
    p.add_argument('--timeout',type=float,default=30.)
    p.add_argument('--execute',action='store_true',help='Publish motion; default is diagnostics only')
    p.add_argument('--yes',action='store_true')
    a=p.parse_args()
    if min(a.duration,a.rate,a.max_sensor_age,a.timeout)<=0:
        p.error('durations, rate and freshness must be positive')
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from astribot_ros2 import AstribotS1Node
    cfg=yaml.safe_load(Path(a.ros_config).read_text())
    target=S1_HOME_POSE if a.pose=='home' else S1_VMO_READY_POSE
    cfg['required_state_parts']=list(target)
    rclpy.init()
    node=AstribotS1Node(cfg,{},1,30.)
    executor=MultiThreadedExecutor(num_threads=2); executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True); thread.start()
    started=False
    try:
        deadline=time.monotonic()+a.timeout
        while not node.readiness()[0]:
            if time.monotonic()>deadline: raise TimeoutError(node.readiness()[1])
            time.sleep(.05)
        _,_,_,parts=node.trace_snapshot()
        for key,value in target.items():
            print(f'{key}: {parts[key].tolist()} -> {value.tolist()}')
        if not a.execute:
            print('DRY-RUN: no commands published'); return
        if not a.yes and input(f"Enter 's' to move to {a.pose.upper()}: ").strip().lower()!='s':
            return
        started=True
        node.move_to_pose(target,a.duration,a.rate,a.max_sensor_age)
        print(f'{a.pose.upper()} target published successfully')
    finally:
        try:
            if started: node.publish_hold()
        finally:
            executor.shutdown(timeout_sec=2.); node.destroy_node()
            if rclpy.ok(): rclpy.shutdown()
            thread.join(2.)


if __name__=='__main__': main()
