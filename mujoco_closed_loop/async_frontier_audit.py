"""Check online frontier selection while MuJoCo keeps advancing physics."""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from async_frontier import AsyncFrontierSelector
from planner_adapter import ROOT  # registers the V2 library source path
from uav_continuum_whole_body_v2.mapping import MapSnapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('map', type=Path)
    parser.add_argument('--state', nargs=6, type=float,
                        default=[-.6, -1.2, 1.5, 0., 0., 0.])
    parser.add_argument('--goal', nargs=3, type=float, default=[-.6, 0., 1.5])
    args = parser.parse_args()
    snapshot = MapSnapshot.load(args.map)
    model = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.001"/><worldbody><body><freejoint/><geom type="sphere" size="0.02" mass="1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    selector = AsyncFrontierSelector()
    selector.request(args.map, np.asarray(args.state), np.asarray(args.goal), snapshot.map_hash)
    started = time.monotonic()
    steps = 0
    try:
        while selector.process is not None:
            mujoco.mj_step(model, data)
            steps += 1
            result = selector.poll()
            if result is not None:
                print(json.dumps(dict(result=result, physics_steps_while_selecting=steps,
                                      simulated_time_advanced_s=data.time,
                                      wall_s=time.monotonic()-started)))
                return
            time.sleep(.001)
    finally:
        selector.close()


if __name__ == '__main__':
    main()
