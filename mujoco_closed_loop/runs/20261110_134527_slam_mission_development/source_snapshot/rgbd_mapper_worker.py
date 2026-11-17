"""Independent RGB-D map process; never blocks visual target/shape feedback."""
import traceback
from pathlib import Path

import numpy as np

from batched_mapper import BatchedVoxelMapper
from depth_mapping import integrate_rgbd
from estimated_self_mask import EstimatedSelfMask
from estimation import pcc_tool_transform
from robot_volume_prior import RobotVolumePrior


def run(inputs, directory, calibration):
    directory = Path(directory)
    nav_self = wrist_self = survey_self = None
    try:
        mapper = BatchedVoxelMapper(.05, 6.)
        nav_profile=calibration.get('nav_profile','front_plus_survey')
        super_visual=calibration.get('super_visual',False)
        nav_self = EstimatedSelfMask('navigation', nav_profile=nav_profile, super_visual=super_visual)
        wrist_self = nav_self if super_visual else EstimatedSelfMask('wrist_rgbd', nav_profile=nav_profile, super_visual=False)
        if 'US' in calibration:
            survey_self = EstimatedSelfMask('survey_rgbd',width=320,height=240,nav_profile=nav_profile,super_visual=super_visual)
        robot = RobotVolumePrior(profile='grasp_visible', nav_profile=nav_profile,approved_stage1=super_visual)
        positions = []
        last_save = -10.
        frames = 0
        def save(stamp):
            snapshot = mapper.snapshot(frame_id='world', source='fast_lio_registered_rgbd',
                stamp=stamp, metadata={'truth_geometry_access': False,
                    'body_pose_source': 'FAST_LIO',
                    'wrist_pose_source': 'measured_strain_PCC',
                    'self_mask': 'estimated_CAD_depth',
                    'integrated_frame_pairs': frames,
                    'navigation_camera_profile':nav_profile,
                    'underbody_survey_camera_used':survey_self is not None},
                traversed_positions=positions)
            snapshot.save(directory/'rgbd_online_map.npz')
        stamp = 0.
        while True:
            packet = inputs.get()
            if packet is None:
                break
            stamp, nav, wrist, survey, WU, bend = packet
            nav_self.camera='navigation'
            nav_mask = nav_self.mask(nav, bend)
            wrist_self.camera='wrist_rgbd'
            wrist_mask = wrist_self.mask(wrist, bend)
            wrist_camera = WU @ calibration['UB'] @ pcc_tool_transform(bend) @ calibration['TC']
            integrate_rgbd(mapper, nav, WU @ calibration['UC'], nav_mask, stride=8)
            integrate_rgbd(mapper, wrist, wrist_camera, wrist_mask, stride=8)
            if survey is not None:
                survey_mask=survey_self.mask(survey,bend)
                integrate_rgbd(mapper,survey,WU @ calibration['US'],survey_mask,stride=6)
            robot.integrate(mapper, WU, bend)
            positions.append(WU[:3, 3].copy())
            frames += 1
            if stamp-last_save >= 2.:
                save(stamp)
                last_save = stamp
        if frames:
            save(stamp)
    except Exception:
        (directory/'rgbd_mapper_error.txt').write_text(traceback.format_exc())
    finally:
        if nav_self is not None:
            nav_self.close()
        if wrist_self is not None and wrist_self is not nav_self:
            wrist_self.close()
        if survey_self is not None:
            survey_self.close()
