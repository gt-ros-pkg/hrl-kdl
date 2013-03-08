#!/usr/bin/python
import numpy as np
import roslib
roslib.load_manifest("hrl_geom")
import rospy
from hrl_geom.pose_converter import PoseConv

if __name__ == "__main__":
    rospy.init_node("test_poseconv")
    homo_mat = PoseConv.to_homo_mat([0., 1., 2.], [0., 0., np.pi/2])
    pose_msg = PoseConv.to_pose_msg(homo_mat)
    pos, quat = PoseConv.to_pos_quat(pose_msg)
    pose_stamped_msg = PoseConv.to_pose_stamped_msg("/base_link", pos, quat)
    pose_stamped_msg2 = PoseConv.to_pose_stamped_msg("/base_link", [pos, quat])
    tf_stamped = PoseConv.to_tf_stamped_msg("/new_link", pose_stamped_msg)
    p, R = PoseConv.to_pos_rot("/new_link", tf_stamped)
    for name in ["homo_mat", "pose_msg", "pos", "quat", "pose_stamped_msg", 
                 "pose_stamped_msg2", "tf_stamped", "p", "R"]:
        print "%s: \n" % name, locals()[name]
