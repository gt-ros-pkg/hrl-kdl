#! /usr/bin/python

import numpy as np

import roslib
roslib.load_manifest('pykdl_utils')

import rospy
from sensor_msgs.msg import JointState

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

##
# Kinematics class which subscribes to the /joint_states topic, recording the current
# joint states for the kinematic chain designated.
class JointStateKDLKin(KDLKinematics):
    ##
    # Constructor
    # @param urdf URDF object of robot.
    # @param base_link Name of the root link of the kinematic chain.
    # @param end_link Name of the end link of the kinematic chain.
    # @param kdl_tree Optional KDL.Tree object to use. If None, one will be generated
    #                          from the URDF.
    # @param timeout Time in seconds to wait for the /joint_states topic.
    def __init__(self, urdf, base_link, end_link, kdl_tree=None, timeout=1.):
        super(JointStateKDLKin, self).__init__(urdf, base_link, end_link, kdl_tree)
        self._joint_angles = None
        self._joint_velocities = None
        self._joint_efforts = None
        self._joint_state_inds = None

        rospy.Subscriber('/joint_states', JointState, self._joint_state_cb)

        if timeout > 0:
            self.wait_for_joint_angles(timeout)

    ##
    # Joint angles listener callback
    def _joint_state_cb(self, msg):
        if self._joint_state_inds is None:
            joint_names_list = self.get_joint_names()
            self._joint_state_inds = [msg.name.index(joint_name) for 
                                     joint_name in joint_names_list]
        self._joint_angles = [msg.position[i] for i in self._joint_state_inds]
        self._joint_velocities = [msg.velocity[i] for i in self._joint_state_inds]
        self._joint_efforts = [msg.effort[i] for i in self._joint_state_inds]

    ##
    # Wait until we have found the current joint angles.
    # @param timeout Time at which we break if we haven't recieved the angles.
    def wait_for_joint_angles(self, timeout=1.):
        start_time = rospy.get_time()
        r = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.get_time() - start_time < timeout:
            if self._joint_angles is not None:
                return True
            r.sleep()
        if not rospy.is_shutdown():
            rospy.logwarn("[pr2_kin] Cannot read joint angles, timing out.")
        return False

    ##
    # Returns the current joint angle positions
    # @param wrapped If False returns the raw encoded positions, if True returns
    #                the angles with the forearm and wrist roll in the range -pi to pi
    def get_joint_angles(self, wrapped=False):
        if self._joint_angles is None:
            rospy.logwarn("[pr2_kin] Joint angles haven't been filled yet.")
            return None
        if wrapped:
            return self.wrap_angles(self._joint_angles)
        else:
            return np.array(self._joint_angles)

    ##
    # Returns the current joint velocities
    def get_joint_velocities(self):
        if self._joint_velocities is None:
            rospy.logwarn("[pr2_kin] Joint velocities haven't been filled yet.")
            return None
        return np.array(self._joint_velocities)

    ##
    # Returns the current joint efforts
    def get_joint_efforts(self):
        if self._joint_efforts is None:
            rospy.logwarn("[pr2_kin] Joint efforts haven't been filled yet.")
            return None
        return np.array(self._joint_efforts)

    ##
    # Perform forward kinematics on the current joint angles.
    # @param q List of joint angles for the full kinematic chain. 
    #          If None, the current joint angles are used.
    # @param end_link Name of the link the pose should be obtained for.
    # @param base_link Name of the root link frame the end_link should be found in.
    # @return 4x4 numpy.mat homogeneous transformation or None if the joint angles are not filled.
    def forward(self, q=None, end_link=None, base_link=None):
        if q is None:
            q = self.get_joint_angles()
            if q is None:
                return None
        return super(JointStateKDLKin, self).forward(q, end_link, base_link)

    ##
    # Returns the Jacobian matrix at the end_link from the current joint angles.
    # @param q List of joint angles. If None, the current joint angles are used.
    # @return 6xN np.mat Jacobian or None if the joint angles are not filled.
    def jacobian(self, q=None):
        if q is None:
            q = self.get_joint_angles()
            if q is None:
                return None
        return super(JointStateKDLKin, self).jacobian(q)

    ##
    # Returns joint angles for continuous joints to a range [0, 2*PI)
    # @param q List of joint angles.
    # @return np.array of wrapped joint angles.
    def wrap_angles(self, q):
        contins = self.joint_types == 'continuous'
        return np.where(contins, np.mod(q, 2 * np.pi), q)

    ##
    # Returns the current effective end effector force.
    def end_effector_force(self):
        J = self.jacobian()
        tau = self.get_joint_efforts()
        if J is None or tau is None:
            return None
        f, _, _, _ = np.linalg.lstsq(J.T, tau)
        return f

def main():
    rospy.init_node("jointspace_kdl_kin")
    import sys
    def usage():
        print("Tests for kdl_parser:\n")
        print("kdl_parser <urdf file>")
        print("\tLoad the URDF from file.")
        print("kdl_parser")
        print("\tLoad the URDF from the parameter server.")
        sys.exit(1)

    if len(sys.argv) > 2:
        usage()
    if len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
        usage()
    if (len(sys.argv) == 1):
        robot = URDF.load_from_parameter_server(verbose=False)
    else:
        robot = URDF.load_xml_file(sys.argv[1], verbose=False)

    if True:
        import random
        base_link = robot.get_root()
        end_link = robot.links.keys()[random.randint(0, len(robot.links)-1)]
        print "Root link: %s; Random end link: %s" % (base_link, end_link)
        js_kin = JointStateKDLKin(robot, base_link, end_link)
        print "Joint angles:", js_kin.get_joint_angles()
        print "Joint angles (wrapped):", js_kin.get_joint_angles(True)
        print "Joint velocities:", js_kin.get_joint_velocities()
        print "Joint efforts:", js_kin.get_joint_efforts()
        print "Jacobian:", js_kin.jacobian()
        kdl_pose = js_kin.forward()
        print "FK:", kdl_pose
        print "End effector force:", js_kin.end_effector_force()

        if True:
            import tf 
            from hrl_geom.pose_converter import PoseConv
            tf_list = tf.TransformListener()
            rospy.sleep(1)
            t = tf_list.getLatestCommonTime(base_link, end_link)
            tf_pose = PoseConv.to_homo_mat(tf_list.lookupTransform(base_link, end_link, t))
            print "Error with TF:", np.linalg.norm(kdl_pose - tf_pose)



if __name__ == "__main__":
    main()
