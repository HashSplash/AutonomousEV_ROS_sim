#/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray


if __name__=="__main__":
    rospy.init_node('keyboard',anonymous=True)
    pub=rospy.Publisher('/keyboard_interrupt',Float32MultiArray,queue_size=10)
    t=rospy.Rate(10)
    while not rospy.is_shutdown():
        pass

