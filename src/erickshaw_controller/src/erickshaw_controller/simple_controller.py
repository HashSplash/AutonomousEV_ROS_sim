#!usr/bin/env python3

import rospy
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates

w_gazebo = 0.1
w=0
s=0
i=0

def Acc(a=2):
    # s=w_gazebo
    global s
    global w
    t=abs(w-w_gazebo)/a
    T=t*10
    s=s+(w-w_gazebo)/T
    f=s
    if f<0.1:
        f=0
    return f


def Callback(msg):
    global w_gazebo
    global i
    if i==0:
        s=w_gazebo
        i=1
    w_gazebo =-(msg.twist[5].angular.y + msg.twist[4].angular.y)/2
    # w_gazebo =-msg.twist[5].angular.y
    # rospy.loginfo(w_gazebo)




if __name__=='__main__':
    rospy.init_node('simple_controller_py', anonymous=True)
    rospy.Subscriber("gazebo/link_states",LinkStates,Callback)
    pub_left= rospy.Publisher("wheel_left_controller/command",Float64, queue_size=10)
    pub_right= rospy.Publisher("wheel_right_controller/command", Float64, queue_size=10)
    t=rospy.Rate(10)
    # rospy.spin()
    while not rospy.is_shutdown():
        x=Float64(Acc())
        # x=Float64(0)
        rospy.loginfo("x= {} w_gazebo= {}".format(x,w_gazebo))
        pub_left.publish(x)
        pub_right.publish(x)
        t.sleep()