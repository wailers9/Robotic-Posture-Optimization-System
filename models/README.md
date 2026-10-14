# Demonstration robot model

`cr7.urdf` and `meshes/*.STL` originate from the Dobot model assets:
https://github.com/Dobot-Arm/TCP-IP-ROS-6AXis/tree/feature/v4-optimization/dobot_description

The retained model license is in `LICENSE`. Joint origins, axes and angular limits are retained. Mesh references use paths relative to this directory. Application-specific attached tool cylinders were removed. The TCP and robot mounting transform are configured per task, independently of this example model.

The model provides a runnable six-joint demonstration; the optimizer reads the selected chain dynamically and also supports bounded revolute redundant chains. Other robots can be supplied through an external URDF. Temporary Tesseract scene files add a world root, virtual TCP and configured obstacles without modifying the source URDF.

Mesh collision geometry uses convex hulls. Some source dynamics fields are zero; this project performs geometric path analysis and does not time-parameterize trajectories.
