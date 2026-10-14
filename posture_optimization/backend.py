"""Tesseract adapter: URDF, KDL IK/FK/Jacobian, Bullet contacts and Descartes."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from .geometry import transform, axis_frame


class TesseractBackend:
    def __init__(self, config):
        from tesseract_robotics import ensure_configured
        from tesseract_robotics.planning import Robot
        ensure_configured()
        self.config = config
        self._temp = tempfile.TemporaryDirectory(prefix="movl_")
        self.directory = Path(self._temp.name)
        self._prepare()
        self.robot = Robot.from_files(self.directory/"scene.urdf", self.directory/"scene.srdf")
        self.group = self.robot.env.getKinematicGroup("manipulator")
        self.joint_names = list(self.group.getJointNames())
        if len(self.joint_names) < 6:
            raise ValueError("Full-pose optimization requires at least six revolute joints")
        self.limits = np.array(self.group.getLimits().joint_limits)
        if self.limits.shape != (len(self.joint_names), 2) or not np.isfinite(self.limits).all():
            raise ValueError("Finite joint limits are required")
        self.robot.set_collision_margin(config["validation"]["collision_margin_mm"]/1000)
        self.manager = self.robot.env.getDiscreteContactManager()
        if self.manager is None:
            raise RuntimeError("No Tesseract discrete collision manager available")
        self.manager.setActiveCollisionObjects(self.robot.env.getActiveLinkNames())
        self.continuous_manager = self.robot.env.getContinuousContactManager()
        if self.continuous_manager is None:
            raise RuntimeError("No Tesseract continuous collision manager available")
        self.continuous_manager.setActiveCollisionObjects(self.robot.env.getActiveLinkNames())
        self.continuous_manager.setCollisionObjectsTransform(self.robot.env.getState().link_transforms)
        self.base_pose = transform(config["robot"]["base_pose_mm_deg"])

    def _prepare(self):
        c = self.config
        source = Path(c["robot"]["urdf"])
        root = ET.parse(source).getroot()
        root.set("xmlns:tesseract", "https://github.com/tesseract-robotics/tesseract")
        root.set("tesseract:make_convex", "true")
        removed = set(c["robot"].get("exclude_links", []))
        for item in list(root):
            if (item.tag == "link" and item.get("name") in removed) or (item.tag == "joint" and
                    (item.find("child").get("link") in removed or item.find("parent").get("link") in removed)):
                root.remove(item)
        existing_links = {x.get("name") for x in root.findall("link")}
        if "optimization_world" in existing_links or "optimization_tcp" in existing_links:
            raise ValueError("URDF uses reserved optimization link names")
        parents = {j.find("child").get("link") for j in root.findall("joint")}
        roots = existing_links-parents
        if len(roots) != 1:
            raise ValueError("URDF must be one connected rooted tree")
        root_link = roots.pop()
        tip = c["robot"]["tip_link"]
        if tip not in existing_links:
            raise ValueError("tip_link does not exist")
        joint_by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        current = tip
        while current in joint_by_child:
            joint = joint_by_child[current]
            if joint.get("type") not in ("fixed", "revolute"):
                raise ValueError("Selected chain supports bounded revolute joints only")
            current = joint.find("parent").get("link")
        for mesh in root.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename.startswith("package://"):
                package, relative = filename[len("package://"):].split("/", 1)
                packages = c["robot"].get("package_paths", {})
                if package not in packages:
                    raise ValueError(f"Set robot.package_paths for {package}")
                resolved = source.parent / packages[package] / relative
            elif filename.startswith("file://"):
                from urllib.parse import urlparse, unquote
                resolved = Path(unquote(urlparse(filename).path).lstrip("/") if __import__("os").name == "nt" else unquote(urlparse(filename).path))
            else:
                resolved = source.parent / filename
            if not resolved.is_file():
                raise ValueError(f"Mesh does not exist: {resolved}")
            mesh.set("filename", str(resolved.resolve()).replace("\\", "/"))
        ET.SubElement(root, "link", name="optimization_world")
        self._fixed(root, "optimization_base", "optimization_world", root_link, c["robot"]["base_pose_mm_deg"])
        ET.SubElement(root, "link", name="optimization_tcp")
        self._fixed(root, "optimization_tcp_joint", tip, "optimization_tcp", c["robot"]["tcp_pose_mm_deg"])
        # Fixed environment geometry is expressed in the world frame.
        obstacles = list(c.get("obstacles", []))
        if c.get("reference_axis", {}).get("collision_enabled", False):
            a, axis, length = axis_frame(c["reference_axis"])
            # Cylinder is symmetric; any twist about its axis is equivalent.
            rotation = Rotation.align_vectors([axis], [[0, 0, 1]])[0]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rpy = rotation.as_euler("xyz", degrees=True)
            obstacles.append({"name": "reference_axis", "kind": "cylinder", "radius_mm": c["reference_axis"]["physical_radius_mm"],
                              "length_mm": length*1000, "pose_mm_deg": [*((a+axis*length/2)*1000), *rpy]})
        for i, obstacle in enumerate(obstacles):
            name = f"optimization_obstacle_{i}"
            link = ET.SubElement(root, "link", name=name)
            geometry = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
            kind = obstacle["kind"]
            if kind == "box":
                ET.SubElement(geometry, "box", size=" ".join(str(v/1000) for v in obstacle["size_mm"]))
            elif kind == "sphere":
                ET.SubElement(geometry, "sphere", radius=str(obstacle["radius_mm"]/1000))
            elif kind == "cylinder":
                ET.SubElement(geometry, "cylinder", radius=str(obstacle["radius_mm"]/1000), length=str(obstacle["length_mm"]/1000))
            else:
                raise ValueError(f"Unsupported obstacle: {kind}")
            self._fixed(root, name+"_joint", "optimization_world", name, obstacle["pose_mm_deg"])
        ET.ElementTree(root).write(self.directory/"scene.urdf", encoding="utf-8", xml_declaration=True)
        srdf = ET.Element("robot", name=root.get("name"))
        group = ET.SubElement(srdf, "group", name="manipulator")
        ET.SubElement(group, "chain", base_link="optimization_world", tip_link="optimization_tcp")
        for joint in root.findall("joint"):
            ET.SubElement(srdf, "disable_collisions", link1=joint.find("parent").get("link"),
                          link2=joint.find("child").get("link"), reason="Adjacent")
        for left, right in c["robot"].get("allowed_collision_pairs", []):
            if left not in existing_links or right not in existing_links:
                raise ValueError("Allowed collision pair refers to missing links")
            ET.SubElement(srdf, "disable_collisions", link1=left, link2=right, reason="Configured mounting contact")
        plugins = {"kinematic_plugins": {"search_libraries": ["tesseract_kinematics_kdl_factories"]}}
        for kind, plugin in (("fwd", "KDLFwdKinChain"), ("inv", "KDLInvKinChainLMA")):
            plugins["kinematic_plugins"][kind+"_kin_plugins"] = {"manipulator": {"default": plugin, "plugins": {
                plugin: {"class": plugin+"Factory", "config": {"base_link": "optimization_world", "tip_link": "optimization_tcp"}}}}}
        plugin_path = self.directory/"kinematics.yaml"
        plugin_path.write_text(yaml.safe_dump(plugins), encoding="utf-8")
        ET.SubElement(srdf, "kinematics_plugin_config", filename=str(plugin_path).replace("\\", "/"))
        ET.SubElement(srdf, "contact_managers_plugin_config", filename="package://tesseract/support/urdf/contact_manager_plugins.yaml")
        ET.ElementTree(srdf).write(self.directory/"scene.srdf", encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _fixed(root, name, parent, child, pose):
        joint = ET.SubElement(root, "joint", name=name, type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz=" ".join(str(v/1000) for v in pose[:3]),
                      rpy=" ".join(str(np.deg2rad(v)) for v in pose[3:]))

    def fk(self, q):
        return np.asarray(self.group.calcFwdKin(np.asarray(q, dtype=float))["optimization_tcp"].matrix)

    def jacobian(self, q):
        return np.asarray(self.group.calcJacobian(np.asarray(q, dtype=float), "optimization_tcp"))

    def ik(self, position, rotation, seed):
        from tesseract_robotics.tesseract_kinematics import KinGroupIKInput
        from tesseract_robotics.planning import Pose
        pose = Pose.from_matrix_position(rotation, position)
        solutions = self.group.calcInvKin(KinGroupIKInput(pose, "optimization_world", "optimization_tcp"), np.asarray(seed, dtype=float))
        if not solutions:
            return None
        near = np.asarray(seed)
        # KDL can return angles beyond one turn. Choose equivalent angles near seed,
        # but only within literal URDF bounds. Never wrap a trajectory's differences.
        valid = []
        for solution in solutions:
            q = np.asarray(solution).copy()
            for i in range(len(self.joint_names)):
                turns = np.arange(np.ceil((self.limits[i, 0]-q[i])/(2*np.pi)), np.floor((self.limits[i, 1]-q[i])/(2*np.pi))+1)
                if len(turns) == 0:
                    break
                choices = q[i]+2*np.pi*turns
                q[i] = choices[np.argmin(abs(choices-near[i]))]
            else:
                valid.append(q)
        return min(valid, key=lambda q: np.linalg.norm(q-near)) if valid else None

    def collision(self, q):
        from tesseract_robotics.tesseract_collision import ContactRequest, ContactResultMap, ContactTestType_ALL
        state = self.robot.env.getStateSolver().getState(self.joint_names, np.asarray(q, dtype=float))
        self.manager.setCollisionObjectsTransform(state.link_transforms)
        result = ContactResultMap()
        self.manager.contactTest(result, ContactRequest(ContactTestType_ALL))
        return result.size() != 0

    def descartes(self, q, positions, rotation):
        from tesseract_robotics.planning import MotionProgram, StateTarget, CartesianTarget, Pose
        from tesseract_robotics.planning.profiles import create_descartes_default_profiles
        from tesseract_robotics.tesseract_motion_planners import PlannerRequest
        from tesseract_robotics.tesseract_motion_planners_descartes import DescartesMotionPlannerD, DescartesDefaultPlanProfileD
        from tesseract_robotics.tesseract_command_language import (
            InstructionPoly_as_MoveInstructionPoly, WaypointPoly_as_StateWaypointPoly)
        self.robot.set_joints(q, self.joint_names)
        program = MotionProgram("manipulator", tcp_frame="optimization_tcp", working_frame="optimization_world").set_joint_names(self.joint_names)
        program.start_at(StateTarget(q, names=self.joint_names))
        for position in positions:
            program.linear_to(CartesianTarget(Pose.from_matrix_position(rotation, position)))
        # Direct geometric Descartes API avoids time parameterization: upstream
        # This project optimizes geometric reach rather than timing.
        request = PlannerRequest()
        request.instructions = program.to_composite_instruction()
        request.env = self.robot.env
        collision_check = self.config["validation"].get("collision_check", False)
        profile = DescartesDefaultPlanProfileD()
        profile.enable_collision = collision_check
        profile.enable_edge_collision = collision_check
        profile.allow_collision = not collision_check
        request.profiles = create_descartes_default_profiles(num_threads=1, move_profile=profile)
        result = DescartesMotionPlannerD("DescartesMotionPlannerTask").solve(request)
        if not result.successful:
            raise RuntimeError(result.message or "Descartes failed")
        trajectory = []
        for instruction in result.results:
            move = InstructionPoly_as_MoveInstructionPoly(instruction)
            state = WaypointPoly_as_StateWaypointPoly(move.getWaypoint())
            positions = dict(zip(state.getNames(), state.getPosition()))
            raw = np.asarray([positions[name] for name in self.joint_names], float)
            near = trajectory[-1] if trajectory else q
            for i in range(len(self.joint_names)):
                turns = np.arange(np.ceil((self.limits[i, 0]-raw[i])/(2*np.pi)),
                                  np.floor((self.limits[i, 1]-raw[i])/(2*np.pi))+1)
                if len(turns):
                    equivalents = raw[i]+2*np.pi*turns
                    raw[i] = equivalents[np.argmin(abs(equivalents-near[i]))]
            trajectory.append(raw)
        return np.asarray(trajectory)

    def segment_collision(self, a, b):
        from tesseract_robotics.tesseract_collision import ContactRequest, ContactResultMap, ContactTestType_ALL
        solver = self.robot.env.getStateSolver()
        start = solver.getState(self.joint_names, np.asarray(a, float)).link_transforms
        end = solver.getState(self.joint_names, np.asarray(b, float)).link_transforms
        active = self.robot.env.getActiveLinkNames()
        self.continuous_manager.setCollisionObjectsTransformCast(
            {name: start[name] for name in active}, {name: end[name] for name in active})
        result = ContactResultMap()
        self.continuous_manager.contactTest(result, ContactRequest(ContactTestType_ALL))
        return result.size() != 0

    def close(self):
        self._temp.cleanup()
