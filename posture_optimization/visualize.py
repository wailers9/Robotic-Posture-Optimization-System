import numpy as np
from .optimize import coverage


def plot(points, qs, matrix, selected, backend, config, destination, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(12, 8))
    axes = figure.add_subplot(111, projection="3d")
    covered = coverage(matrix, selected)
    p = points*1000
    displayed = [p]
    axes.scatter(*p[~covered].T, s=12, c="#bcc6d0", alpha=.4, label="Uncovered")
    axes.scatter(*p[covered].T, s=22, c="#16a085", label="Covered union")
    for i in selected:
        transforms = backend.group.calcFwdKin(qs[i])
        chain = [backend.base_pose[:3, 3]]
        # Follow the configured joint chain, independent of link naming.
        graph = backend.robot.scene_graph
        for name in backend.joint_names:
            chain.append(np.asarray(transforms[graph.getJoint(name).child_link_name].translation))
        chain.append(backend.fk(qs[i])[:3, 3])
        line = np.asarray(chain)*1000
        displayed.append(line)
        axes.plot(*line.T, "o-", linewidth=2, markersize=4, label=f"Start {i}")
    axis = config.get("reference_axis")
    if axis:
        line = np.array([axis["start_mm"], axis["end_mm"]])
        displayed.append(line)
        axes.plot(*line.T, color="#d35400", label="Reference axis")
    axes.set(xlabel="World X (mm)", ylabel="World Y (mm)", zlabel="World Z (mm)",
             title=f"MovL coverage: {covered.sum()}/{len(points)} ({covered.mean():.1%}) | K={len(selected)}")
    axes.set_box_aspect(np.maximum(np.ptp(np.vstack(displayed), axis=0), 100))
    axes.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
