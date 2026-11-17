#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import rospy
import tf.transformations as tft
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from std_msgs.msg import String


MESH_SAMPLE_CACHE = {}


def pose_values(elem):
    if elem is None or not elem.text:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    vals = [float(v) for v in elem.text.split()]
    return (vals + [0.0] * 6)[:6]


def pose_matrix(vals):
    mat = tft.euler_matrix(vals[3], vals[4], vals[5])
    mat[0, 3], mat[1, 3], mat[2, 3] = vals[0], vals[1], vals[2]
    return mat


def scale_matrix(vals):
    sx, sy, sz = (vals + [1.0, 1.0, 1.0])[:3]
    mat = np.identity(4)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    return mat


def text_floats(elem, default):
    if elem is None or not elem.text:
        return default
    return [float(v) for v in elem.text.split()]


def transform_points(mat, pts):
    if not pts:
        return []
    arr = np.asarray(pts, dtype=float)
    homo = np.ones((arr.shape[0], 4), dtype=float)
    homo[:, 0:3] = arr[:, 0:3]
    out = homo.dot(mat.T)
    return [(float(p[0]), float(p[1]), float(p[2])) for p in out[:, 0:3]]


def xml_name(elem):
    return elem.tag.split("}", 1)[-1]


def direct_children(elem, name):
    return [child for child in list(elem) if xml_name(child) == name]


def first_child(elem, name):
    for child in list(elem):
        if xml_name(child) == name:
            return child
    return None


def text_values(elem):
    if elem is None or not elem.text:
        return []
    return [float(v) for v in elem.text.split()]


def frange_closed(start, stop, step, max_count=80):
    if step <= 0:
        return [start, stop]
    count = max(2, int(math.ceil((stop - start) / step)) + 1)
    count = min(count, max_count)
    return [start + (stop - start) * i / float(count - 1) for i in range(count)]


def sample_box(size, resolution, max_axis_samples=80):
    sx, sy, sz = [max(float(v), 0.001) for v in size]
    xs = frange_closed(-sx * 0.5, sx * 0.5, resolution, max_axis_samples)
    ys = frange_closed(-sy * 0.5, sy * 0.5, resolution, max_axis_samples)
    zs = frange_closed(-sz * 0.5, sz * 0.5, resolution, max_axis_samples)
    pts = []
    for x in xs:
        for y in ys:
            pts.append((x, y, -sz * 0.5))
            pts.append((x, y, sz * 0.5))
    for x in xs:
        for z in zs:
            pts.append((x, -sy * 0.5, z))
            pts.append((x, sy * 0.5, z))
    for y in ys:
        for z in zs:
            pts.append((-sx * 0.5, y, z))
            pts.append((sx * 0.5, y, z))
    return pts


def sample_cylinder(radius, length, resolution, max_axis_samples=80):
    radius = max(float(radius), 0.001)
    length = max(float(length), 0.001)
    n_angle = min(max_axis_samples, max(16, int(math.ceil(2.0 * math.pi * radius / resolution))))
    zs = frange_closed(-length * 0.5, length * 0.5, resolution, max_axis_samples)
    rs = frange_closed(0.0, radius, resolution, max_axis_samples)
    pts = []
    for i in range(n_angle):
        a = 2.0 * math.pi * i / float(n_angle)
        ca, sa = math.cos(a), math.sin(a)
        for z in zs:
            pts.append((radius * ca, radius * sa, z))
        for r in rs:
            pts.append((r * ca, r * sa, -length * 0.5))
            pts.append((r * ca, r * sa, length * 0.5))
    return pts


def sample_sphere(radius, resolution, max_axis_samples=80):
    radius = max(float(radius), 0.001)
    n_theta = min(max_axis_samples, max(12, int(math.ceil(math.pi * radius / resolution))))
    n_phi = min(max_axis_samples, max(24, int(math.ceil(2.0 * math.pi * radius / resolution))))
    pts = []
    for i in range(n_theta + 1):
        theta = math.pi * i / float(n_theta)
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(n_phi):
            phi = 2.0 * math.pi * j / float(n_phi)
            pts.append((radius * st * math.cos(phi), radius * st * math.sin(phi), radius * ct))
    return pts


def parse_dae_legacy_vertices(path):
    root = ET.parse(path).getroot()
    sources = {}
    position_sources = set()
    for elem in root.iter():
        if elem.tag.endswith("source"):
            source_id = elem.get("id")
            float_array = None
            for child in elem:
                if child.tag.endswith("float_array"):
                    float_array = child
                    break
            if source_id and float_array is not None and float_array.text:
                vals = [float(v) for v in float_array.text.split()]
                sources["#" + source_id] = vals
        if elem.tag.endswith("vertices"):
            for child in elem:
                if child.tag.endswith("input") and child.get("semantic") == "POSITION":
                    source = child.get("source")
                    if source:
                        position_sources.add(source)

    vertices = []
    for source in sorted(position_sources):
        vals = sources.get(source, [])
        for i in range(0, len(vals) - 2, 3):
            vertices.append((vals[i], vals[i + 1], vals[i + 2]))
    return vertices


def dae_transform_from_node(node):
    mat = np.identity(4)
    for child in list(node):
        name = xml_name(child)
        vals = text_values(child)
        if name == "matrix" and len(vals) >= 16:
            mat = mat.dot(np.asarray(vals[:16], dtype=float).reshape((4, 4)))
        elif name == "translate" and len(vals) >= 3:
            trans = np.identity(4)
            trans[0, 3], trans[1, 3], trans[2, 3] = vals[0], vals[1], vals[2]
            mat = mat.dot(trans)
        elif name == "rotate" and len(vals) >= 4:
            mat = mat.dot(tft.rotation_matrix(math.radians(vals[3]), vals[:3]))
        elif name == "scale" and len(vals) >= 3:
            mat = mat.dot(scale_matrix(vals[:3]))
    return mat


def sample_triangle(a, b, c, resolution):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ab = b - a
    ac = c - a
    area = 0.5 * np.linalg.norm(np.cross(ab, ac))
    if area <= 1e-10:
        return [tuple(a.tolist())]
    steps = max(1, int(math.ceil(math.sqrt(area) / max(resolution, 1e-3))))
    pts = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            u = i / float(steps)
            v = j / float(steps)
            p = a + ab * u + ac * v
            pts.append((float(p[0]), float(p[1]), float(p[2])))
    return pts


def dae_unit_scale(root):
    asset = first_child(root, "asset")
    unit = first_child(asset, "unit") if asset is not None else None
    if unit is None:
        return 1.0
    try:
        return float(unit.get("meter", "1.0"))
    except ValueError:
        return 1.0


def parse_dae_sources(mesh, unit_scale):
    sources = {}
    for source in direct_children(mesh, "source"):
        source_id = source.get("id")
        float_array = first_child(source, "float_array")
        if not source_id or float_array is None or not float_array.text:
            continue
        vals = [float(v) * unit_scale for v in float_array.text.split()]
        stride = 3
        technique = first_child(source, "technique_common")
        accessor = first_child(technique, "accessor") if technique is not None else None
        if accessor is not None:
            try:
                stride = int(accessor.get("stride", "3"))
            except ValueError:
                stride = 3
        tuples = []
        for i in range(0, len(vals) - stride + 1, stride):
            if stride >= 3:
                tuples.append((vals[i], vals[i + 1], vals[i + 2]))
        sources["#" + source_id] = tuples
    return sources


def parse_dae_geometries(root):
    unit_scale = dae_unit_scale(root)
    geometries = {}
    for geometry in root.iter():
        if xml_name(geometry) != "geometry":
            continue
        geom_id = geometry.get("id")
        mesh = first_child(geometry, "mesh")
        if not geom_id or mesh is None:
            continue

        sources = parse_dae_sources(mesh, unit_scale)
        vertex_sources = {}
        for vertices in direct_children(mesh, "vertices"):
            vertices_id = vertices.get("id")
            if not vertices_id:
                continue
            for inp in direct_children(vertices, "input"):
                if inp.get("semantic") == "POSITION" and inp.get("source"):
                    vertex_sources["#" + vertices_id] = inp.get("source")

        triangles = []
        fallback_vertices = []
        for triangles_elem in direct_children(mesh, "triangles"):
            inputs = direct_children(triangles_elem, "input")
            if not inputs:
                continue
            offsets = []
            vertex_source = None
            for inp in inputs:
                try:
                    offset = int(inp.get("offset", "0"))
                except ValueError:
                    offset = 0
                offsets.append(offset)
                semantic = inp.get("semantic")
                source = inp.get("source")
                if semantic == "VERTEX":
                    vertex_source = vertex_sources.get(source, source)
                    vertex_offset = offset
                elif semantic == "POSITION":
                    vertex_source = source
                    vertex_offset = offset
            if not vertex_source:
                continue
            positions = sources.get(vertex_source, [])
            if not positions:
                continue
            stride = max(offsets) + 1
            p_elem = first_child(triangles_elem, "p")
            if p_elem is None or not p_elem.text:
                continue
            raw_indices = [int(v) for v in p_elem.text.split()]
            vertex_indices = []
            for i in range(0, len(raw_indices) - stride + 1, stride):
                idx = raw_indices[i + vertex_offset]
                if 0 <= idx < len(positions):
                    vertex_indices.append(idx)
            for i in range(0, len(vertex_indices) - 2, 3):
                tri = (
                    positions[vertex_indices[i]],
                    positions[vertex_indices[i + 1]],
                    positions[vertex_indices[i + 2]],
                )
                triangles.append(tri)

        for source in sources.values():
            fallback_vertices.extend(source)

        if triangles:
            geometries["#" + geom_id] = {"triangles": triangles, "vertices": fallback_vertices}
        elif fallback_vertices:
            geometries["#" + geom_id] = {"triangles": [], "vertices": fallback_vertices}
    return geometries, unit_scale


def collect_dae_instances(root, geometries):
    library_nodes = {}
    for lib_nodes in root.iter():
        if xml_name(lib_nodes) != "library_nodes":
            continue
        for node in direct_children(lib_nodes, "node"):
            node_id = node.get("id")
            if node_id:
                library_nodes["#" + node_id] = node

    visual_scenes = [elem for elem in root.iter() if xml_name(elem) == "visual_scene"]
    instances = []

    def walk(node, parent_mat):
        mat = parent_mat.dot(dae_transform_from_node(node))
        for child in list(node):
            name = xml_name(child)
            if name == "instance_geometry":
                geom = geometries.get(child.get("url"))
                if geom is not None:
                    instances.append((mat.copy(), geom))
            elif name == "instance_node":
                ref = library_nodes.get(child.get("url"))
                if ref is not None:
                    walk(ref, mat)
            elif name == "node":
                walk(child, mat)

    for scene in visual_scenes:
        for node in direct_children(scene, "node"):
            walk(node, np.identity(4))
    return instances


def parse_dae_mesh(path):
    root = ET.parse(path).getroot()
    geometries, unit_scale = parse_dae_geometries(root)
    instances = collect_dae_instances(root, geometries)
    if not instances:
        instances = [(np.identity(4), geom) for geom in geometries.values()]
    return instances, unit_scale, len(geometries)


def sample_dae_mesh(path, resolution, max_points, surface_mode=True):
    instances, unit_scale, geom_count = parse_dae_mesh(path)
    points = []
    tri_count = 0
    for mat, geom in instances:
        local = []
        if surface_mode and geom["triangles"]:
            for tri in geom["triangles"]:
                tri_count += 1
                local.extend(sample_triangle(tri[0], tri[1], tri[2], resolution))
        else:
            local.extend(geom["vertices"])
        points.extend(transform_points(mat, local))
        if len(points) >= max_points * 3:
            break
    downsampled = voxel_downsample(points, resolution, max_points)
    return downsampled, unit_scale, geom_count, len(instances), tri_count


def voxel_downsample(points, leaf_size, max_points):
    if not points:
        return []
    leaf_size = max(float(leaf_size), 1e-4)
    seen = set()
    out = []
    for p in points:
        key = (
            int(round(p[0] / leaf_size)),
            int(round(p[1] / leaf_size)),
            int(round(p[2] / leaf_size)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_points:
            break
    return out


def sample_mesh_vertices(mesh_path, resolution, max_points):
    cache_key = (mesh_path, float(resolution), int(max_points), "surface")
    if cache_key in MESH_SAMPLE_CACHE:
        return MESH_SAMPLE_CACHE[cache_key]
    try:
        result = sample_dae_mesh(mesh_path, resolution, max_points, surface_mode=True)
        MESH_SAMPLE_CACHE[cache_key] = result
        return result
    except Exception as exc:
        rospy.logwarn("static_scene_cloud: DAE surface sampling failed for %s: %s", mesh_path, exc)
        vertices = parse_dae_legacy_vertices(mesh_path)
        if not vertices:
            result = ([], 1.0, 0, 0, 0)
        else:
            result = (voxel_downsample(vertices, resolution, max_points), 1.0, 0, 0, 0)
        MESH_SAMPLE_CACHE[cache_key] = result
        return result


def sample_mesh_bbox(mesh_path, resolution, max_axis_samples=80):
    vertices = parse_dae_legacy_vertices(mesh_path)
    if not vertices:
        return []
    arr = np.asarray(vertices, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    size = maxs - mins
    center = (mins + maxs) * 0.5
    pts = sample_box(size, resolution, max_axis_samples)
    return [(p[0] + center[0], p[1] + center[1], p[2] + center[2]) for p in pts]


def model_search_paths(extra_paths):
    paths = []
    for value in os.environ.get("GAZEBO_MODEL_PATH", "").split(":"):
        if value:
            paths.append(value)
    paths.extend(extra_paths)
    paths.extend([
        os.path.expanduser("~/.gazebo/models"),
        "/home/uav/桌面/XTDrone_test/XTDrone/sitl_config/models",
        "/home/uav/桌面/XTDrone_Test2/catkin_ws/src/XTDrone/sitl_config/models",
    ])
    out = []
    seen = set()
    for path in paths:
        path = os.path.abspath(os.path.expanduser(path))
        if path not in seen and os.path.isdir(path):
            seen.add(path)
            out.append(path)
    return out


def resolve_model_uri(uri, search_paths):
    if uri.startswith("file://"):
        path = uri[len("file://"):]
        return path if os.path.exists(path) else None
    if uri.startswith("model://"):
        rel = uri[len("model://"):]
        for base in search_paths:
            path = os.path.join(base, rel)
            if os.path.exists(path):
                return path
        return None
    if os.path.exists(uri):
        return uri
    return None


def iter_geometry_elements(link, include_visuals, include_collisions):
    if include_collisions:
        for elem in link.findall("collision"):
            yield "collision", elem
    if include_visuals:
        for elem in link.findall("visual"):
            yield "visual", elem


def child_models(model):
    return [child for child in list(model) if child.tag == "model"]


def direct_links(model):
    return [child for child in list(model) if child.tag == "link"]


def sample_geometry(elem, resolution, search_paths, report, model_name, geom_kind, max_axis_samples, mesh_mode, max_mesh_points):
    geom = elem.find("geometry")
    if geom is None:
        return []
    local_mat = pose_matrix(pose_values(elem.find("pose")))

    box = geom.find("box/size")
    if box is not None and box.text:
        return transform_points(local_mat, sample_box(text_floats(box, [1.0, 1.0, 1.0]), resolution, max_axis_samples))

    cyl = geom.find("cylinder")
    if cyl is not None:
        radius = text_floats(cyl.find("radius"), [0.5])[0]
        length = text_floats(cyl.find("length"), [1.0])[0]
        return transform_points(local_mat, sample_cylinder(radius, length, resolution, max_axis_samples))

    sph = geom.find("sphere/radius")
    if sph is not None and sph.text:
        return transform_points(local_mat, sample_sphere(float(sph.text), resolution, max_axis_samples))

    mesh = geom.find("mesh")
    if mesh is not None:
        uri_elem = mesh.find("uri")
        if uri_elem is None or not uri_elem.text:
            report.append("%s %s mesh missing_uri" % (model_name, geom_kind))
            return []
        mesh_path = resolve_model_uri(uri_elem.text.strip(), search_paths)
        if mesh_path is None:
            report.append("%s %s mesh not_found %s" % (model_name, geom_kind, uri_elem.text.strip()))
            return []
        mesh_scale = scale_matrix(text_floats(mesh.find("scale"), [1.0, 1.0, 1.0]))
        if mesh_mode == "off":
            report.append("%s %s mesh skipped %s" % (model_name, geom_kind, mesh_path))
            return []
        if mesh_mode == "bbox":
            pts = sample_mesh_bbox(mesh_path, resolution, max_axis_samples)
            mode_label = "mesh_bbox"
            report.append("%s %s %s yes %s points=%d" % (model_name, geom_kind, mode_label, mesh_path, len(pts)))
        else:
            pts, unit_scale, geom_count, instance_count, tri_count = sample_mesh_vertices(mesh_path, resolution, max_mesh_points)
            mode_label = "mesh_surface"
            report.append(
                "%s %s %s yes %s unit_scale=%.6g geometries=%d instances=%d triangles=%d points=%d"
                % (model_name, geom_kind, mode_label, mesh_path, unit_scale, geom_count, instance_count, tri_count, len(pts))
            )
        if not pts:
            report.append("%s %s mesh no_vertices %s" % (model_name, geom_kind, mesh_path))
            return []
        return transform_points(local_mat.dot(mesh_scale), pts)

    return []


def export_scene_cloud(
    world_path,
    resolution,
    include_visuals,
    include_collisions,
    extra_model_paths,
    exclude_models,
    max_axis_samples=80,
    mesh_mode="vertices",
    max_mesh_points=20000,
):
    root = ET.parse(world_path).getroot()
    search_paths = model_search_paths(extra_model_paths)
    report = ["world=%s" % world_path, "resolution=%.3f" % resolution, "mesh_mode=%s" % mesh_mode]
    points = []
    world = root.find("./world")

    def export_model(model, parent_mat, parent_name=""):
        model_name = model.get("name", "model")
        full_name = ("%s/%s" % (parent_name, model_name)) if parent_name else model_name
        if model_name in exclude_models or any(model_name.startswith(prefix) for prefix in exclude_models):
            report.append("%s skipped" % full_name)
            return
        model_mat = parent_mat.dot(pose_matrix(pose_values(model.find("pose"))))
        model_count = 0
        for link in direct_links(model):
            link_pose = pose_matrix(pose_values(link.find("pose")))
            link_mat = link_pose if parent_name else model_mat.dot(link_pose)
            for geom_kind, elem in iter_geometry_elements(link, include_visuals, include_collisions):
                local_points = sample_geometry(
                    elem,
                    resolution,
                    search_paths,
                    report,
                    full_name,
                    geom_kind,
                    max_axis_samples,
                    mesh_mode,
                    max_mesh_points,
                )
                world_points = transform_points(link_mat, local_points)
                points.extend(world_points)
                model_count += len(world_points)
        for child in child_models(model):
            before = len(points)
            export_model(child, model_mat, full_name)
            model_count += len(points) - before
        report.append("%s points=%d" % (full_name, model_count))

    for model in direct_children(world, "model") if world is not None else []:
        export_model(model, np.identity(4))
    return points, report


def cloud_msg(frame_id, points):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    return point_cloud2.create_cloud(header, fields, points)


class StaticSceneCloudNode:
    def __init__(self):
        self.world = rospy.get_param("~world", "/home/uav/桌面/XTDrone_test/XTDrone/sitl_config/worlds/indoor1.world")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.resolution = float(rospy.get_param("~resolution", 0.08))
        self.include_visuals = bool(rospy.get_param("~include_visuals", True))
        self.include_collisions = bool(rospy.get_param("~include_collisions", True))
        self.max_axis_samples = int(rospy.get_param("~max_axis_samples", 80))
        self.mesh_mode = rospy.get_param("~mesh_mode", "vertices")
        self.max_mesh_points = int(rospy.get_param("~max_mesh_points", 20000))
        self.extra_model_paths = rospy.get_param("~extra_model_paths", [])
        self.exclude_models = rospy.get_param("~exclude_models", ["ground_plane"])
        self.cloud_pub = rospy.Publisher("/world/static_scene_cloud", PointCloud2, queue_size=1, latch=True)
        self.report_pub = rospy.Publisher("/world/static_scene_cloud_report", String, queue_size=1, latch=True)
        rospy.loginfo("static_scene_cloud: exporting %s", self.world)
        self.points, self.report = export_scene_cloud(
            self.world,
            self.resolution,
            self.include_visuals,
            self.include_collisions,
            self.extra_model_paths,
            self.exclude_models,
            self.max_axis_samples,
            self.mesh_mode,
            self.max_mesh_points,
        )
        self.report.append("total_points=%d" % len(self.points))
        self.cloud_msg = cloud_msg(self.frame_id, self.points)
        self.report_msg = String(data="\n".join(self.report))
        rospy.loginfo("static_scene_cloud: total_points=%d", len(self.points))

    def spin(self):
        period = 1.0 / max(float(rospy.get_param("~rate_hz", 1.0)), 1e-3)
        while not rospy.is_shutdown():
            self.cloud_msg.header.stamp = rospy.Time.now()
            self.cloud_pub.publish(self.cloud_msg)
            self.report_pub.publish(self.report_msg)
            rospy.loginfo_throttle(10.0, "static_scene_cloud: publishing points=%d", len(self.points))
            time.sleep(period)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--offline":
        parser = argparse.ArgumentParser()
        parser.add_argument("--world", required=True)
        parser.add_argument("--resolution", type=float, default=0.08)
        parser.add_argument("--max-axis-samples", type=int, default=80)
        parser.add_argument("--mesh-mode", default="vertices", choices=["vertices", "bbox", "off"])
        parser.add_argument("--max-mesh-points", type=int, default=20000)
        args = parser.parse_args(sys.argv[2:])
        points, report = export_scene_cloud(
            args.world,
            args.resolution,
            True,
            True,
            [],
            ["ground_plane"],
            args.max_axis_samples,
            args.mesh_mode,
            args.max_mesh_points,
        )
        print("\n".join(report))
        print("total_points=%d" % len(points))
        return
    rospy.init_node("static_scene_cloud_node")
    StaticSceneCloudNode().spin()


if __name__ == "__main__":
    main()
