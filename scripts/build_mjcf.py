
#!/usr/bin/env python3
"""Generate two native MJCF models directly from the supplied PIVOT Xacro.

Only the explicitly defined links/joints in pivot.urdf.xacro are used.
The three missing ROS xacro:include files are not required for balance physics.
No ROS 2 or xacro installation needed. Collision and inertia come from the Xacro.
"""
from pathlib import Path
import json
from math import sin, cos
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/urdf/pivot.urdf.xacro'
OUT = ROOT / 'models'
MESHES = ROOT / 'assets/meshes'


def rpy_to_quat(rpy):
    roll, pitch, yaw = [float(n) for n in rpy.split()]
    cr, sr = cos(roll/2), sin(roll/2)
    cp, sp = cos(pitch/2), sin(pitch/2)
    cy, sy = cos(yaw/2), sin(yaw/2)
    return ' '.join(f'{v:.10g}' for v in (
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ))


def get_origin(parent):
    child = parent.find('origin') if parent is not None else None
    xyz = '0 0 0' if child is None else child.get('xyz', '0 0 0')
    rpy = '0 0 0' if child is None else child.get('rpy', '0 0 0')
    return xyz, rpy_to_quat(rpy)


def inertial(body, link):
    inert = link.find('inertial')
    if inert is None:
        raise ValueError(f'Missing inertial for {link.get("name")}')
    xyz, quat = get_origin(inert)
    i = inert.find('inertia')

    # FIX: MuJoCo allows orientation with diaginertia,
    # but not with fullinertia.
    # The supplied PIVOT Xacro has zero products of inertia.
    off_diagonal = (
        float(i.get('ixy', '0')),
        float(i.get('ixz', '0')),
        float(i.get('iyz', '0'))
    )

    if any(abs(v) > 1e-12 for v in off_diagonal):
        raise ValueError(
            f'Non-diagonal inertia for {link.get("name")}: '
            'diagonalize the inertia tensor and rotate the inertial frame '
            'before exporting to MuJoCo.'
        )

    diagonal = ' '.join(i.get(k) for k in ('ixx', 'iyy', 'izz'))

    ET.SubElement(
        body, 'inertial',
        pos=xyz,
        quat=quat,
        mass=inert.find('mass').get('value'),
        diaginertia=diagonal
    )


def collision(body, link, name):
    col = link.find('collision')
    if col is None:
        return

    xyz, quat = get_origin(col)
    geo = col.find('geometry')
    box, cylinder = geo.find('box'), geo.find('cylinder')

    common = dict(
        name='collision_' + name,
        pos=xyz,
        quat=quat,
        group='3',
        contype='1',
        conaffinity='1'
    )

    if box is not None:
        size = [float(v) / 2 for v in box.get('size').split()]
        ET.SubElement(
            body, 'geom',
            **common,
            type='box',
            size=' '.join(f'{v:.9g}' for v in size),
            rgba='0.16 0.50 0.72 0.22'
        )
    elif cylinder is not None:
        ET.SubElement(
            body, 'geom',
            **common,
            type='cylinder',
            size=f'{cylinder.get("radius")} {float(cylinder.get("length"))/2:.9g}',
            friction='1.25 0.005 0.0001',
            rgba='0.10 0.10 0.11 0.85'
        )
    else:
        raise ValueError(f'Unimplemented collision primitive in {name}')


def visual(body, link, name, assets, enabled):
    if not enabled:
        return

    vis = link.find('visual')
    if vis is None:
        return

    mesh_el = vis.find('./geometry/mesh')
    if mesh_el is None:
        return

    mesh_name = name + '_mesh'
    filename = mesh_el.get('filename').split('/')[-1]
    mesh_path = MESHES / filename

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    ET.SubElement(assets, 'mesh', name=mesh_name, file=filename)

    xyz, quat = get_origin(vis)
    material = vis.find('./material/color')
    rgba = material.get('rgba') if material is not None else '0.6 0.6 0.6 1'

    ET.SubElement(
        body, 'geom',
        name='visual_' + name,
        type='mesh',
        mesh=mesh_name,
        pos=xyz,
        quat=quat,
        rgba=rgba,
        group='1',
        contype='0',
        conaffinity='0'
    )


def generate(enabled):
    source = ET.parse(SOURCE).getroot()
    links = {l.get('name'): l for l in source.findall('link')}
    joints = {j.get('name'): j for j in source.findall('joint')}

    if not {'base_link', 'left_wheel_link', 'right_wheel_link'}.issubset(links):
        raise ValueError('Expected links missing from the source')

    if not {'base_joint', 'left_wheel_joint', 'right_wheel_joint'}.issubset(joints):
        raise ValueError('Expected joints missing from the source')

    m = ET.Element('mujoco', model='PIVOT-Reflex')

    ET.SubElement(
        m, 'compiler',
        angle='radian',
        autolimits='true',
        meshdir='../assets/meshes',
        discardvisual='false',
        fusestatic='false',
        inertiafromgeom='false'
    )

    ET.SubElement(
        m, 'option',
        timestep='0.002',
        gravity='0 0 -9.81',
        integrator='implicitfast'
    )

    ET.SubElement(m, 'statistic', center='0 0 0.12', extent='0.75')

    ET.SubElement(m, 'visual').append(
        ET.Element('headlight', ambient='0.6 0.6 0.6')
    )

    assets = ET.SubElement(m, 'asset')

    # Dark checkerboard ground
    ET.SubElement(
        assets, 'texture',
        name='ground_checker',
        type='2d',
        builtin='checker',
        rgb1='0.11 0.17 0.23',
        rgb2='0.19 0.27 0.34',
        width='512',
        height='512'
    )

    ET.SubElement(
        assets, 'material',
        name='ground_material',
        texture='ground_checker',
        texrepeat='12 12',
        texuniform='true',
        reflectance='0'
    )

    world = ET.SubElement(m, 'worldbody')

    ET.SubElement(
        world, 'light',
        pos='0 -2 3',
        directional='true',
        dir='0 0 -1'
    )

    ET.SubElement(
        world, 'geom',
        name='floor',
        type='plane',
        pos='0 0 0',
        size='2 2 0.1',
        material='ground_material',
        friction='1.25 0.005 0.0001',
        condim='3'
    )

    root_xyz, root_quat = get_origin(joints['base_joint'])
    base = ET.SubElement(
        world, 'body',
        name='base_link',
        pos=root_xyz,
        quat=root_quat
    )

    ET.SubElement(base, 'freejoint', name='root')
    inertial(base, links['base_link'])
    collision(base, links['base_link'], 'base_link')
    visual(base, links['base_link'], 'base_link', assets, enabled)

    for n in ('left_wheel_joint', 'right_wheel_joint'):
        joint = joints[n]
        linkname = joint.find('child').get('link')
        pos, quat = get_origin(joint)

        body = ET.SubElement(
            base, 'body',
            name=linkname,
            pos=pos,
            quat=quat
        )

        ET.SubElement(
            body, 'joint',
            name=n,
            type='hinge',
            axis=joint.find('axis').get('xyz'),
            limited='false',
            damping=joint.find('dynamics').get('damping', '0'),
            frictionloss=joint.find('dynamics').get('friction', '0')
        )

        inertial(body, links[linkname])
        collision(body, links[linkname], linkname)
        visual(body, links[linkname], linkname, assets, enabled)

    # Fixed links attached to the floating base, not the world.
    for name, joint in joints.items():
        if joint.get('type') != 'fixed' or name == 'base_joint':
            continue

        linkname = joint.find('child').get('link')
        pos, quat = get_origin(joint)

        body = ET.SubElement(
            base, 'body',
            name=linkname,
            pos=pos,
            quat=quat
        )

        inertial(body, links[linkname])
        visual(body, links[linkname], linkname, assets, enabled)

        if linkname == 'imu_link':
            ET.SubElement(
                body, 'site',
                name='imu_site',
                pos='0 0 0',
                size='0.004',
                rgba='1 0.15 0.15 1'
            )

    actuators = ET.SubElement(m, 'actuator')

    # Motor controls represent output-joint torque in Nm.
    profile = json.loads(
        (ROOT / 'configs/motor_md520z30.json').read_text()
    )

    max_stall_torque = float(profile['stall_torque_nm_published'])

    for side in ('left', 'right'):
        ET.SubElement(
            actuators, 'motor',
            name=side + '_motor',
            joint=side + '_wheel_joint',
            gear='1',
            ctrllimited='true',
            ctrlrange=f'{-max_stall_torque:.6f} {max_stall_torque:.6f}'
        )

    sensors = ET.SubElement(m, 'sensor')

    ET.SubElement(
        sensors, 'gyro',
        name='imu_gyro',
        site='imu_site'
    )

    ET.SubElement(
        sensors, 'accelerometer',
        name='imu_accel',
        site='imu_site'
    )

    ET.indent(m, space='  ')

    dest = OUT / ('pivot_visual.xml' if enabled else 'pivot_train.xml')

    ET.ElementTree(m).write(
        dest,
        encoding='unicode',
        xml_declaration=True
    )

    print(f'Created {dest.relative_to(ROOT)}')


if __name__ == '__main__':
    for graphics in (False, True):
        generate(graphics)

    print('ROS-specific included Xacro plugins were intentionally not imported.')
    print('Validate the resulting MJCFs with scripts/smoke_test.py when MuJoCo is installed.')
