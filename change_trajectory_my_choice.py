import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def parse_vertex(line):
    parts = line.strip().split()
    vid = int(parts[1])
    pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
    quat = np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])
    return vid, pos, quat

def parse_edge(line):
    parts = line.strip().split()
    id1 = int(parts[1])
    id2 = int(parts[2])
    trans = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
    quat = np.array([float(parts[6]), float(parts[7]), float(parts[8]), float(parts[9])])
    info = np.array(list(map(float, parts[10:])))
    return id1, id2, trans, quat, info

def pose_to_transform(pos, quat):
    rot = R.from_quat(quat)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T

def transform_to_pose(T):
    pos = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    quat = rot.as_quat()
    return pos, quat

def invert_transform(T):
    R_inv = T[:3, :3].T
    t_inv = -R_inv @ T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def split_robot_vertex_ids(vertex_ids, threshold=1000):
    vertex_ids = sorted(vertex_ids)
    groups = []
    current_group = [vertex_ids[0]]

    for i in range(1, len(vertex_ids)):
        if vertex_ids[i] - vertex_ids[i-1] > threshold:
            groups.append(current_group)
            current_group = [vertex_ids[i]]
        else:
            current_group.append(vertex_ids[i])
    groups.append(current_group)
    return groups

def main(input_file, output_file, robot_index, threshold, mean, std):
    vertices = {}
    edges = []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    vertex_lines = []
    edge_lines = []
    other_lines = []

    for line in lines:
        if line.startswith("VERTEX_SE3:QUAT"):
            vertex_lines.append(line)
        elif line.startswith("EDGE_SE3:QUAT"):
            edge_lines.append(line)
        else:
            other_lines.append(line)

    for line in vertex_lines:
        vid, pos, quat = parse_vertex(line)
        vertices[vid] = {'pos': pos, 'quat': quat}

    all_vertex_ids = sorted(vertices.keys())
    groups = split_robot_vertex_ids(all_vertex_ids, threshold=threshold)

    if robot_index >= len(groups):
        print(f"❌ ERRORE: hai richiesto il robot {robot_index} ma ci sono solo {len(groups)} robot rilevati.")
        return

    selected_ids = set(groups[robot_index])
    print(f"✅ Robot {robot_index} identificato con {len(selected_ids)} vertici. ID da {min(selected_ids)} a {max(selected_ids)}.")

    # Applica drift
    np.random.seed(42)
    for vid in selected_ids:
        drift_xy = np.random.normal(loc=mean, scale=std, size=2)
        vertices[vid]['pos'][0] += drift_xy[0]
        vertices[vid]['pos'][1] += drift_xy[1]

    # Ricostruzione edges
    new_edge_lines = []
    for line in edge_lines:
        id1, id2, trans, quat, info = parse_edge(line)
        if id1 in selected_ids and id2 in selected_ids:
            T1 = pose_to_transform(vertices[id1]['pos'], vertices[id1]['quat'])
            T2 = pose_to_transform(vertices[id2]['pos'], vertices[id2]['quat'])
            T_rel = invert_transform(T1) @ T2
            pos_new, quat_new = transform_to_pose(T_rel)
            info_str = ' '.join(f'{v:.6f}' for v in info)
            new_line = f"EDGE_SE3:QUAT {id1} {id2} {pos_new[0]:.6f} {pos_new[1]:.6f} {pos_new[2]:.6f} {quat_new[0]:.6f} {quat_new[1]:.6f} {quat_new[2]:.6f} {quat_new[3]:.6f} {info_str}\n"
            new_edge_lines.append(new_line)
        else:
            new_edge_lines.append(line)

    with open(output_file, 'w') as f:
        for line in other_lines:
            f.write(line)
        for vid in sorted(vertices.keys()):
            p = vertices[vid]['pos']
            q = vertices[vid]['quat']
            f.write(f"VERTEX_SE3:QUAT {vid} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        for line in new_edge_lines:
            f.write(line)

    print(f"✅ File '{output_file}' generato con drift su robot {robot_index} (soglia salto ID = {threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Applica drift gaussiano a un robot selezionato (identificato da salti nei vertici).")
    parser.add_argument("input_file", help="File .g2o in input")
    parser.add_argument("output_file", help="File .g2o in output")
    parser.add_argument("--robot_index", type=int, required=True, help="Indice del robot da modificare (es. 0 per il primo robot trovato)")
    parser.add_argument("--threshold", type=int, default=1000, help="Soglia per salto tra ID vertici per distinguere i robot")
    parser.add_argument("--mean", type=float, default=1, help="Media del drift gaussiano")
    parser.add_argument("--std", type=float, default=0.5, help="Deviazione standard del drift")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.robot_index, args.threshold, args.mean, args.std)

#example
#python3 change_trajectory_my_choice.py graph.g2o graph_robot2_modified.g2o --robot_index 2 --mean 0.2 --std 0.1
#index goes from 0 on

