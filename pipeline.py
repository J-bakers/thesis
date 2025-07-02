import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import csv
import subprocess 

# === STEP 0: CSV → TXT ===
def convert_csv_to_txt(csv_file, txt_file):
    with open(csv_file, 'r') as f_in, open(txt_file, 'w') as f_out:
        reader = csv.reader(f_in)
        for row in reader:
            row = [field.strip() for field in row if field.strip() != '']
            f_out.write(' '.join(row) + '\n')
    print(f"✅ File CSV convertito in TXT: {txt_file}")

# === STEP 1: Rimuove la prima colonna ===
def remove_first_column(lines):
    return [' '.join(line.strip().split()[1:]) for line in lines if line.strip()]

# === STEP 2: Modifica ultimo valore dei VERTEX_SE3:QUAT ===
def fix_last_vertex_value(lines):
    return [' '.join(parts[:-1] + ['1']) if line.startswith('VERTEX_SE3:QUAT') else line.strip()
            for line in lines if (parts := line.strip().split())]

# === STEP 3: Aggiunge matrice informativa agli EDGE_SE3:QUAT ===
def add_information_matrix(lines):
    info = '100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 10000 0 0 10000 0 10000'
    return [line.strip() + ' ' + info if line.startswith('EDGE_SE3:QUAT') else line.strip() for line in lines]

# === STEP 4: Rinumera vertici con gap ===
def renumber_with_robot_gaps(lines, id_jump=10000, start_id=1):
    id_mapping = {}
    result = []
    current_id = start_id
    previous_original_id = None
    robot_offset = 0

    vertex_lines = [line.strip() for line in lines if line.startswith("VERTEX_SE3:QUAT")]
    edge_lines = [line.strip() for line in lines if line.startswith("EDGE_SE3:QUAT")]
    other_lines = [line.strip() for line in lines if not line.startswith("VERTEX_SE3:QUAT") and not line.startswith("EDGE_SE3:QUAT")]

    for line in vertex_lines:
        parts = line.split()
        old_id = int(parts[1])
        if previous_original_id is not None and old_id < previous_original_id:
            robot_offset += id_jump
            current_id = start_id + robot_offset
        previous_original_id = old_id
        id_mapping[old_id] = current_id
        parts[1] = str(current_id)
        result.append(" ".join(parts))
        current_id += 1

    for line in edge_lines:
        parts = line.split()
        old_from, old_to = int(parts[1]), int(parts[2])
        if old_from in id_mapping and old_to in id_mapping:
            parts[1] = str(id_mapping[old_from])
            parts[2] = str(id_mapping[old_to])
            result.append(" ".join(parts))
        else:
            print(f"⚠️ Ignoring EDGE with unknown vertex ID: {line}")

    result.extend(other_lines)
    return result

# === STEP 5: Corregge il campo qw negli EDGE_SE3:QUAT ===
def fix_edge_quaternion(lines):
    result = []
    for line in lines:
        if line.startswith('EDGE_SE3:QUAT'):
            parts = line.strip().split()
            if len(parts) >= 11:
                parts[9] = '1'
                line = ' '.join(parts)
        result.append(line.strip())
    return result

# === Drift: Parser e Utilità ===
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
        if vertex_ids[i] - vertex_ids[i - 1] > threshold:
            groups.append(current_group)
            current_group = [vertex_ids[i]]
        else:
            current_group.append(vertex_ids[i])
    groups.append(current_group)
    return groups

# === MAIN ===
def main(csv_input_file, temp_txt_file, output_file, robot_index, threshold, mean, std):
    convert_csv_to_txt(csv_input_file, temp_txt_file)

    with open(temp_txt_file, 'r') as f:
        lines = f.readlines()

    # Processamento base
    lines = remove_first_column(lines)
    lines = fix_last_vertex_value(lines)
    lines = add_information_matrix(lines)
    lines = renumber_with_robot_gaps(lines)
    lines = fix_edge_quaternion(lines)

    # Salva temporaneo per drift
    temp_processed_file = 'processed_posegraph.g2o'
    with open(temp_processed_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    # Drift
    vertices, edge_lines, other_lines = {}, [], []
    with open(temp_processed_file, 'r') as f:
        for line in f:
            if line.startswith("VERTEX_SE3:QUAT"):
                vid, pos, quat = parse_vertex(line)
                vertices[vid] = {'pos': pos, 'quat': quat}
            elif line.startswith("EDGE_SE3:QUAT"):
                edge_lines.append(line)
            else:
                other_lines.append(line)

    all_vertex_ids = sorted(vertices.keys())
    groups = split_robot_vertex_ids(all_vertex_ids, threshold=threshold)

    if robot_index >= len(groups):
        print(f"❌ ERRORE: richiesto robot {robot_index}, ma ce ne sono solo {len(groups)}")
        return

    selected_ids = set(groups[robot_index])
    print(f"✅ Robot {robot_index}: {len(selected_ids)} vertici ({min(selected_ids)} - {max(selected_ids)})")

    np.random.seed(42)
    for vid in selected_ids:
        drift_xy = np.random.normal(loc=mean, scale=std, size=2)
        vertices[vid]['pos'][0] += drift_xy[0]
        vertices[vid]['pos'][1] += drift_xy[1]

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
            
    gnc_exec = "/home/iacopo/optimizer/gnc_individual_error"  # Cambia se è in un'altra directory
    try:
        subprocess.run([gnc_exec, output_file], check=True)
        print("✅ gnc_individual_error eseguito con successo.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione di gnc_individual_error: {e}")
    except FileNotFoundError:
        print("❌ gnc_individual_error non trovato. Assicurati che sia nel PATH o specifica il path completo.")
        

    print(f"✅ File finale generato: {output_file}")

# === ESEMPIO USO ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/iacopo/toychain-argos/SwarmSLAM/logs/slam_data_all.csv", help="File CSV da convertire")
    parser.add_argument("--txt", default="slam_data_all.txt", help="File intermedio .txt")
    parser.add_argument("--output", default="argos_ordered_traj_modified.g2o", help="File di output finale .g2o")
    parser.add_argument("--robot_index", type=int, required=True, help="Indice del robot a cui applicare drift")
    parser.add_argument("--threshold", type=int, default=1000, help="Soglia salto ID tra robot")
    parser.add_argument("--mean", type=float, default=1, help="Media del drift")
    parser.add_argument("--std", type=float, default=0.5, help="Deviazione standard del drift")
    args = parser.parse_args()

    main(args.csv, args.txt, args.output, args.robot_index, args.threshold, args.mean, args.std)

