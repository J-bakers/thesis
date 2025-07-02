#version with proportional error

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import csv
import subprocess 
import re
import csv
import time
import os
import shutil
import signal
import atexit
import matplotlib.pyplot as plt
import sys


import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools import file_interface
from evo.core.sync import associate_trajectories
from evo.core.trajectory import PoseTrajectory3D


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
    
    
def g2o_to_tum(g2o_file, tum_file):
    with open(g2o_file, 'r') as fi:
        content = fi.readlines()
        with open(tum_file, 'w') as fo:
            for line in content:
                if line[0] == 'V':
                    elems = line.split()
                    tum_line = ""
                    for elem in elems[1:]:
                        tum_line += elem + " "
                    tum_line_to_write = tum_line[0:-1] + '\n'
                    fo.write(tum_line_to_write)

def build_vertex_robot_map(g2o_path, threshold=1000):
    vertex_robot_map = {}
    current_robot_id = 0
    previous_id = None

    with open(g2o_path, "r") as f:
        for line in f:
            if line.startswith("VERTEX_SE3:QUAT"):
                tokens = line.strip().split()
                current_id = int(tokens[1])

                if previous_id is not None and abs(current_id - previous_id) >= threshold:
                    current_robot_id += 1
                
                vertex_robot_map[current_id] = current_robot_id
                previous_id = current_id

    return vertex_robot_map

def make_info_matrix(val):
    # Restituisce la matrice informativa come lista di stringhe, con valore 'val' sulla diagonale
    return [
        str(val), "0", "0", "0", "0", "0",
        str(val), "0", "0", "0", "0",
        str(val), "0", "0", "0",
        str(val), "0", "0",
        str(val), "0",
        str(val)
    ]


# === PLOT RESULTS === 
def plot_results():
    print("🛑 Experiment finished. Generating final plot...")
    filename = "/home/iacopo/optimizer_argos/ape_metrics.csv"  # cambia col tuo file

    baseline_means = []
    penalized_means = []
    baseline_rmses = []
    penalized_rmses = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    current_section = None

    for i, line in enumerate(lines):
        line = line.strip()
        if line == "BASELINE":
            current_section = "BASELINE"
        elif line == "PENALIZED":
            current_section = "PENALIZED"
        elif line.startswith("mean,") and current_section is not None:
            _, value = line.split(",")
            value = float(value)
            if current_section == "BASELINE":
                baseline_means.append(value)
            elif current_section == "PENALIZED":
                penalized_means.append(value)
        elif line.startswith("rmse,") and current_section is not None:
            _, value = line.split(",")
            value = float(value)
            if current_section == "BASELINE":
                baseline_rmses.append(value)
            elif current_section == "PENALIZED":
                penalized_rmses.append(value)

    iterations = range(1, len(baseline_means) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, baseline_means, marker='o', label='BASELINE')
    plt.plot(iterations, penalized_means, marker='x', label='PENALIZED')
    plt.xlabel('Iterazione')
    plt.ylabel('APE mean')
    plt.title('APE mean per BASELINE e PENALIZED')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, baseline_rmses, marker='o', label='BASELINE')
    plt.plot(iterations, penalized_rmses, marker='x', label='PENALIZED')
    plt.xlabel('Iterazione')
    plt.ylabel('APE RMSE')
    plt.title('APE RMSE per BASELINE e PENALIZED')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_info_matrix_history():
    import matplotlib.pyplot as plt
    import csv

    filename = "info_matrix_history.csv"
    if not os.path.exists(filename):
        print("Nessun file info_matrix_history.csv trovato.")
        return

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) < 2:
        print("File info_matrix_history.csv vuoto o senza dati.")
        return

    header = rows[0][1:]  # salta timestamp
    data = rows[1:]

    iterations = list(range(1, len(data) + 1))
    for i, robot in enumerate(header):
        values = [float(row[i+1]) for row in data]
        plt.plot(iterations, values, marker='o', label=robot)

    plt.xlabel('Iterazione')
    plt.ylabel('Valore diagonale info matrix')
    plt.title('Evoluzione info matrix per robot')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    

# === MAIN ===
def main(csv_input_file, temp_txt_file, modified_traj_file, robot_index, threshold, mean, std):
   

    # Step 0: Carica gli errori precedenti dei robot
    prev_robot_errors = {}
    if os.path.exists("last_robot_errors.csv"):
        with open("last_robot_errors.csv", newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:
                    prev_robot_errors[int(row[0])] = float(row[1])
    
    csv_snapshot = "slam_data_all_snapshot.csv"
    shutil.copy(csv_input_file, csv_snapshot)

    convert_csv_to_txt(csv_snapshot, temp_txt_file)

    with open(temp_txt_file, 'r') as f:
        lines = f.readlines()

    # Step 1: Preprocessing
    lines = remove_first_column(lines)
    lines = fix_last_vertex_value(lines)
    lines = add_information_matrix(lines)
    lines = renumber_with_robot_gaps(lines)
    lines = fix_edge_quaternion(lines)

    # Save temp processed file
    temp_processed_file = 'processed_posegraph.g2o'
    with open(temp_processed_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    # Step 2: Apply drift to selected robot
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

    with open(modified_traj_file, 'w') as f:
        for line in other_lines:
            f.write(line)
        for vid in sorted(vertices.keys()):
            p = vertices[vid]['pos']
            q = vertices[vid]['quat']
            f.write(f"VERTEX_SE3:QUAT {vid} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
        for line in new_edge_lines:
            f.write(line)

    modified_traj_file = "argos_ordered_traj_modified.g2o"

    # Step 1: Calcolo degli errori individuali
    gnc_exec = "/home/iacopo/optimizer/gnc_individual_error"
    try:
        result = subprocess.run([gnc_exec, modified_traj_file], check=True, text=True, capture_output=True)
        output = result.stdout
        print("✅ gnc_individual_error eseguito con successo.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione di gnc_individual_error: {e}")
        output = e.stdout if e.stdout else ""
    except FileNotFoundError:
        print("❌ gnc_individual_error non trovato.")
        output = ""
   
   # Step 2: Parsing degli errori
    robot_errors = {}
    for line in output.splitlines():
        match = re.match(r"Robot (\d+) individual error: ([\d\.eE+-]+)", line)
        if match:
            robot_id = int(match.group(1))
            error = float(match.group(2))
            robot_errors[robot_id] = error
    print("📊 Errori individuali:", robot_errors)


    if prev_robot_errors:
        # Penalizza in base agli errori della iterazione precedente
        base_info_value = 1000.0
        min_info_value = 1e-3
        worst_factor = 0.001

        total_error = sum(prev_robot_errors.values()) if prev_robot_errors else 1.0
        worst_robot = max(prev_robot_errors, key=prev_robot_errors.get) if prev_robot_errors else None

        penalties = {}
        for robot_id, error in prev_robot_errors.items():
            norm_error = error / total_error if total_error > 0 else 0
            info_value = max(base_info_value * (1 - norm_error), min_info_value)
            if robot_id == worst_robot:
                info_value *= worst_factor
            penalties[robot_id] = info_value

        print("Valori info matrix applicati ai robot:", penalties)
       
        # Salva la storia della info matrix
        with open("info_matrix_history.csv", mode="a", newline='') as file:
            writer = csv.writer(file)
            row = [time.time()] + [penalties[robot_id] for robot_id in sorted(penalties.keys())]
            writer.writerow(row)

        # Step 4: Modifica matrice informativa per tutti i robot
        vertex_robot_map = build_vertex_robot_map(modified_traj_file)
        info_modified_file = "argos_ordered_traj_modified_cov_modified.g2o"
        with open(modified_traj_file, "r") as infile, open(info_modified_file, "w") as outfile:
            for line in infile:
                if line.startswith("EDGE_SE3:QUAT"):
                    tokens = line.strip().split()
                    id1 = int(tokens[1])
                    id2 = int(tokens[2])
                    robot1 = vertex_robot_map.get(id1)
                    robot2 = vertex_robot_map.get(id2)
                    if robot1 == robot2 and robot1 in penalties:
                        info_value = penalties[robot1]
                        new_info_matrix = make_info_matrix(info_value)
                        new_line = " ".join(tokens[:10] + new_info_matrix) + "\n"
                        outfile.write(new_line)
                    else:
                        outfile.write(line)
                else:
                    outfile.write(line)
        print("✅ Info matrix aggiornata per tutti i robot in modo proporzionale.")
        

        # Step 8: Ottimizzazione finale
        gnc_opt_exec = "/home/iacopo/optimizer/gnc_optimizer"
        optimized_file = "output_optimized.g2o"
        try:
            result = subprocess.run([gnc_opt_exec, info_modified_file, optimized_file], check=True, text=True, capture_output=True)
            print("✅ Ottimizzazione GNC completata.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore nell'ottimizzazione GNC: {e}")
            print(e.stderr)
        except FileNotFoundError:
            print("❌ Eseguibile gnc_optimizer non trovato.")

        # Step 9: Eval con evo
        g2o_to_tum('output_optimized.g2o', 'output_optimized.txt')
        g2o_to_tum('processed_posegraph.g2o', 'processed_posegraph.txt')

        gt = file_interface.read_tum_trajectory_file("processed_posegraph.txt")
        est = file_interface.read_tum_trajectory_file("output_optimized.txt")
        gt, est = associate_trajectories(gt, est)

        ape_result = main_ape.ape(gt, est, pose_relation=PoseRelation.translation_part, align=False)

        print("APE RMSE:", ape_result.stats["rmse"])
        print("APE mean:", ape_result.stats["mean"])
        print("APE max:", ape_result.stats["max"])
        print("APE median:", ape_result.stats["median"])

        # Calcola anche APE baseline tra ground truth e output_baseline_optimized.g2o
        baseline_input = "argos_ordered_traj_modified.g2o"
        baseline_optimized = "output_baseline_optimized.g2o"
        try:
            result = subprocess.run([gnc_opt_exec, baseline_input, baseline_optimized], check=True, text=True, capture_output=True)
            print("✅ Ottimizzazione baseline GNC completata.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore nell'ottimizzazione baseline GNC: {e}")
            print(e.stderr)
        except FileNotFoundError:
            print("❌ Eseguibile gnc_optimizer non trovato (baseline).")

        g2o_to_tum('output_baseline_optimized.g2o', 'output_baseline_optimized.txt')
        est_baseline = file_interface.read_tum_trajectory_file("output_baseline_optimized.txt")
        gt_baseline = file_interface.read_tum_trajectory_file("processed_posegraph.txt")
        gt_baseline, est_baseline = associate_trajectories(gt_baseline, est_baseline)
        ape_baseline = main_ape.ape(gt_baseline, est_baseline, pose_relation=PoseRelation.translation_part, align=False)

        print("BASELINE APE RMSE:", ape_baseline.stats["rmse"])
        print("BASELINE APE mean:", ape_baseline.stats["mean"])
        print("BASELINE APE max:", ape_baseline.stats["max"])
        print("BASELINE APE median:", ape_baseline.stats["median"])

    else:
        print("Prima iterazione: nessuna penalizzazione applicata.")
         # ... (salta la penalizzazione, esegui solo baseline)
          # --- BASELINE OPTIMIZATION --- optimization of the g2o file with fixed info matrix values
        gnc_opt_exec = "/home/iacopo/optimizer/gnc_optimizer"
        baseline_input = "argos_ordered_traj_modified.g2o"
        baseline_optimized = "output_baseline_optimized.g2o"
        try:
            result = subprocess.run([gnc_opt_exec, baseline_input, baseline_optimized], check=True, text=True, capture_output=True)
            print("✅ Ottimizzazione baseline GNC completata.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Errore nell'ottimizzazione baseline GNC: {e}")
            print(e.stderr)
        except FileNotFoundError:
            print("❌ Eseguibile gnc_optimizer non trovato (baseline).")

        # Converti baseline in TUM
        g2o_to_tum(baseline_optimized, 'output_baseline_optimized.txt')
        g2o_to_tum('processed_posegraph.g2o', 'processed_posegraph.txt')

        # Calcola APE baseline
        est_baseline = file_interface.read_tum_trajectory_file("output_baseline_optimized.txt")
        gt = file_interface.read_tum_trajectory_file("processed_posegraph.txt")
        gt, est_baseline = associate_trajectories(gt, est_baseline)
        ape_baseline = main_ape.ape(gt, est_baseline, pose_relation=PoseRelation.translation_part, align=False)
        ape_result = main_ape.ape(gt, est_baseline, pose_relation=PoseRelation.translation_part, align=False)


        print("BASELINE APE RMSE:", ape_baseline.stats["rmse"])
        print("BASELINE APE mean:", ape_baseline.stats["mean"])
        print("BASELINE APE max:", ape_baseline.stats["max"])
        print("BASELINE APE median:", ape_baseline.stats["median"])




   #save a csv file with the results
    with open("ape_metrics.csv", mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["--- Nuova Iterazione ---"])
        writer.writerow(["BASELINE"])
        for key, value in ape_baseline.stats.items():
            writer.writerow([key, value])
        writer.writerow(["PENALIZED"])
        for key, value in ape_result.stats.items():
            writer.writerow([key, value])
        writer.writerow([])  # Riga vuota per separazione




    # Salva errori individuali per la prossima iterazione
    with open("last_robot_errors.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        for robot_id, error in robot_errors.items():
            writer.writerow([robot_id, error])


   






# === ESEMPIO USO ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/home/iacopo/toychain-argos/SwarmSLAM/logs/slam_data_all.csv", help="File CSV da convertire")
    parser.add_argument("--txt", default="slam_data_all.txt", help="File intermedio .txt")
    parser.add_argument("--output", default="argos_ordered_traj_modified.g2o", help="File di output finale .g2o")
    parser.add_argument("--robot_index", type=int, required=True, help="Indice del robot a cui applicare drift")
    parser.add_argument("--threshold", type=int, default=1000, help="Soglia salto ID tra robot")
    parser.add_argument("--mean", type=float, default=1.5, help="Media del drift")
    parser.add_argument("--std", type=float, default=1, help="Deviazione standard del drift")
    parser.add_argument("--interval", type=int, default=30, help="Ogni quanti secondi eseguire il processo")
    args = parser.parse_args()

    print("⏳ Attendo 60 secondi prima di iniziare il ciclo...")
    time.sleep(60)

    print(f"🔁 Avvio loop di elaborazione ogni {args.interval} secondi. CTRL+C per interrompere.")

    try:
        while True:
            print(f"\n⏱️  Esecuzione in corso (robot_index = {args.robot_index})...")
            main(args.csv, args.txt, args.output, args.robot_index, args.threshold, args.mean, args.std)
            print(f"✅ Attesa di {args.interval} secondi prima della prossima esecuzione...")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 Interrotto manualmente. Uscita dal ciclo.")



signal.signal(signal.SIGINT, lambda *_: (plot_results(), sys.exit(0)))
atexit.register(plot_results)         #grafico ape protected vs baseline
atexit.register(plot_info_matrix_history)  #grafico info matrix history





