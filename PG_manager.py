#script that gathers all the functions for processing argos data

import sys

# === Configura i nomi dei file input/output ===
input_file = 'slam_data_all.txt'
final_output_file = 'argos_trial_pieces.g2o'

# === Step 1: Rimuove la prima colonna da ogni riga ===
def remove_first_column(lines):
    result = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 1:
            result.append(' '.join(parts[1:]))
        else:
            result.append('')
    return result

# === Step 2: Modifica l'ultimo valore dei VERTEX_SE3:QUAT a "1" ===
def fix_last_vertex_value(lines):
    result = []
    for line in lines:
        if line.startswith('VERTEX_SE3:QUAT'):
            parts = line.strip().split()
            parts[-1] = '1'
            result.append(' '.join(parts))
        else:
            result.append(line.strip())
    return result

# === Step 3: Aggiunge la matrice informativa agli EDGE_SE3:QUAT ===
def add_information_matrix(lines):
    info = '100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 10000 0 0 10000 0 10000'
    result = []
    for line in lines:
        if line.startswith('EDGE_SE3:QUAT'):
            line = line.strip() + ' ' + info
        result.append(line.strip())
    return result

# === Step 4: Rinumera i vertici con salti tra robot (gap di 10000) ===
def renumber_with_robot_gaps(lines, id_jump=10000, start_id=1):
    id_mapping = {}
    result = []
    current_id = start_id
    previous_original_id = None
    robot_offset = 0

    vertex_lines = []
    edge_lines = []
    other_lines = []

    for line in lines:
        if line.startswith("VERTEX_SE3:QUAT"):
            vertex_lines.append(line.strip())
        elif line.startswith("EDGE_SE3:QUAT"):
            edge_lines.append(line.strip())
        else:
            other_lines.append(line.strip())

    for line in vertex_lines:
        parts = line.strip().split()
        old_id = int(parts[1])
        if previous_original_id is not None and old_id < previous_original_id:
            robot_offset += id_jump
            current_id = start_id + robot_offset
        previous_original_id = old_id
        new_id = current_id
        id_mapping[old_id] = new_id
        parts[1] = str(new_id)
        result.append(" ".join(parts))
        current_id += 1

    for line in edge_lines:
        parts = line.strip().split()
        old_from = int(parts[1])
        old_to = int(parts[2])
        if old_from not in id_mapping or old_to not in id_mapping:
            print(f"⚠️ Ignoring EDGE with unknown vertex ID: {line}")
            continue
        parts[1] = str(id_mapping[old_from])
        parts[2] = str(id_mapping[old_to])
        result.append(" ".join(parts))

    result.extend(other_lines)
    return result

# === Step 5: Corregge il campo qw negli EDGE_SE3:QUAT ===
def fix_edge_quaternion(lines):
    result = []
    for line in lines:
        if line.startswith('EDGE_SE3:QUAT'):
            parts = line.strip().split()
            if len(parts) >= 11:
                parts[9] = '1'  # qw (decimo campo)
                line = ' '.join(parts)
        result.append(line.strip())
    return result

# === MAIN: esegui tutti gli step ===
def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    lines = remove_first_column(lines)
    lines = fix_last_vertex_value(lines)
    lines = add_information_matrix(lines)
    lines = renumber_with_robot_gaps(lines)
    lines = fix_edge_quaternion(lines)

    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"✅ File processato con successo e salvato in: {output_file}")

# === Avvia lo script ===
if __name__ == '__main__':
    process_file(input_file, final_output_file)

