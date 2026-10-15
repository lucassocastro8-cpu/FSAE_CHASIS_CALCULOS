import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =====================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO UX/UI (ROJO, NEGRO Y BLANCO)
# =====================================================================
st.set_page_config(
    page_title="FSAE 2026 - Analizador Estructural de Chasis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0B0B0E;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #121216;
        border-right: 2px solid #E10600;
    }
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
        letter-spacing: 0.5px;
    }
    h1 {
        border-bottom: 3px solid #E10600;
        padding-bottom: 10px;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #16161C, #0F0F13);
        border: 1px solid #2A2A35;
        border-left: 5px solid #E10600;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800;
    }
    div[data-testid="stMetricLabel"] {
        color: #CCCCCC !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
    }
    button[data-baseweb="tab"] {
        background-color: #16161C !important;
        color: #CCCCCC !important;
        border-radius: 6px 6px 0 0 !important;
        margin-right: 4px !important;
        border: 1px solid #2A2A35 !important;
        padding-left: 16px !important;
        padding-right: 16px !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #E10600 !important;
        color: #FFFFFF !important;
        border-color: #FF2800 !important;
    }
    .status-pass {
        background-color: #141418;
        border: 2px solid #FFFFFF;
        border-left: 8px solid #FFFFFF;
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 6px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .status-fail {
        background-color: #2A0808;
        border: 2px solid #E10600;
        border-left: 8px solid #E10600;
        color: #FFFFFF;
        padding: 12px 18px;
        border-radius: 6px;
        font-weight: bold;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def mostrar_estado(aprobado, texto):
    clase = "status-pass" if aprobado else "status-fail"
    icono = "✅" if aprobado else "❌"
    html = f'<div class="{clase}">{icono} {texto}</div>'
    st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# 2. LECTOR Y AUTO-ORIENTADOR DE CHASIS CAD (.IGS / .DXF)
# =====================================================================
def clean_and_weld_chassis_mesh(
    raw_lines,
    scale_factor=1.0,
    rot_roll_90=False,
    flip_front_rear=False,
    invert_z=False,
    tol_mm=5.0
):
    if not raw_lines:
        return np.zeros((0, 3)), []

    scaled_lines = []
    lengths = []
    for p1, p2 in raw_lines:
        a = np.array(p1, dtype=float) * scale_factor
        b = np.array(p2, dtype=float) * scale_factor
        L = np.linalg.norm(b - a)
        if L > tol_mm:
            scaled_lines.append((a, b))
            lengths.append(L)

    if not scaled_lines:
        return np.zeros((0, 3)), []

    med_L = float(np.median(lengths))
    valid_lines = [seg for seg, L in zip(scaled_lines, lengths) if L <= max(med_L * 6.0, 3500.0)]
    if len(valid_lines) < 4:
        valid_lines = scaled_lines

    nodes = []

    def get_or_add_node(pt):
        for idx, n in enumerate(nodes):
            if np.linalg.norm(pt - n) <= tol_mm:
                return idx
        nodes.append(pt.copy())
        return len(nodes) - 1

    initial_edges = []
    for a, b in valid_lines:
        u = get_or_add_node(a)
        v = get_or_add_node(b)
        if u != v:
            edge = (min(u, v), max(u, v))
            if edge not in initial_edges:
                initial_edges.append(edge)

    nodes_arr = np.array(nodes, dtype=float)

    # Partir tubos en uniones tipo "T"
    split_edges = []
    for u, v in initial_edges:
        pA = nodes_arr[u]
        pB = nodes_arr[v]
        vec = pB - pA
        L_sq = np.dot(vec, vec)
        if L_sq < 1e-6:
            continue

        intermediate_nodes = []
        for w in range(len(nodes_arr)):
            if w == u or w == v:
                continue
            pW = nodes_arr[w]
            t_proj = np.dot(pW - pA, vec) / L_sq
            if 0.02 < t_proj < 0.98:
                proj_pt = pA + t_proj * vec
                dist = np.linalg.norm(pW - proj_pt)
                if dist <= tol_mm * 1.5:
                    intermediate_nodes.append((t_proj, w))

        if not intermediate_nodes:
            split_edges.append((min(u, v), max(u, v)))
        else:
            intermediate_nodes.sort(key=lambda x: x[0])
            chain = [u] + [w for _, w in intermediate_nodes] + [v]
            for k in range(len(chain) - 1):
                n1, n2 = chain[k], chain[k + 1]
                if n1 != n2:
                    split_edges.append((min(n1, n2), max(n1, n2)))

    unique_edges = []
    for e in split_edges:
        if e not in unique_edges:
            unique_edges.append(e)

    # Conservar el componente conexo principal
    if len(nodes_arr) > 0 and len(unique_edges) > 0:
        adj = {i: [] for i in range(len(nodes_arr))}
        for u, v in unique_edges:
            adj[u].append(v)
            adj[v].append(u)

        visited = set()
        components = []
        for i in range(len(nodes_arr)):
            if i not in visited and len(adj[i]) > 0:
                comp = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nxt in adj[curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
                components.append(comp)

        if components:
            largest_comp = set(max(components, key=len))
            old_to_new = {}
            new_nodes = []
            for old_idx in sorted(largest_comp):
                old_to_new[old_idx] = len(new_nodes)
                new_nodes.append(nodes_arr[old_idx])

            new_edges = []
            for u, v in unique_edges:
                if u in largest_comp and v in largest_comp:
                    nu, nv = old_to_new[u], old_to_new[v]
                    new_edges.append((min(nu, nv), max(nu, nv)))

            nodes_arr = np.array(new_nodes, dtype=float)
            unique_edges = new_edges

    # Auto-orientación inteligente del chasis FSAE
    if len(nodes_arr) >= 4:
        spans = np.ptp(nodes_arr, axis=0)
        idx_long = int(np.argmax(spans))
        rem = [i for i in range(3) if i != idx_long]

        if spans[rem[0]] >= spans[rem[1]]:
            idx_vert = rem[0]
            idx_lat = rem[1]
        else:
            idx_vert = rem[1]
            idx_lat = rem[0]

        nodes_arr = nodes_arr[:, [idx_long, idx_lat, idx_vert]]

        if rot_roll_90:
            nodes_arr = nodes_arr[:, [0, 2, 1]]

        # Asegurar que el Main Hoop (parte superior) apunte hacia +Z (arriba)
        z_vals = nodes_arr[:, 2]
        z_mid = 0.5 * (z_vals.min() + z_vals.max())
        n_below = np.sum(z_vals < z_mid)
        n_above = np.sum(z_vals >= z_mid)
        if n_above > n_below:
            nodes_arr[:, 2] = -nodes_arr[:, 2]

        if invert_z:
            nodes_arr[:, 2] = -nodes_arr[:, 2]

        # Asegurar que el Frente del auto quede en X = 0 y el Main Hoop hacia atrás
        nodes_arr[:, 2] -= nodes_arr[:, 2].min()
        top_z_nodes = np.argsort(nodes_arr[:, 2])[-3:]
        x_of_top = np.mean(nodes_arr[top_z_nodes, 0])
        x_mid = 0.5 * (nodes_arr[:, 0].min() + nodes_arr[:, 0].max())
        if x_of_top < x_mid:
            nodes_arr[:, 0] = -nodes_arr[:, 0]

        if flip_front_rear:
            nodes_arr[:, 0] = -nodes_arr[:, 0]

        nodes_arr[:, 0] -= nodes_arr[:, 0].min()
        nodes_arr[:, 1] -= 0.5 * (nodes_arr[:, 1].max() + nodes_arr[:, 1].min())
        nodes_arr[:, 2] -= nodes_arr[:, 2].min()

    return nodes_arr, unique_edges


def parse_dxf(content_str, scale_factor=1.0, rot_roll_90=False, flip_front_rear=False, invert_z=False, tol_mm=5.0):
    lines_list = [line.strip() for line in content_str.splitlines()]
    raw_segments = []
    i = 0
    n = len(lines_list)

    while i < n - 1:
        code = lines_list[i]
        val = lines_list[i + 1]
        if code == "0" and val == "LINE":
            i += 2
            x1 = y1 = z1 = x2 = y2 = z2 = 0.0
            while i < n - 1 and lines_list[i] != "0":
                c = lines_list[i]
                v = lines_list[i + 1]
                try:
                    if c == "10": x1 = float(v)
                    elif c == "20": y1 = float(v)
                    elif c == "30": z1 = float(v)
                    elif c == "11": x2 = float(v)
                    elif c == "21": y2 = float(v)
                    elif c == "31": z2 = float(v)
                except ValueError:
                    pass
                i += 2
            raw_segments.append(([x1, y1, z1], [x2, y2, z2]))
        else:
            i += 2

    return clean_and_weld_chassis_mesh(raw_segments, scale_factor, rot_roll_90, flip_front_rear, invert_z, tol_mm)


def parse_iges(content_str, scale_factor=1.0, rot_roll_90=False, flip_front_rear=False, invert_z=False, tol_mm=5.0):
    p_lines = []
    for line in content_str.splitlines():
        if len(line) >= 73 and line[72] == "P":
            p_lines.append(line[:64].strip())

    full_p_text = "".join(p_lines)
    records = [r.strip() for r in full_p_text.split(";") if r.strip()]

    raw_segments = []
    for rec in records:
        clean_rec = rec.replace("D", "E").replace("d", "e")
        parts = [p.strip() for p in clean_rec.split(",") if p.strip()]
        if not parts:
            continue
        try:
            ent_type = int(float(parts[0]))
            if ent_type == 110 and len(parts) >= 7:
                x1, y1, z1 = float(parts[1]), float(parts[2]), float(parts[3])
                x2, y2, z2 = float(parts[4]), float(parts[5]), float(parts[6])
                raw_segments.append(([x1, y1, z1], [x2, y2, z2]))
            elif ent_type == 126 and len(parts) >= 14:
                K = int(float(parts[1]))
                M = int(float(parts[2]))
                N = 1 + K - M
                A = N + 2 * M
                start_cp = 7 + (A + 1) + (K + 1)
                end_cp = start_cp + 3 * K
                if len(parts) > end_cp + 2:
                    x1 = float(parts[start_cp])
                    y1 = float(parts[start_cp + 1])
                    z1 = float(parts[start_cp + 2])
                    x2 = float(parts[end_cp])
                    y2 = float(parts[end_cp + 1])
                    z2 = float(parts[end_cp + 2])
                    raw_segments.append(([x1, y1, z1], [x2, y2, z2]))
            elif ent_type == 106 and len(parts) >= 9:
                ip = int(float(parts[1]))
                n_pts = int(float(parts[2]))
                if ip == 2 and len(parts) >= 3 + 3 * n_pts:
                    pts = []
                    for k in range(n_pts):
                        px = float(parts[3 + 3 * k])
                        py = float(parts[4 + 3 * k])
                        pz = float(parts[5 + 3 * k])
                        pts.append([px, py, pz])
                    for k in range(len(pts) - 1):
                        raw_segments.append((pts[k], pts[k + 1]))
        except (ValueError, IndexError):
            continue

    return clean_and_weld_chassis_mesh(raw_segments, scale_factor, rot_roll_90, flip_front_rear, invert_z, tol_mm)


def generate_reference_fsae_chassis():
    nodes = np.array([
        [0, -190, 20],    [0, 190, 20],     [0, 190, 360],    [0, -190, 360],
        [520, -220, 0],   [520, 220, 0],    [520, 238, 420],  [520, -238, 420],
        [980, -260, 0],   [980, 260, 0],    [980, 280, 560],  [980, -280, 560],
        [1350, -320, 0],  [1350, 320, 0],   [1350, 320, 330], [1350, -320, 330],
        [1350, -520, 160],[1350, 520, 160],
        [1720, -290, 0],  [1720, 290, 0],   [1720, 300, 330], [1720, -300, 330],
        [1720, 250, 620], [1720, -250, 620],
        [1800, 110, 1109],[1800, -110, 1109],
        [2180, -230, 40], [2180, 230, 40],  [2180, 230, 435], [2180, -230, 435],
        [2480, -180, 90], [2480, 180, 90],  [2480, 180, 360], [2480, -180, 360],
    ], dtype=float)

    elements = [
        (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),
        (0, 4), (1, 5), (2, 6), (3, 7), (0, 7), (1, 6), (0, 5), (1, 4),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (4, 8), (5, 9), (6, 10), (7, 11), (4, 9), (5, 8),
        (2, 10), (3, 11), (4, 11), (5, 10), (6, 9), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 8),
        (8, 12), (12, 18), (9, 13), (13, 19), (12, 13), (8, 13), (12, 19),
        (8, 15), (15, 21), (9, 14), (14, 20),
        (8, 12), (15, 18), (9, 13), (14, 19), (12, 15), (13, 14),
        (10, 22), (11, 23),
        (8, 16), (16, 18), (12, 16), (15, 16),
        (9, 17), (17, 19), (13, 17), (14, 17),
        (18, 19), (18, 21), (19, 20), (21, 20),
        (20, 22), (21, 23), (22, 23),
        (22, 24), (23, 25), (24, 25),
        (18, 20), (19, 21),
        (24, 28), (25, 29),
        (18, 26), (19, 27), (20, 28), (21, 29),
        (18, 29), (19, 28), (22, 28), (23, 29), (18, 27),
        (26, 27), (27, 28), (28, 29), (29, 26), (26, 28),
        (26, 30), (27, 31), (28, 32), (29, 33),
        (26, 33), (27, 32), (26, 31),
        (30, 31), (31, 32), (32, 33), (33, 30), (30, 32)
    ]
    unique_elems = []
    for u, v in elements:
        edge = (min(u, v), max(u, v))
        if edge not in unique_elems:
            unique_elems.append(edge)
    return nodes, unique_elems


# =====================================================================
# 3. MOTOR DE ELEMENTOS FINITOS 3D EN MILÍMETROS (N, mm, MPa)
# =====================================================================
def beam_3d_transformation_matrix(p1, p2):
    dx, dy, dz = p2 - p1
    L = max(math.sqrt(dx*dx + dy*dy + dz*dz), 1.0)
    cx, cy, cz = dx / L, dy / L, dz / L

    if abs(cx) < 1e-6 and abs(cy) < 1e-6:
        r = np.array([[0.0, 0.0, cz], [0.0, 1.0, 0.0], [-cz, 0.0, 0.0]])
    else:
        d = math.sqrt(cx*cx + cy*cy)
        r = np.array([
            [cx, cy, cz],
            [-cy / d, cx / d, 0.0],
            [-cx * cz / d, -cy * cz / d, d]
        ])

    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*i+3, 3*i:3*i+3] = r
    return T, L


def beam_3d_local_stiffness(E, G, A, Iy, Iz, J, L):
    k = np.zeros((12, 12))
    EA_L = E * A / L
    GJ_L = G * J / L
    EIy = E * Iy
    EIz = E * Iz

    k[0, 0] = k[6, 6] = EA_L
    k[0, 6] = k[6, 0] = -EA_L
    k[3, 3] = k[9, 9] = GJ_L
    k[3, 9] = k[9, 3] = -GJ_L
    k[1, 1] = k[7, 7] = 12 * EIz / L**3
    k[1, 7] = k[7, 1] = -12 * EIz / L**3
    k[1, 5] = k[5, 1] = k[1, 11] = k[11, 1] = 6 * EIz / L**2
    k[7, 5] = k[5, 7] = k[7, 11] = k[11, 7] = -6 * EIz / L**2
    k[5, 5] = k[11, 11] = 4 * EIz / L
    k[5, 11] = k[11, 5] = 2 * EIz / L
    k[2, 2] = k[8, 8] = 12 * EIy / L**3
    k[2, 8] = k[8, 2] = -12 * EIy / L**3
    k[2, 4] = k[4, 2] = k[2, 10] = k[10, 2] = -6 * EIy / L**2
    k[8, 4] = k[4, 8] = k[8, 10] = k[10, 8] = 6 * EIy / L**2
    k[4, 4] = k[10, 10] = 4 * EIy / L
    k[4, 10] = k[10, 4] = 2 * EIy / L

    return k


def solve_frame_3d(nodes, elements, prop_dict, fixed_dofs, nodal_loads):
    n_nodes = len(nodes)
    ndof = 6 * n_nodes
    K_global = np.zeros((ndof, ndof))
    F_global = np.zeros(ndof)

    E = prop_dict["E"]
    G = prop_dict["G"]
    D = prop_dict["OD"]
    t = prop_dict["t"]
    rho = prop_dict["rho"]

    d_in = max(D - 2.0 * t, 0.1)
    A = (math.pi / 4.0) * (D**2 - d_in**2)
    I = (math.pi / 64.0) * (D**4 - d_in**4)
    J = 2.0 * I
    c_outer = D / 2.0

    elem_cache = []
    total_length_mm = 0.0

    for idx, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        T, L_mm = beam_3d_transformation_matrix(p1, p2)
        total_length_mm += L_mm
        k_loc = beam_3d_local_stiffness(E, G, A, I, I, J, L_mm)
        k_glob = T.T @ k_loc @ T

        dofs = [6*n1+j for j in range(6)] + [6*n2+j for j in range(6)]
        for a in range(12):
            for b in range(12):
                K_global[dofs[a], dofs[b]] += k_glob[a, b]

        elem_cache.append((n1, n2, T, k_loc, L_mm, dofs))

    for node_id, force_vec in nodal_loads.items():
        for j in range(min(len(force_vec), 6)):
            F_global[6 * node_id + j] += force_vec[j]

    K_global += np.eye(ndof) * 1e-3

    free_dofs = np.setdiff1d(np.arange(ndof), np.unique(fixed_dofs))
    U = np.zeros(ndof)

    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    F_f = F_global[free_dofs]

    try:
        U[free_dofs] = np.linalg.solve(K_ff, F_f)
    except np.linalg.LinAlgError:
        U[free_dofs] = np.linalg.lstsq(K_ff, F_f, rcond=None)[0]

    u_xyz_mm = U.reshape((n_nodes, 6))[:, :3]
    disp_mag_mm = np.linalg.norm(u_xyz_mm, axis=1)

    von_mises_elem = np.zeros(len(elements))
    disp_elem_mm = np.zeros(len(elements))

    for idx, (n1, n2, T, k_loc, L_mm, dofs) in enumerate(elem_cache):
        u_e_glob = U[dofs]
        u_e_loc = T @ u_e_glob
        f_loc = k_loc @ u_e_loc

        N_axial = max(abs(f_loc[0]), abs(f_loc[6]))
        V_shear = max(math.hypot(f_loc[1], f_loc[2]), math.hypot(f_loc[7], f_loc[8]))
        T_torsion = max(abs(f_loc[3]), abs(f_loc[9]))
        M_bend = max(math.hypot(f_loc[4], f_loc[5]), math.hypot(f_loc[10], f_loc[11]))

        sigma_normal = (N_axial / A) + (M_bend * c_outer / I)
        tau_total = (T_torsion * c_outer / J) + (2.0 * V_shear / A)

        vm = math.sqrt(sigma_normal**2 + 3.0 * (tau_total**2))
        von_mises_elem[idx] = vm
        disp_elem_mm[idx] = 0.5 * (disp_mag_mm[n1] + disp_mag_mm[n2])

    chassis_mass_kg = total_length_mm * A * rho
    return {
        "u_xyz": u_xyz_mm,
        "disp_node": disp_mag_mm,
        "disp_elem": disp_elem_mm,
        "vm_elem": von_mises_elem,
        "mass_kg": chassis_mass_kg,
        "total_length_mm": total_length_mm,
        "A_mm2": A,
        "I_mm4": I
    }


# =====================================================================
# 4. GENERADOR DE ESCENARIOS DE PRUEBA (FSAE 2026 + REFERENCIA)
# =====================================================================
def identify_key_regions(nodes):
    x = nodes[:, 0]
    z = nodes[:, 2]
    x_min, x_max = x.min(), x.max()
    L_total = max(x_max - x_min, 1.0)
    z_min, z_max = z.min(), z.max()
    H_total = max(z_max - z_min, 1.0)

    front_bulkhead = np.where(x <= x_min + 0.08 * L_total)[0]
    if len(front_bulkhead) < 2:
        front_bulkhead = np.argsort(x)[:4]

    cond_fs = (x >= x_min + 0.14 * L_total) & (x <= x_min + 0.36 * L_total) & (z <= z_min + 0.45 * H_total)
    front_susp = np.where(cond_fs)[0]
    if len(front_susp) < 2:
        front_susp = np.argsort(np.abs(x - (x_min + 0.24 * L_total)))[:4]

    cond_ck = (x >= x_min + 0.38 * L_total) & (x <= x_min + 0.72 * L_total) & (z <= z_min + 0.55 * H_total)
    cockpit_nodes = np.where(cond_ck)[0]
    if len(cockpit_nodes) < 2:
        cockpit_nodes = np.argsort(np.abs(x - (x_min + 0.55 * L_total)))[:6]

    main_hoop_top = np.where(z >= z_min + 0.75 * H_total)[0]
    if len(main_hoop_top) < 1:
        main_hoop_top = np.argsort(z)[-2:]

    cond_en = (x >= x_min + 0.68 * L_total) & (x <= x_min + 0.92 * L_total) & (z <= z_min + 0.55 * H_total)
    engine_nodes = np.where(cond_en)[0]
    if len(engine_nodes) < 2:
        engine_nodes = np.argsort(np.abs(x - (x_min + 0.80 * L_total)))[:4]

    cond_rs = (x >= x_min + 0.78 * L_total) & (x <= x_min + 0.96 * L_total) & (z <= z_min + 0.50 * H_total)
    rear_susp = np.where(cond_rs)[0]
    if len(rear_susp) < 2:
        rear_susp = np.argsort(x)[-6:-2]

    jacking_nodes = np.where(x >= x_min + 0.92 * L_total)[0]
    if len(jacking_nodes) < 2:
        jacking_nodes = np.argsort(x)[-4:]

    return {
        "front_bulkhead": front_bulkhead,
        "front_susp": front_susp,
        "cockpit": cockpit_nodes,
        "main_hoop_top": main_hoop_top,
        "engine": engine_nodes,
        "rear_susp": rear_susp,
        "jacking": jacking_nodes
    }


def build_test_case(test_name, nodes, regions, params):
    fixed_dofs = []
    nodal_loads = {i: np.zeros(6) for i in range(len(nodes))}
    g = 9.81
    m_pilot = params["m_pilot"]
    m_engine = params["m_engine"]
    m_misc = params["m_misc"]

    def apply_mass_accelerations(ax_g, ay_g, az_g):
        n_c = len(regions["cockpit"])
        f_c = np.array([ax_g, ay_g, az_g, 0, 0, 0]) * ((m_pilot + 0.6 * m_misc) * g / max(n_c, 1))
        for nid in regions["cockpit"]:
            nodal_loads[nid] += f_c
        n_e = len(regions["engine"])
        f_e = np.array([ax_g, ay_g, az_g, 0, 0, 0]) * ((m_engine + 0.4 * m_misc) * g / max(n_e, 1))
        for nid in regions["engine"]:
            nodal_loads[nid] += f_e

    if test_name == "torsion":
        for nid in regions["rear_susp"]:
            fixed_dofs.extend([6*nid + j for j in range(6)])
        y_coords = nodes[regions["front_susp"], 1]
        left_nodes = regions["front_susp"][y_coords > 10.0]
        right_nodes = regions["front_susp"][y_coords < -10.0]
        if len(left_nodes) == 0 or len(right_nodes) == 0:
            order_y = np.argsort(y_coords)
            half = max(len(order_y) // 2, 1)
            right_nodes = regions["front_susp"][order_y[:half]]
            left_nodes = regions["front_susp"][order_y[-half:]]

        lever_arm_mm = max(abs(np.mean(nodes[left_nodes, 1]) - np.mean(nodes[right_nodes, 1])), 400.0)
        torque_Nmm = params["torque_Nm"] * 1000.0
        F_vert_N = torque_Nmm / lever_arm_mm

        for nid in left_nodes:
            nodal_loads[nid][2] += F_vert_N / len(left_nodes)
        for nid in right_nodes:
            nodal_loads[nid][2] -= F_vert_N / len(right_nodes)

        meta = {
            "lever_arm_mm": lever_arm_mm,
            "left_nodes": left_nodes,
            "right_nodes": right_nodes
        }
        return fixed_dofs, nodal_loads, meta

    elif test_name == "braking":
        for nid in np.concatenate([regions["front_susp"], regions["rear_susp"]]):
            fixed_dofs.extend([6*nid, 6*nid+1, 6*nid+2])
        apply_mass_accelerations(-params["a_brake_g"], 0.0, -1.0)
        for nid in regions["front_bulkhead"]:
            nodal_loads[nid][0] -= 2000.0 / len(regions["front_bulkhead"])
        return fixed_dofs, nodal_loads, {}

    elif test_name == "acceleration":
        for nid in regions["rear_susp"]:
            fixed_dofs.extend([6*nid, 6*nid+1, 6*nid+2])
        for nid in regions["front_susp"]:
            fixed_dofs.extend([6*nid+1, 6*nid+2])
        apply_mass_accelerations(params["a_accel_g"], 0.0, -1.0)
        return fixed_dofs, nodal_loads, {}

    elif test_name == "cornering":
        for nid in np.concatenate([regions["front_susp"], regions["rear_susp"]]):
            fixed_dofs.extend([6*nid, 6*nid+1, 6*nid+2])
        apply_mass_accelerations(0.0, params["a_lat_g"], -1.0)
        return fixed_dofs, nodal_loads, {}

    elif test_name == "towing":
        for nid in regions["jacking"]:
            fixed_dofs.extend([6*nid + j for j in range(6)])
        apply_mass_accelerations(0.0, 0.0, -1.0)
        return fixed_dofs, nodal_loads, {}

    elif test_name == "impact_fsae":
        for nid in regions["rear_susp"]:
            fixed_dofs.extend([6*nid + j for j in range(6)])
        F_impact_N = params["f_impact_kN"] * 1000.0
        for nid in regions["front_bulkhead"]:
            nodal_loads[nid][0] += F_impact_N / len(regions["front_bulkhead"])
        F_harness_N = 30000.0
        for nid in regions["cockpit"]:
            nodal_loads[nid][0] -= F_harness_N / len(regions["cockpit"])
        for nid in regions["main_hoop_top"]:
            nodal_loads[nid][0] += 900.0 / len(regions["main_hoop_top"])
        return fixed_dofs, nodal_loads, {}


# =====================================================================
# 5. RENDERIZADOR 3D INTERACTIVO CON CÁMARA ESTABILIZADA (PLOTLY)
# =====================================================================
def plot_chassis_3d(nodes, elements, res, mode="Von Mises (MPa)", scale_def=15.0, sy_limit=305.0):
    u_xyz = res["u_xyz"]
    max_raw_disp = float(np.max(res["disp_node"]))
    chassis_span = float(np.max(np.ptp(nodes, axis=0)))

    if max_raw_disp * scale_def > 0.12 * chassis_span and max_raw_disp > 1e-6:
        eff_scale = (0.12 * chassis_span) / max_raw_disp
    else:
        eff_scale = scale_def

    def_nodes = nodes + u_xyz * eff_scale

    if mode == "Von Mises (MPa)":
        values = res["vm_elem"]
        unit_label = "MPa"
        cmin, cmax = 0.0, max(float(np.percentile(values, 98)), 1.0)
    else:
        values = res["disp_elem"]
        unit_label = "mm"
        cmin, cmax = 0.0, max(float(np.percentile(values, 98)), 1e-4)

    fig = go.Figure()

    orig_x, orig_y, orig_z = [], [], []
    for n1, n2 in elements:
        orig_x.extend([nodes[n1, 0], nodes[n2, 0], None])
        orig_y.extend([nodes[n1, 1], nodes[n2, 1], None])
        orig_z.extend([nodes[n1, 2], nodes[n2, 2], None])

    fig.add_trace(go.Scatter3d(
        x=orig_x, y=orig_y, z=orig_z,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.25)", width=2, dash="dot"),
        name="Geometría Original (mm)",
        hoverinfo="skip"
    ))

    colorscale_fsae = [
        [0.00, "#FFFFFF"],
        [0.25, "#CCCCCC"],
        [0.50, "#FF9900"],
        [0.75, "#FF3B00"],
        [1.00, "#E10600"]
    ]

    def_x, def_y, def_z, line_colors, hover_texts = [], [], [], [], []
    for idx, (n1, n2) in enumerate(elements):
        val = float(np.clip(values[idx], cmin, cmax))
        vm_val = res["vm_elem"][idx]
        d_val = res["disp_elem"][idx]
        L_tube = np.linalg.norm(nodes[n2] - nodes[n1])
        FS = sy_limit / vm_val if vm_val > 1e-3 else 99.9

        txt = (
            f"<b>Tubo #{idx+1} (Nodos {n1}–{n2})</b><br>"
            f"Longitud: {L_tube:.1f} mm<br>"
            f"Von Mises: {vm_val:.2f} MPa<br>"
            f"Desplazamiento: {d_val:.3f} mm<br>"
            f"Factor de Seguridad: {FS:.2f}"
        )
        def_x.extend([def_nodes[n1, 0], def_nodes[n2, 0], None])
        def_y.extend([def_nodes[n1, 1], def_nodes[n2, 1], None])
        def_z.extend([def_nodes[n1, 2], def_nodes[n2, 2], None])
        line_colors.extend([val, val, val])
        hover_texts.extend([txt, txt, txt])

    fig.add_trace(go.Scatter3d(
        x=def_x, y=def_y, z=def_z,
        mode="lines",
        line=dict(
            width=8,
            color=line_colors,
            colorscale=colorscale_fsae,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=dict(text=mode, font=dict(color="#FFFFFF")),
                tickfont=dict(color="#FFFFFF"),
                bgcolor="#121216",
                bordercolor="#E10600",
                borderwidth=1
            )
        ),
        text=hover_texts,
        hoverinfo="text",
        name=f"Chasis Analizado ({mode})"
    ))

    max_elem_idx = int(np.argmax(values))
    n1_crit, n2_crit = elements[max_elem_idx]
    mid_crit = 0.5 * (def_nodes[n1_crit] + def_nodes[n2_crit])

    fig.add_trace(go.Scatter3d(
        x=[mid_crit[0]], y=[mid_crit[1]], z=[mid_crit[2]],
        mode="markers+text",
        marker=dict(size=7, color="#E10600", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
        text=[f"MÁX: {values[max_elem_idx]:.1f} {unit_label} (Tubo #{max_elem_idx+1})"],
        textposition="top center",
        textfont=dict(color="#FFFFFF", size=12),
        name="Punto Crítico Máximo"
    ))

    pad = 0.12 * chassis_span
    xr = [float(nodes[:, 0].min() - pad), float(nodes[:, 0].max() + pad)]
    yr = [float(nodes[:, 1].min() - pad), float(nodes[:, 1].max() + pad)]
    zr = [float(nodes[:, 2].min() - pad), float(nodes[:, 2].max() + pad)]

    fig.update_layout(
        paper_bgcolor="#0E0E12",
        plot_bgcolor="#0E0E12",
        margin=dict(l=0, r=0, t=30, b=0),
        height=580,
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(18, 18, 22, 0.85)",
            bordercolor="#E10600",
            borderwidth=1,
            font=dict(color="#FFFFFF")
        ),
        scene=dict(
            bgcolor="#0E0E12",
            xaxis=dict(title="X Longitudinal (mm)", range=xr, color="#FFFFFF", gridcolor="#22222A", zerolinecolor="#E10600"),
            yaxis=dict(title="Y Lateral (mm)", range=yr, color="#FFFFFF", gridcolor="#22222A", zerolinecolor="#E10600"),
            zaxis=dict(title="Z Vertical (mm)", range=zr, color="#FFFFFF", gridcolor="#22222A", zerolinecolor="#E10600"),
            aspectmode="data",
            camera=dict(eye=dict(x=-1.45, y=-1.55, z=0.85))
        )
    )
    return fig


# =====================================================================
# 6. INTERFAZ PRINCIPAL Y PESTAÑAS DE PRUEBAS
# =====================================================================
st.title("🏎️ FSAE 2026 | Analizador Estructural 3D de Chasis (mm)")
st.markdown(
    "Plataforma de verificación por **Elementos Finitos 3D en Milímetros (`mm`, `N`, `MPa`)** "
    "basada en el reglamento **Formula SAE® Rules 2026**."
)

with st.sidebar:
    st.header("📂 1. Geometría CAD (en mm)")
    uploaded_file = st.file_uploader(
        "Subir estructura del chasis (.IGS, .IGES, .DXF)",
        type=["igs", "iges", "dxf"]
    )

    unit_option = st.selectbox(
        "Escala de coordenadas del archivo CAD:",
        [
            "Milímetros (mm) — Directo 1:1",
            "Metros a Milímetros (x1000)",
            "Centímetros a Milímetros (x10)",
            "Pulgadas a Milímetros (x25.4)"
        ],
        index=0
    )
    scale_map = {
        "Milímetros (mm) — Directo 1:1": 1.0,
        "Metros a Milímetros (x1000)": 1000.0,
        "Centímetros a Milímetros (x10)": 10.0,
        "Pulgadas a Milímetros (x25.4)": 25.4
    }
    cad_scale = scale_map[unit_option]

    st.caption("Ajuste de orientación del chasis:")
    rot_roll_90 = st.checkbox("Rotar 90° sobre eje X (Intercambiar Y ↔ Z)", value=False)
    invert_z = st.checkbox("Invertir Arriba ↔ Abajo (Eje Z)", value=False)
    flip_front_rear = st.checkbox("Invertir Frente ↔ Atrás (Eje X)", value=False)
    tol_mm = st.number_input("Tolerancia soldadura de nodos (mm)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)

    st.header("🔧 2. Tubería y Material (F.3.4)")
    tube_preset = st.selectbox(
        "Categoría de Tubo FSAE 2026 (Regla F.3.4.1)",
        [
            "Size A - Roll Hoops / Shoulder Bar (25.0 x 2.5 mm)",
            "Size B - Front Bulkhead / Side Impact (25.4 x 1.6 mm)",
            "Size C - Structural / Bracing Supports (25.4 x 1.25 mm)",
            "Size D - Bent Upper Side Impact (35.0 x 1.25 mm)",
            "Personalizado (Catálogo)"
        ]
    )

    preset_map = {
        "Size A - Roll Hoops / Shoulder Bar (25.0 x 2.5 mm)": (25.0, 2.5),
        "Size B - Front Bulkhead / Side Impact (25.4 x 1.6 mm)": (25.4, 1.6),
        "Size C - Structural / Bracing Supports (25.4 x 1.25 mm)": (25.4, 1.25),
        "Size D - Bent Upper Side Impact (35.0 x 1.25 mm)": (35.0, 1.25),
        "Personalizado (Catálogo)": (28.0, 2.3)
    }
    def_od, def_t = preset_map[tube_preset]

    col_s1, col_s2 = st.columns(2)
    od_mm = col_s1.number_input("Diámetro Ext. (mm)", min_value=15.0, max_value=60.0, value=def_od, step=0.5)
    t_mm = col_s2.number_input("Espesor (mm)", min_value=0.8, max_value=5.0, value=def_t, step=0.1)

    mat_criteria = st.radio(
        "Criterio de Límite Elástico (Regla F.3.4.2)",
        ["Acero Continuo SES (Sy = 305 MPa)", "Unión Soldada SES (Sy = 180 MPa)"],
        index=0
    )
    sy_limit = 305.0 if "305" in mat_criteria else 180.0
    E_mpa = 200000.0

    st.header("⚖️ 3. Masas y Cargas")
    m_pilot = st.slider("Masa del Piloto (kg)", 50, 110, 75)
    m_engine = st.slider("Masa Motor / Tractive Pack (kg)", 25, 100, 58)
    m_misc = st.slider("Resto de Masas Suspendidas (kg)", 20, 120, 45)

    st.header("🎨 4. Visualización 3D")
    scale_def = st.slider("Factor de Amplificación de Deformada", 1, 100, 15)

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext == "dxf":
        nodes, elements = parse_dxf(raw_text, cad_scale, rot_roll_90, flip_front_rear, invert_z, tol_mm)
    else:
        nodes, elements = parse_iges(raw_text, cad_scale, rot_roll_90, flip_front_rear, invert_z, tol_mm)

    if len(nodes) < 4 or len(elements) < 3:
        st.warning("⚠️ El archivo no contiene suficientes líneas 3D conectadas. Mostrando chasis de referencia.")
        nodes, elements = generate_reference_fsae_chassis()
    else:
        spans_mm = np.ptp(nodes, axis=0)
        st.success(
            f"✅ Archivo **{uploaded_file.name}** orientado en **MILÍMETROS**: "
            f"**{len(nodes)} nodos** y **{len(elements)} tubos** | "
            f"Largo X = **{spans_mm[0]:.1f} mm**, Ancho Y = **{spans_mm[1]:.1f} mm**, Alto Z = **{spans_mm[2]:.1f} mm**."
        )
else:
    st.info(
        "💡 Mostrando el **Chasis Tubular FSAE de Referencia en Milímetros** "
        "(Largo: **2480 mm**, Ancho: **1040 mm**, Alto: **1109 mm**). "
        "Subí tu archivo `.IGS` o `.DXF` en el menú izquierdo."
    )
    nodes, elements = generate_reference_fsae_chassis()

regions = identify_key_regions(nodes)

prop_dict = {
    "E": E_mpa,
    "G": E_mpa / (2.0 * (1.0 + 0.3)),
    "OD": od_mm,
    "t": t_mm,
    "rho": 7.85e-6
}

d_in = max(od_mm - 2.0 * t_mm, 0.1)
area_sec = (math.pi / 4.0) * (od_mm**2 - d_in**2)
inertia_sec = (math.pi / 64.0) * (od_mm**4 - d_in**4)

tabs = st.tabs([
    "🔄 1. Rigidez Torsional",
    "🛑 2. Frenada (1.2g + Pedal 2kN)",
    "🚀 3. Aceleración (0.7g)",
    "📐 4. Curva y Tilt Test (60° / 1.7g)",
    "🏗️ 5. Remolque (Jacking Point)",
    "💥 6. Impacto Frontal (120 kN) y Arnés"
])

# PESTAÑA 1: TORSIÓN
with tabs[0]:
    col_ctrl, col_viz = st.columns([1, 2.6])
    with col_ctrl:
        st.subheader("Parámetros de Torsión")
        torque_Nm = st.number_input("Par Torsor M (N·m)", 500.0, 10000.0, 3000.0, 250.0)
        target_K = st.number_input("Rigidez Objetivo (N·m/°)", 1000.0, 6000.0, 3000.0, 100.0)
        viz_mode_1 = st.radio("Colorear en 3D:", ["Von Mises (MPa)", "Desplazamiento (mm)"], key="v1")

    fixed_1, loads_1, meta_1 = build_test_case("torsion", nodes, regions, {
        "m_pilot": m_pilot, "m_engine": m_engine, "m_misc": m_misc, "torque_Nm": torque_Nm
    })
    res_1 = solve_frame_3d(nodes, elements, prop_dict, fixed_1, loads_1)

    z_left = np.mean(np.abs(res_1["u_xyz"][meta_1["left_nodes"], 2]))
    z_right = np.mean(np.abs(res_1["u_xyz"][meta_1["right_nodes"], 2]))
    z_defl_mm = max(0.5 * (z_left + z_right), 1e-3)
    half_span_mm = meta_1["lever_arm_mm"] / 2.0
    theta_deg = max(math.degrees(math.atan(z_defl_mm / max(half_span_mm, 1.0))), 1e-4)
    K_torsion = torque_Nm / theta_deg
    vm_max_1 = float(np.max(res_1["vm_elem"]))
    disp_max_1 = float(np.max(res_1["disp_node"]))

    with col_ctrl:
        ok_1 = K_torsion >= target_K
        msg_1 = f"Rigidez Torsional: {K_torsion:.1f} N·m/° (Meta: {target_K:.0f} N·m/°)"
        mostrar_estado(ok_1, msg_1)
        st.metric("Rigidez Torsional (K)", f"{K_torsion:.1f} N·m/°", f"{K_torsion - target_K:+.1f} N·m/°")
        st.metric("Ángulo de Giro (θ)", f"{theta_deg:.3f} °")
        st.metric("Von Mises Máximo", f"{vm_max_1:.1f} MPa")
        st.metric("Desplazamiento Máx.", f"{disp_max_1:.2f} mm")
        st.metric("Masa Chasis", f"{res_1['mass_kg']:.2f} kg ({res_1['total_length_mm']/1000:.1f} m)")

    with col_viz:
        fig_1 = plot_chassis_3d(nodes, elements, res_1, viz_mode_1, scale_def, sy_limit)
        st.plotly_chart(fig_1, use_container_width=True)

# PESTAÑA 2: FRENADA
with tabs[1]:
    col_ctrl, col_viz = st.columns([1, 2.6])
    with col_ctrl:
        st.subheader("Parámetros de Frenada")
        a_brake_g = st.slider("Desaceleración de Frenada (g)", 0.8, 2.5, 1.2, 0.1)
        st.caption("Incluye 2000 N en soporte de pedal (Regla T.3.2.2).")
        viz_mode_2 = st.radio("Colorear en 3D:", ["Von Mises (MPa)", "Desplazamiento (mm)"], key="v2")

    fixed_2, loads_2, _ = build_test_case("braking", nodes, regions, {
        "m_pilot": m_pilot, "m_engine": m_engine, "m_misc": m_misc, "a_brake_g": a_brake_g
    })
    res_2 = solve_frame_3d(nodes, elements, prop_dict, fixed_2, loads_2)
    vm_max_2 = float(np.max(res_2["vm_elem"]))
    disp_max_2 = float(np.max(res_2["disp_node"]))
    fs_2 = sy_limit / max(vm_max_2, 1e-3)

    with col_ctrl:
        ok_2 = (vm_max_2 <= sy_limit) and (disp_max_2 <= 2.0)
        msg_2 = f"Factor de Seguridad = {fs_2:.2f} | Deformación máx: {disp_max_2:.2f} mm"
        mostrar_estado(ok_2, msg_2)
        st.metric("Von Mises Máximo", f"{vm_max_2:.2f} MPa", f"Límite: {sy_limit:.0f} MPa")
        st.metric("Desplazamiento Máx.", f"{disp_max_2:.3f} mm", "Ref: < 2.0 mm")
        st.metric("Factor de Seguridad (FS)", f"{fs_2:.2f}")

    with col_viz:
        fig_2 = plot_chassis_3d(nodes, elements, res_2, viz_mode_2, scale_def, sy_limit)
        st.plotly_chart(fig_2, use_container_width=True)

# PESTAÑA 3: ACELERACIÓN
with tabs[2]:
    col_ctrl, col_viz = st.columns([1, 2.6])
    with col_ctrl:
        st.subheader("Parámetros de Aceleración")
        a_accel_g = st.slider("Aceleración Longitudinal (g)", 0.4, 2.0, 0.7, 0.1)
        viz_mode_3 = st.radio("Colorear en 3D:", ["Von Mises (MPa)", "Desplazamiento (mm)"], key="v3")

    fixed_3, loads_3, _ = build_test_case("acceleration", nodes, regions, {
        "m_pilot": m_pilot, "m_engine": m_engine, "m_misc": m_misc, "a_accel_g": a_accel_g
    })
    res_3 = solve_frame_3d(nodes, elements, prop_dict, fixed_3, loads_3)
    vm_max_3 = float(np.max(res_3["vm_elem"]))
    disp_max_3 = float(np.max(res_3["disp_node"]))
    fs_3 = sy_limit / max(vm_max_3, 1e-3)

    with col_ctrl:
        ok_3 = (vm_max_3 <= sy_limit) and (disp_max_3 <= 2.0)
        msg_3 = f"Factor de Seguridad = {fs_3:.2f} | Tensión: {vm_max_3:.1f} MPa"
        mostrar_estado(ok_3, msg_3)
        st.metric("Von Mises Máximo", f"{vm_max_3:.2f} MPa", f"Límite: {sy_limit:.0f} MPa")
        st.metric("Desplazamiento Máx.", f"{disp_max_3:.3f} mm")
        st.metric("Factor de Seguridad (FS)", f"{fs_3:.2f}")

    with col_viz:
        fig_3 = plot_chassis_3d(nodes, elements, res_3, viz_mode_3, scale_def, sy_limit)
        st.plotly_chart(fig_3, use_container_width=True)

# PESTAÑA 4: CURVA Y TILT TEST
with tabs[3]:
    col_ctrl, col_viz = st.columns([1, 2.6])
    with col_ctrl:
        st.subheader("Curva / Tilt Test (IN.11.2.2)")
        lat_mode = st.selectbox(
            "Escenario Lateral",
            ["Tilt Test 60° (1.7 g - Regla IN.11.2.2)", "Curva en Pista (1.3 g)"]
        )
        default_lat = 1.7 if "1.7" in lat_mode else 1.3
        a_lat_g = st.slider("Aceleración Lateral (g)", 0.9, 2.5, default_lat, 0.1)
        viz_mode_4 = st.radio("Colorear en 3D:", ["Von Mises (MPa)", "Desplazamiento (mm)"], key="v4")

    fixed_4, loads_4, _ = build_test_case("cornering", nodes, regions, {
        "m_pilot": m_pilot, "m_engine": m_engine, "m_misc": m_misc, "a_lat_g": a_lat_g
    })
    res_4 = solve_frame_3d(nodes, elements, prop_dict, fixed_4, loads_4)
    vm_max_4 = float(np.max(res_4["vm_elem"]))
    disp_max_4 = float(np.max(res_4["disp_node"]))
    fs_4 = sy_limit / max(vm_max_4, 1e-3)

    with col_ctrl:
        ok_4 = (vm_max_4 <= sy_limit) and (disp_max_4 <= 2.5)
        msg_4 = f"Soporta {a_lat_g:.1f} g laterales (FS = {fs_4:.2f})"
        mostrar_estado(ok_4, msg_4)
        st.metric("Von Mises Máximo", f"{vm_max_4:.2f} MPa", f"Límite: {sy_limit:.0f} MPa")
        st.metric("Desplazamiento Lateral Máx.", f"{disp_max_4:.3f} mm")
        st.metric("Factor de Seguridad (FS)", f"{fs_4:.2f}")

    with col_viz:
        fig_4 = plot_chassis_3d(nodes, elements, res_4, viz_mode_4, scale_def, sy_limit)
        st.plotly_chart(fig_4, use_container_width=True)

# PESTAÑA 5: REMOLQUE
with tabs[4]:
    col_ctrl, col_viz = st.columns([1, 2.6])
    with col_ctrl:
        st.subheader("Prueba de Remolque (VE.2.1)")
        st.caption("Sujeción en el Jacking Point trasero soportando el peso del vehículo.")
        viz_mode_5 = st.radio("Colorear en 3D:", ["Von Mises (MPa)", "Desplazamiento (mm)"], key="v5")

    fixed_5, loads_5, _ = build_test_case("towing", nodes, regions, {
        "m_pilot": m_pilot, "m_engine": m_engine, "m_misc": m_misc
    })
    res_5 = solve_frame_3d(nodes, elements, prop_dict, fixed_5, loads_5)
    vm_max_5 = float(np.max(res_5["vm_elem"]))
    disp_max_5 = float(np.max(res_5["disp_node"]))
    fs_5 = sy_limit / max(vm_max_5, 1e-3)

    with col_ctrl:
        ok_5 = vm_max_5 <= sy_limit
        msg_5 = f"Jacking Point verificado (FS = {fs_5:.2f} | {vm_max_5:.1f} MPa)"
        mostrar_estado(ok_5, msg_5)
        st.metric("Von Mises Máximo", f"{vm_max_5:.2f} MPa", f"Límite: {sy_limit:.0f} MPa")
        st.metric("Desplazamiento en Punta", f"{disp_max_5:.2f} mm")
        st.metric("Factor de Seguridad (FS)", f"{fs_5:.2f}")

    with col_viz:
        fig_5 = plot_chassis_3d(nodes, elements, res_5, viz_mode_5, scale_def, sy_limit)
        st.plotly_chart(fig_5, use_container_width=True)

# PESTAÑA 6: IMPACTO FRONTAL Y ARNÉS
with tabs[5]:
    col_ctrl, col_viz = st.columns([1, 2.6])
    with col_ctrl:
        st.subheader("Impacto y Seguridad FSAE 2026")
        ia_type = st.selectbox(
            "Nivel de Carga Frontal (Reglas F.8.3 / F.8.7)",
            [
                "Placa Anti-Intrusión (120 kN - Regla F.8.3.1)",
                "Atenuador Espuma (95 kN - Regla F.8.7.2)",
                "Atenuador Honeycomb (60 kN - Regla F.8.7.2)"
            ]
        )
        f_imp = 120.0 if "120" in ia_type else (95.0 if "95" in ia_type else 60.0)
        st.caption("Incluye 30 kN en arnés (F.7.9.1) y 900 N en reposacabezas (T.2.8.5).")
        viz_mode_6 = st.radio("Colorear en 3D:", ["Von Mises (MPa)", "Desplazamiento (mm)"], key="v6")

    fixed_6, loads_6, _ = build_test_case("impact_fsae", nodes, regions, {
        "m_pilot": m_pilot, "m_engine": m_engine, "m_misc": m_misc, "f_impact_kN": f_imp
    })
    res_6 = solve_frame_3d(nodes, elements, prop_dict, fixed_6, loads_6)
    vm_max_6 = float(np.max(res_6["vm_elem"]))
    disp_max_6 = float(np.max(res_6["disp_node"]))
    su_limit = 365.0
    fs_6 = su_limit / max(vm_max_6, 1e-3)

    with col_ctrl:
        ok_6 = (vm_max_6 <= su_limit) and (disp_max_6 <= 25.0)
        msg_6 = f"Impacto F.8: {vm_max_6:.1f} MPa (Límite Su: {su_limit:.0f} MPa | FS: {fs_6:.2f})"
        mostrar_estado(ok_6, msg_6)
        st.metric("Von Mises Pico Impacto", f"{vm_max_6:.1f} MPa", f"Su máx: {su_limit:.0f} MPa")
        st.metric("Deflexión Front Bulkhead", f"{disp_max_6:.2f} mm", "Máx F.8.8.6: 25.0 mm")
        st.metric("Área / Inercia Tubo", f"{area_sec:.0f} mm² | {inertia_sec:.0f} mm⁴")

    with col_viz:
        fig_6 = plot_chassis_3d(nodes, elements, res_6, viz_mode_6, max(scale_def // 4, 1), su_limit)
        st.plotly_chart(fig_6, use_container_width=True)
