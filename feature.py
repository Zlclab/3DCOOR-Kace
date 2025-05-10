import os
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations


def read_fasta(file):
    name1 = []
    names = []
    labels = []
    seqs = []

    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if '>' in line:
            names.append(line)
            name1.append(line.strip('>').split()[0])
            if 'pos' in line:
                labels.append(1)
            else:
                labels.append(0)
        else:
            seqs.append(line)
    return name1, labels, seqs


def out_file(code, labels):
    data_withlabel = []
    name = []
    name.append('class')
    for j in range(code.shape[1]):
        name.append(('V' + str(j + 1)))
    # data_withlabel.append(name)
    for i in range(len(labels)):
        a = list(code[i])
        a.insert(0, labels[i])
        data_withlabel.append(a)
    df_train = pd.DataFrame(data_withlabel, columns=name)

    return df_train


# 旋转算子
def rotation(a, r):
    if a == "x":
        R = np.array([[1, 0, 0],
                      [0, np.cos(r), -np.sin(r)],
                      [0, np.sin(r), np.cos(r)]])
    elif a == "y":
        R = np.array([[np.cos(r), 0, np.sin(r)],
                      [0, 1, 0],
                      [-np.sin(r), 0, np.cos(r)]])
    elif a == "z":
        R = np.array([[np.cos(r), -np.sin(r), 0],
                      [np.sin(r), np.cos(r), 0],
                      [0, 0, 1]])  #
    else:
        raise ValueError(f"Invalid axis '{a}', expected 'x', 'y', or 'z'.")  # 直接抛出异常
    return R


# S
def s_i(stru):
    k = 0
    kk_1 = []  #
    while k < len(stru):
        count = 1
        while k + 1 < len(stru) and stru[k] == stru[k + 1]:
            count += 1
            k += 1
        kk_1.extend([count] * count)
        k += 1

    return kk_1


def create_graph_from_coordinates(coordinates, threshold):
    w = len(coordinates)
    net = np.zeros((w, w), dtype=int)
    for p in range(w):
        for q in range(w):
            if p != q:
                d_pq = np.sqrt((coordinates[p][0] - coordinates[q][0]) ** 2 +
                               (coordinates[p][1] - coordinates[q][1]) ** 2 +
                               (coordinates[p][2] - coordinates[q][2]) ** 2)
                if d_pq <= threshold:
                    net[p, q] = 1
    return nx.from_numpy_array(net)



# 计算距离矩阵
def compute_distance_matrix(coords):
    num_points = len(coords)
    dist_matrix = np.zeros((num_points, num_points))  # 初始化距离矩阵

    # 计算任意两点间的距离
    for i in range(num_points):
        for j in range(i + 1, num_points):
            # 计算欧几里得距离
            dist = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # 距离矩阵是对称的

    return dist_matrix


def categorization_property(property):
    keys = 'ARNDCQEGHILKMFPSTWYVO'

    # 定义分类组常量
    group_dict = {
        'c': {
            'G0': ['O'],
            'G1': ['A', 'C', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'P', 'Q', 'S', 'T', 'V', 'W', 'Y'],
            'G2': ['D', 'E'],
            'G3': ['K', 'R']
        },
        'h': {
            'G0': ['O'],
            'G1': ['C', 'F', 'I', 'L', 'M', 'V', 'W'],
            'G2': ['A', 'G', 'H', 'P', 'S', 'T', 'Y'],
            'G3': ['D', 'E', 'K', 'N', 'Q', 'R']
        },
        'v ': {
            'G0': ['O'],
            'G1': ['A', 'C', 'D', 'G', 'P', 'S', 'T'],
            'G2': ['E', 'I', 'L', 'N', 'Q', 'V'],
            'G3': ['F', 'H', 'K', 'M', 'R', 'W', 'Y']
        },
        'p': {
            'G0': ['O'],
            'G1': ['C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y'],
            'G2': ['A', 'G', 'P', 'S', 'T'],
            'G3': ['D', 'E', 'H', 'K', 'N', 'Q', 'R']
        },
        'pz': {
            'G0': ['O'],
            'G1': ['A', 'D', 'G', 'S', 'T'],
            'G2': ['C', 'E', 'I', 'L', 'N', 'P', 'Q', 'V'],
            'G3': ['F', 'H', 'K', 'M', 'R', 'W', 'Y']
        },
        'ss': {
            'G0': ['O'],
            'G1': ['D', 'G', 'N', 'P', 'S'],
            'G2': ['A', 'E', 'H', 'K', 'L', 'M', 'Q', 'R'],
            'G3': ['C', 'F', 'I', 'T', 'V', 'W', 'Y']
        },
        'sa': {
            'G0': ['O'],
            'G1': ['A', 'L', 'F', 'C', 'G', 'I', 'V', 'W'],
            'G2': ['R', 'K', 'Q', 'E', 'N', 'D'],
            'G3': ['M', 'S', 'P', 'T', 'H', 'Y']
        }
    }

    # 如果 property 不在预定义分类中，抛出异常
    if property not in group_dict:
        raise ValueError(f"Invalid property: {property}. Expected one of {list(group_dict.keys())}.")

    # 获取该属性的分类组
    group_g0, group_g1, group_g2, group_g3 = group_dict[property].values()

    # 创建字典 dc，映射氨基酸到分类组
    dc = {}
    for key in keys:
        if key in group_g0:
            dc[key] = 'G0'
        elif key in group_g1:
            dc[key] = 'G1'
        elif key in group_g2:
            dc[key] = 'G2'
        elif key in group_g3:
            dc[key] = 'G3'
        else:
            raise ValueError(f"Unexpected amino acid: {key}")

    return dc


def seq_stru_feature(path, prop, a1, a2, a3):
    name1, labels, seqs = read_fasta(path)

    s = len(name1)
    w = len(seqs[0])
    # w,s,#w为序列长度,s为样本个数
    print(s, w)
    # 定义文件夹名称
    folder_name = f'net_{prop}_{a1}_{a2}_{a3}'
    if not os.path.exists(folder_name):

        keys = 'ARNDCQEGHILKMFPSTWYVO'

        # 生成字典

        # n = 21
        I = np.identity(w)

        dict_code = {}
        i = 0
        for key in keys:
            dict_code[key] = list(I[i])
            i += 1
        one_hot = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + dict_code[mod]
            one_hot.append(mat)
        one_hot = np.array(one_hot)
        data_one_hot = out_file(one_hot, labels)

        #########################BLOSUM62阵
        blosum62 = {
            'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
            'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
            'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
            'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
            'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
            'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
            'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
            'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
            'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
            'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
            'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
            'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
            'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
            'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
            'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        }
        blosum = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + blosum62[mod]
            blosum.append(mat)
        blosum = np.array(blosum)
        data_blosum = out_file(blosum, labels)

        #######################################################

        AAindex_path = './' + 'AAindex.txt'
        AA = 'ARNDCQEGHILKMFPSTWYV'
        with open(AAindex_path, 'r') as f:
            lines = f.readlines()
        AAindex_dict = {}
        for line in lines:
            line = line.strip().split('\t')
            if line[0] == 'AccNo':
                for i in range(len(line) - 1):
                    AAindex_dict[line[i + 1]] = []
            else:
                for i in range(len(line) - 1):
                    AAindex_dict[AA[i]].append(line[i + 1])
        AAindex_dict['O'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        AAindex = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + AAindex_dict[mod]
            AAindex.append([float(num_str) for num_str in mat])
        AAindex = np.array(AAindex)
        data_AAindex = out_file(AAindex, labels)
        # data_AAindex.astype(float64)

        ##########################################SD
        SD_dict = {
            'A': [1, 0, 0, 0, 0, 0, 0],  # A
            'R': [0, 0, 0, 0, 1, 0, 0],  # R
            'N': [0, 0, 0, 1, 0, 0, 0],  # N
            'D': [0, 0, 0, 0, 0, 1, 0],  # D
            'C': [0, 0, 0, 0, 0, 0, 1],  # C
            'Q': [0, 0, 0, 1, 0, 0, 0],  # Q
            'E': [0, 0, 0, 0, 0, 1, 0],  # E
            'G': [1, 0, 0, 0, 0, 0, 0],  # G
            'H': [0, 0, 0, 1, 0, 0, 0],  # H
            'I': [0, 1, 0, 0, 0, 0, 0],  # I
            'L': [0, 1, 0, 0, 0, 0, 0],  # L
            'K': [0, 0, 0, 0, 1, 0, 0],  # K
            'M': [0, 0, 1, 0, 0, 0, 0],  # M
            'F': [0, 1, 0, 0, 0, 0, 0],  # F
            'P': [0, 1, 0, 0, 0, 0, 0],  # P
            'S': [0, 0, 1, 0, 0, 0, 0],  # S
            'T': [0, 0, 1, 0, 0, 0, 0],  # T
            'W': [0, 0, 0, 1, 0, 0, 0],  # W
            'Y': [0, 0, 1, 0, 0, 0, 0],  # Y
            'V': [1, 0, 0, 0, 0, 0, 0],  # V
            'O': [0, 0, 0, 0, 0, 0, 0],  # O
        }
        SD = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + SD_dict[mod]
            SD.append(mat)
        SD = np.array(SD)
        data_SD = out_file(SD, labels)

        PC_dict = {
            'A': [1, 0, 0, 0, 0],  # A
            'R': [0, 0, 1, 0, 0],  # R
            'N': [0, 0, 0, 0, 1],  # N
            'D': [0, 0, 0, 1, 0],  # D
            'C': [0, 0, 0, 0, 1],  # C
            'Q': [0, 0, 0, 0, 1],  # Q
            'E': [0, 0, 0, 1, 0],  # E
            'G': [1, 0, 0, 0, 0],  # G
            'H': [0, 0, 1, 0, 0],  # H
            'I': [1, 0, 0, 0, 0],  # I
            'L': [1, 0, 0, 0, 0],  # L
            'K': [0, 0, 1, 0, 0],  # K
            'M': [1, 0, 0, 0, 0],  # M
            'F': [0, 1, 0, 0, 0],  # F
            'P': [0, 0, 0, 0, 1],  # P
            'S': [0, 0, 0, 0, 1],  # S
            'T': [0, 0, 0, 0, 1],  # T
            'W': [0, 1, 0, 0, 0],  # W
            'Y': [0, 1, 0, 0, 0],  # Y
            'V': [1, 0, 0, 0, 0],  # V
            'O': [0, 0, 0, 0, 0],  # O
        }
        PC = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + PC_dict[mod]
            PC.append(mat)
        PC = np.array(PC)
        data_PC = out_file(PC, labels)

        EGB_dict = {
            'A': [1, 1, 1],  # A
            'R': [0, 0, 1],  # R
            'N': [1, 0, 0],  # N
            'D': [0, 1, 0],  # D
            'C': [1, 0, 0],  # C
            'Q': [1, 0, 0],  # Q
            'E': [0, 1, 0],  # E
            'G': [1, 1, 1],  # G
            'H': [0, 0, 1],  # H
            'I': [1, 1, 1],  # I
            'L': [1, 1, 1],  # L
            'K': [0, 0, 1],  # K
            'M': [1, 1, 1],  # M
            'F': [1, 1, 1],  # F
            'P': [1, 1, 1],  # P
            'S': [1, 0, 0],  # S
            'T': [1, 0, 0],  # T
            'W': [1, 1, 1],  # W
            'Y': [1, 0, 0],  # Y
            'V': [1, 1, 1],  # V
            'O': [0, 0, 0],  # O
        }
        EGB = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + EGB_dict[mod]
            EGB.append(mat)
        EGB = np.array(EGB)
        data_EGB = out_file(EGB, labels)

        PAM250 = {
            'A': [2, -2, 0, 0, -3, 1, -1, -1, -1, -2, -1, 0, 1, 0, -2, 1, 1, 0, -6, -3],  # A
            'R': [-2, -4, -1, -1, -4, -3, 2, -2, 3, -3, 0, 0, 0, 1, 6, 0, -1, -2, 2, -4],  # R
            'N': [0, -4, 2, 1, -3, 0, 2, -2, 1, -3, -2, 2, 0, 1, 0, 1, 0, -2, -4, -2],  # N
            'D': [0, -5, 4, 3, -6, 1, 1, -2, 0, -4, -3, 2, -1, 2, -1, 0, 0, -2, -7, -4],  # D
            'C': [-2, 12, -5, -5, -4, -3, -3, -2, -5, -6, -5, -4, -3, -5, -4, 0, -2, -2, -8, 0],  # C
            'Q': [0, -5, 2, 2, -5, -1, 3, -2, 1, -2, -1, 1, 0, 4, 1, -1, -1, -2, -5, -4],  # Q
            'E': [0, -5, 3, 4, -5, 0, 1, -2, 0, -3, -2, 1, -1, 2, -1, 0, 0, -2, -7, -4],  # E
            'G': [1, -3, 1, 0, -5, 5, -2, -3, -2, -4, -3, 0, 0, -1, -3, 1, 0, -1, -7, -5],  # G
            'H': [-1, -3, 1, 1, -2, -2, 6, -2, 0, -2, -2, 2, 0, 3, 2, -1, -1, -2, -3, 0],  # H
            'I': [-1, -2, -2, -2, 1, -3, -2, 5, -2, 2, 2, -2, -2, -2, -2, -1, 0, 4, -5, -1],  # I
            'L': [-2, -6, -4, -3, 2, -4, -2, 2, -3, 6, 4, -3, -3, -2, -3, -3, -2, 2, -2, -1],  # L
            'K': [-1, -5, 0, 0, -5, -2, 0, -2, 5, -3, 0, 1, -1, 1, 3, 0, 0, -2, -3, -4],  # K
            'M': [-1, -5, -3, -2, 0, -3, -2, 2, 0, 4, 6, -2, -2, -1, 0, -2, -1, 2, -4, -2],  # M
            'F': [-3, -4, -6, -5, 9, -5, -2, 1, -5, 2, 0, -3, -5, -5, -4, -3, -3, -1, 0, 7],  # F
            'P': [1, -3, -1, -1, -5, 0, 0, -2, -1, -3, -2, 0, 6, 0, 0, 1, 0, -1, -6, -5],  # P
            'S': [1, 0, 0, 0, -3, 1, -1, -1, 0, -3, -2, 1, 1, -1, 0, 2, 1, -1, -2, -3],  # S
            'T': [1, -2, 0, 0, -3, 0, -1, 0, 0, -2, -1, 0, 0, -1, -1, 1, 3, 0, -5, -3],  # T
            'W': [-6, -8, -7, -7, 0, -7, -3, -5, -3, -2, -4, -4, -6, -5, 2, -2, -5, -6, 17, 0],  # W
            'Y': [-3, 0, -4, -4, 7, -5, 0, -1, -4, -1, -2, -2, -5, -4, -4, -3, -3, -2, 0, 10],  # Y
            'V': [0, -2, -2, -2, -1, -1, -2, 4, -2, 2, 2, -2, -1, -2, -2, -1, 0, 4, -6, -2],  # V
            'O': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # O
        }
        PAM = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + PAM250[mod]
            PAM.append(mat)
        PAM = np.array(PAM)
        data_PAM = out_file(PAM, labels)

        zscale = {
            'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
            'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
            'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
            'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
            'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
            'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
            'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
            'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
            'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
            'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
            'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
            'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
            'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
            'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
            'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
            'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
            'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
            'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
            'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
            'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
            'O': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
        }
        Zscales = []
        for seq in seqs:
            mat = []
            for mod in seq:
                mat = mat + zscale[mod]
            Zscales.append(mat)
        Zscales = np.array(Zscales)
        data_Zscales = out_file(Zscales, labels)


        # 创建文件夹
        os.makedirs(folder_name, exist_ok=True)  # exist_ok=True 防止已存在时抛出异常

        dc = categorization_property(prop)

        Coor = []
        Coor0 = []
        c0 = np.array([1, 1, 1]).reshape((3, 1))
        for seq in seqs:
            sec = [dc.get(se) for se in seq]
            n_1 = sec.count('G1')
            n_2 = sec.count('G2')
            n_3 = sec.count('G3')
            S = s_i(sec)
            coor_1 = []
            coor_0 = [[1, 1, 1]]
            for j in range(w):
                if j == 0:
                    if sec[j] == 'G1':
                        coor = rotation('x', 2 * np.pi * a1 * S[j] / n_1) @ c0
                    elif sec[j] == 'G2':
                        coor = rotation('y', 2 * np.pi * a2 * S[j] / n_2) @ c0
                    elif sec[j] == 'G3':
                        coor = rotation('z', 2 * np.pi * a3 * S[j] / n_3) @ c0
                    elif sec[j] == 'G0':
                        coor = np.sqrt(3) / w + c0
                    # print(j,coor)
                elif j >= 1:
                    if sec[j] == 'G1':
                        coor = rotation('x', 2 * np.pi * a1 * S[j] / n_1) @ coor
                    elif sec[j] == 'G2':
                        coor = rotation('y', 2 * np.pi * a2 * S[j] / n_2) @ coor
                    elif sec[j] == 'G3':
                        coor = rotation('z', 2 * np.pi * a3 * S[j] / n_3) @ coor
                    elif sec[j] == 'G0':
                        coor = np.sqrt(3) / w + coor
                    # print(j,coor)
                coor_1.append(list(coor))
                coor_0.append(list(coor))
            # print("#####",i)
            Coor.append(coor_1)
            Coor0.append(coor_0)

        DD = []
        for i in range(s):
            DD_1 = []
            for p in range(w):
                for q in range(w):
                    if p < q:
                        dis = ((Coor[i][p][0] - Coor[i][q][0]) ** 2 + (Coor[i][p][1] - Coor[i][q][1]) ** 2 + (
                                Coor[i][p][2] - Coor[i][q][2]) ** 2) ** 0.5
                        DD_1.append(dis)
            DD.append(DD_1)
        aver = np.mean(DD, axis=1)
        # 网络特征
        ## 点
        CC = []
        CL = []
        PG = []
        DC = []
        EC = []
        BC = []
        ## 边
        EBC = []
        EJS = []
        ED = []


        js = 0
        for ii in range(s):

            G = create_graph_from_coordinates(Coor[ii], aver[ii])

            # 点特征
            closeness_centrality = nx.closeness_centrality(G)
            CC.append(closeness_centrality)

            clustering = nx.clustering(G)
            CL.append(clustering)

            pagerank = nx.pagerank(G)
            PG.append(pagerank)

            degree_centrality = nx.degree_centrality(G)
            DC.append(degree_centrality)

            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100000)
            EC.append(eigenvector_centrality)

            betweenness_centrality = nx.betweenness_centrality(G)
            BC.append(betweenness_centrality)

            # 边特征

            ## 边距离
            coords = np.array(Coor[ii])
            distance_matrix = compute_distance_matrix(coords)
            ED.append(distance_matrix)

            ## 边介数
            edge_betweenness = nx.edge_betweenness_centrality(G)
            matrix = np.zeros((w, w))
            for i in range(w):
                for j in range(i + 1, w):
                    # 获取边 (i, j) 或 (j, i) 的介数，若该边不存在则默认为 0
                    matrix[i][j] = edge_betweenness.get((i, j), 0)
                    matrix[j][i] = matrix[i][j]  # 确保对称矩阵
            EBC.append(matrix)

            ## Jaccard系数
            jaccard_scores = {(u, v): 0 for u, v in combinations(G.nodes, 2)}
            for u, v, score in nx.jaccard_coefficient(G):
                jaccard_scores[(u, v)] = score
            for u, v in G.edges():
                jaccard_scores[(u, v)] = 1
            nodes = list(G.nodes())
            jaccard_matrix = np.zeros((len(nodes), len(nodes)))
            for (u, v), score in jaccard_scores.items():
                u_idx = nodes.index(u)
                v_idx = nodes.index(v)
                jaccard_matrix[u_idx, v_idx] = score
                jaccard_matrix[v_idx, u_idx] = score  # Jaccard 系数是对称的
            EJS.append(jaccard_matrix)

            js += 1
            print(js)

        Y = pd.DataFrame(labels)
        Y.to_csv(f'./{folder_name}/labels.csv', index=False)

        CC_data = pd.DataFrame(CC)
        CL_data = pd.DataFrame(CL)
        PG_data = pd.DataFrame(PG)
        DC_data = pd.DataFrame(DC)
        EC_data = pd.DataFrame(EC)
        BC_data = pd.DataFrame(BC)


        CC_data.to_csv(f'./{folder_name}/CC_data.csv', index=False)
        BC_data.to_csv(f'./{folder_name}/BC_data.csv', index=False)
        EC_data.to_csv(f'./{folder_name}/EC_data.csv', index=False)
        DC_data.to_csv(f'./{folder_name}/DC_data.csv', index=False)
        CL_data.to_csv(f'./{folder_name}/CL_data.csv', index=False)
        PG_data.to_csv(f'./{folder_name}/PG_data.csv', index=False)

        np.save(f'./{folder_name}/EBC.npy', EBC)
        np.save(f'./{folder_name}/EJS.npy', EJS)
        np.save(f'./{folder_name}/ED.npy', ED)

        data_one_hot = data_one_hot.iloc[:, 1:]
        data_blosum = data_blosum.iloc[:, 1:]
        data_AAindex = data_AAindex.iloc[:, 1:]
        data_SD = data_SD.iloc[:, 1:]
        data_PC = data_PC.iloc[:, 1:]
        data_EGB = data_EGB.iloc[:, 1:]
        data_PAM = data_PAM.iloc[:, 1:]
        data_Zscales = data_Zscales.iloc[:, 1:]

        data_one_hot.to_csv(f'./{folder_name}/one_hot.csv', index=False)
        data_blosum.to_csv(f'./{folder_name}/blosum.csv', index=False)
        data_AAindex.to_csv(f'./{folder_name}/AAindex.csv', index=False)
        data_SD.to_csv(f'./{folder_name}/SD.csv', index=False)
        data_PC.to_csv(f'./{folder_name}/PC.csv', index=False)
        data_EGB.to_csv(f'./{folder_name}/EGB.csv', index=False)
        data_PAM.to_csv(f'./{folder_name}/PAM.csv', index=False)
        data_Zscales.to_csv(f'./{folder_name}/Zscales.csv', index=False)

    else:
        labels = np.array(pd.read_csv(f'./{folder_name}/labels.csv'))
        data_one_hot = pd.read_csv(f'./{folder_name}/one_hot.csv')
        data_blosum = pd.read_csv(f'./{folder_name}/blosum.csv')
        data_AAindex = pd.read_csv(f'./{folder_name}/AAindex.csv')
        data_SD = pd.read_csv(f'./{folder_name}/SD.csv')
        data_PC = pd.read_csv(f'./{folder_name}/PC.csv')
        data_EGB = pd.read_csv(f'./{folder_name}/EGB.csv')
        data_PAM = pd.read_csv(f'./{folder_name}/PAM.csv')
        data_Zscales = pd.read_csv(f'./{folder_name}/Zscales.csv')

        PG_data = pd.read_csv(f'./{folder_name}/PG_data.csv')
        CL_data = pd.read_csv(f'./{folder_name}/CL_data.csv')
        CC_data = pd.read_csv(f'./{folder_name}/CC_data.csv')
        BC_data = pd.read_csv(f'./{folder_name}/BC_data.csv')
        EC_data = pd.read_csv(f'./{folder_name}/EC_data.csv')
        DC_data = pd.read_csv(f'./{folder_name}/DC_data.csv')

        EJS = np.load(f'./{folder_name}/EJS.npy')
        EBC = np.load(f'./{folder_name}/EBC.npy')
        ED = np.load(f'./{folder_name}/ED.npy')


    return labels, data_one_hot, data_blosum, data_AAindex, data_SD, data_PC, data_EGB, data_PAM, data_Zscales, PG_data, CL_data, CC_data, BC_data, EC_data, DC_data, EJS, EBC, ED