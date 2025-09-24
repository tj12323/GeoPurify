import pandas as pd
import numpy as np

MATTERPORT_CLASS_REMAP = np.zeros(41)
MATTERPORT_CLASS_REMAP[1] = 1
MATTERPORT_CLASS_REMAP[2] = 2
MATTERPORT_CLASS_REMAP[3] = 3
MATTERPORT_CLASS_REMAP[4] = 4
MATTERPORT_CLASS_REMAP[5] = 5
MATTERPORT_CLASS_REMAP[6] = 6
MATTERPORT_CLASS_REMAP[7] = 7
MATTERPORT_CLASS_REMAP[8] = 8
MATTERPORT_CLASS_REMAP[9] = 9
MATTERPORT_CLASS_REMAP[10] = 10
MATTERPORT_CLASS_REMAP[11] = 11
MATTERPORT_CLASS_REMAP[12] = 12
MATTERPORT_CLASS_REMAP[14] = 13
MATTERPORT_CLASS_REMAP[16] = 14
MATTERPORT_CLASS_REMAP[22] = 21  # ceiling
MATTERPORT_CLASS_REMAP[24] = 15
MATTERPORT_CLASS_REMAP[28] = 16
MATTERPORT_CLASS_REMAP[33] = 17
MATTERPORT_CLASS_REMAP[34] = 18
MATTERPORT_CLASS_REMAP[36] = 19
MATTERPORT_CLASS_REMAP[39] = 20

MATTERPORT_ALLOWED_NYU_CLASSES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 22, 24, 28, 33, 34, 36, 39]

def get_final_label_mapping(tsv_file_path):
    try:
        category_mapping = pd.read_csv(tsv_file_path, sep='\t', header=0)
        nyu40_to_name = {}
        for _, row in category_mapping.iterrows():
            if pd.notna(row['nyu40id']):
                nyu40_to_name[int(row['nyu40id'])] = row.get('nyu40class', 'unknown')

        print("Final label mapping (after processing with vertex_labels -= 1):")
        print("=" * 50)

        final_mapping = {}
        for nyu40id in MATTERPORT_ALLOWED_NYU_CLASSES:
            if nyu40id in nyu40_to_name:
                remapped_value = int(MATTERPORT_CLASS_REMAP[nyu40id])
                final_label = remapped_value - 1

                final_mapping[final_label] = {
                    'name': nyu40_to_name[nyu40id],
                    'nyu40id': nyu40id,
                    'remapped_value': remapped_value
                }

        for final_label in sorted(final_mapping.keys()):
            info = final_mapping[final_label]
            print(f"Label {final_label:2d}: {info['name']:<15} (NYU40ID: {info['nyu40id']}, Remapped value: {info['remapped_value']})")

        print(f"Label 255: background/unknown")
        print("=" * 50)

        return final_mapping

    except FileNotFoundError:
        print(f"Error: File {tsv_file_path} not found.")
        print("Please provide the correct path to the category_mapping.tsv file.")
        return None
    except Exception as e:
        print(f"Error occurred while processing the file: {e}")
        return None

if __name__ == "__main__":
    tsv_path = '/path/category_mapping.tsv'
    mapping = get_final_label_mapping(tsv_path)
# import torch
# import pandas as pd
# import numpy as np

# def analyze_labels_mapping(tsv_file, num_classes=160):
#     category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)
#     label_name = []
#     label_id = []
#     label_all = category_mapping['nyuClass'].tolist()
#     eliminated_list = ['void', 'unknown']
#     mapping = np.zeros(len(label_all)+1, dtype=int)
#     instance_count = category_mapping['count'].tolist()
#     ins_count_list = []
#     counter = 1
#     flag_stop = False

#     for i, x in enumerate(label_all):
#         if not flag_stop and isinstance(x, str) and x not in label_name and x not in eliminated_list:
#             label_name.append(x)
#             label_id.append(counter)
#             mapping[i+1] = counter
#             counter += 1
#             ins_count_list.append(instance_count[i])
#             if counter == num_classes+1:
#                 flag_stop = True
#         elif isinstance(x, str) and x in label_name:
#             mapping[i+1] = label_name.index(x)+1

#     final_label_dict = {}
#     for i, name in enumerate(label_name):
#         final_label_dict[i] = name

#     final_label_dict[255] = 'unlabeled/background'
#     return final_label_dict, mapping

# def load_and_check_pth_labels(pth_file_path):
#     coords, colors, normal, vertex_labels = torch.load(pth_file_path)

#     print(f"Number of vertices: {coords.shape[0]}")
#     print(f"Unique value of the tag: {np.unique(vertex_labels)}")
#     print(f"Tag Value Statistics:")

#     unique_labels, counts = np.unique(vertex_labels, return_counts=True)
#     for label, count in zip(unique_labels, counts):
#         print(f"  Label {label}: {count} vertices")

#     return vertex_labels
# if __name__ == "__main__":

#     tsv_file = '/path/category_mapping.tsv'
#     num_classes = 160

#     label_dict, mapping_array = analyze_labels_mapping(tsv_file, num_classes)

#     print("=== The meaning of tags in the final PTH file ===")
#     print("Digital Label -> Actual Category Name:")
#     for label_num, class_name in sorted(label_dict.items()):
#         print(f"  {label_num:3d} -> {class_name}")

#     print(f"\nThere are a total of {len(label_dict)-1} valid categories + 1 background category.")