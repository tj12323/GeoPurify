import pandas as pd
import numpy as np

# 从原代码中提取的映射关系
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
    """
    获取最终的0-20编号对应的实际物体类别名称
    
    Args:
        tsv_file_path: category_mapping.tsv文件的路径
    """
    try:
        # 读取类别映射文件
        category_mapping = pd.read_csv(tsv_file_path, sep='\t', header=0)
        
        # 创建NYU40ID到类别名称的映射
        nyu40_to_name = {}
        for _, row in category_mapping.iterrows():
            if pd.notna(row['nyu40id']):
                nyu40_to_name[int(row['nyu40id'])] = row.get('nyu40class', 'unknown')
        
        print("最终标签映射 (经过vertex_labels -= 1处理后):")
        print("=" * 50)
        
        # 创建最终映射
        final_mapping = {}
        
        for nyu40id in MATTERPORT_ALLOWED_NYU_CLASSES:
            if nyu40id in nyu40_to_name:
                # 获取重映射后的值，然后减1得到最终标签
                remapped_value = int(MATTERPORT_CLASS_REMAP[nyu40id])
                final_label = remapped_value - 1
                
                final_mapping[final_label] = {
                    'name': nyu40_to_name[nyu40id],
                    'nyu40id': nyu40id,
                    'remapped_value': remapped_value
                }
        
        # 按最终标签排序并输出
        for final_label in sorted(final_mapping.keys()):
            info = final_mapping[final_label]
            print(f"标签 {final_label:2d}: {info['name']:<15} (NYU40ID: {info['nyu40id']}, 重映射值: {info['remapped_value']})")
        
        print(f"标签 255: background/unknown")
        print("=" * 50)
        
        return final_mapping
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {tsv_file_path}")
        print("请提供正确的category_mapping.tsv文件路径")
        return None
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 请替换为你的实际TSV文件路径
    tsv_path = '/root/code/CUA_O3D/dataset/matterport/category_mapping.tsv'
    
    # 尝试读取TSV文件
    mapping = get_final_label_mapping(tsv_path)
# import torch
# import pandas as pd
# import numpy as np

# def analyze_labels_mapping(tsv_file, num_classes=160):
#     """
#     分析标签映射关系，显示最终pth文件中数字对应的实际标签含义
#     """
#     # 读取原始类别映射文件
#     category_mapping = pd.read_csv(tsv_file, sep='\t', header=0)
    
#     # 重现原代码中的映射逻辑
#     label_name = []
#     label_id = []
#     label_all = category_mapping['nyuClass'].tolist()
#     eliminated_list = ['void', 'unknown']
#     mapping = np.zeros(len(label_all)+1, dtype=int)
#     instance_count = category_mapping['count'].tolist()
#     ins_count_list = []
    
#     counter = 1
#     flag_stop = False
    
#     # 构建映射表
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
    
#     # 创建最终的标签字典
#     final_label_dict = {}
    
#     # 注意：原代码中有 vertex_labels -= 1 的操作
#     # 所以最终的标签值会减1
#     for i, name in enumerate(label_name):
#         final_label_dict[i] = name  # i 是最终pth中的数值
    
#     # 255 对应原来的0类别（经过减1操作后）
#     final_label_dict[255] = 'unlabeled/background'
    
#     return final_label_dict, mapping

# def load_and_check_pth_labels(pth_file_path):
#     """
#     加载pth文件并检查其中的标签分布
#     """
#     coords, colors, normal, vertex_labels = torch.load(pth_file_path)
    
#     print(f"顶点数量: {coords.shape[0]}")
#     print(f"标签的唯一值: {np.unique(vertex_labels)}")
#     print(f"标签值统计:")
    
#     unique_labels, counts = np.unique(vertex_labels, return_counts=True)
#     for label, count in zip(unique_labels, counts):
#         print(f"  标签 {label}: {count} 个顶点")
    
#     return vertex_labels

# # 使用示例
# if __name__ == "__main__":
#     # 设置文件路径
#     tsv_file = '/root/code/CUA_O3D/dataset/matterport/category_mapping.tsv'
#     num_classes = 160  # 根据你的设置修改
    
#     # 分析标签映射
#     label_dict, mapping_array = analyze_labels_mapping(tsv_file, num_classes)
    
#     print("=== 最终PTH文件中的标签含义 ===")
#     print("数字标签 -> 实际类别名称:")
#     for label_num, class_name in sorted(label_dict.items()):
#         print(f"  {label_num:3d} -> {class_name}")
    
#     print(f"\n总共有 {len(label_dict)-1} 个有效类别 + 1个背景类别")
    
#     # 如果你有具体的pth文件，可以加载查看
#     # pth_file = '/path/to/your/scene_region.pth'
#     # vertex_labels = load_and_check_pth_labels(pth_file)
