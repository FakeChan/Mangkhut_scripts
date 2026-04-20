import os

def is_number(s):
    """辅助函数：判断字符串是否可以转换为数字"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_unlimited_channels(input_file_path, output_file_path):
    """
    从RTTOV output文件中提取亮温数据。
    逻辑：不限制列数，动态读取标题后的所有数字行，直到遇到空行或下一个标题。
    """
    profiles_data = []
    
    try:
        with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 步骤1：定位关键标题
            if "CALCULATED BRIGHTNESS TEMPERATURES (K):" in line:
                current_profile_vals = []
                i += 1  # 移动到标题的下一行开始读取
                
                # 步骤2：动态读取后续数据行
                while i < len(lines):
                    data_line = lines[i].strip()
                    
                    # --- 停止条件 A: 空行 ---
                    if not data_line:
                        break
                    
                    parts = data_line.split()
                    
                    # --- 停止条件 B: 遇到非数字开头的行 (如下一个标题) ---
                    # 检查第一个部分是否为数字，如果不是数字（如 "CALCULATED"），则停止
                    if not parts or not is_number(parts[0]):
                        break
                    
                    # --- 收集数据 ---
                    # 将当前行的所有数据追加到当前廓线列表中
                    current_profile_vals.extend(parts)
                    i += 1
                
                # 如果收集到了数据，加入总列表
                if current_profile_vals:
                    profiles_data.append(current_profile_vals)
            else:
                # 如果不是标题行，继续往下找
                i += 1
                
        print(f"成功提取 {len(profiles_data)} 个廓线的数据。")
        if len(profiles_data) > 0:
            print(f"每个廓线包含 {len(profiles_data[0])} 个通道数据（以第一个廓线为例）。")
        
        # 步骤3：按照NR格式写入
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            for row in profiles_data:
                # 构造NR格式行：
                # 1. 缩进：行首3个空格 "   "
                # 2. 间隔：数据间2个空格 "  "
                # join会自动处理任意数量的通道
                line_str = "   " + "  ".join(row)
                
                f_out.write(line_str + "\n")
                
                # 写入间隔行：NR格式中数据行之间有一个包含空格的空行
                f_out.write("  \n")
                
        print(f"文件已生成：{output_file_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__=='__main__':
    # --- 执行配置 ---
    # 输入文件名
    BT_output_dir=os.getenv('BT_output_dir')
    BT_input=os.getenv('BT_input')
    BT_output=os.getenv('BT_output')
    obs_dir=os.getenv('BT_write_dir')
    instrument=os.getenv('instrument')
    
    input_filename = f'{BT_output_dir}/{BT_input}'
    # 输出文件名
    output_filename = f'{obs_dir}/{instrument}/{BT_output}'

    # 运行提取函数
    extract_unlimited_channels(input_filename, output_filename)