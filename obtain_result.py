import os
import subprocess
import re
from tqdm import tqdm

def run_aout_in_subdirectories(root_dir):
    
    result_tmp = []
    
    """遍历根目录下的所有子目录，并执行其中的a.out文件"""
    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir)):
        # 检查当前目录中是否有a.out文件
        if 'a.out' in filenames:
            aout_relative_path = os.path.join(dirpath, 'a.out')
            aout_absolute_path = os.path.abspath(aout_relative_path)
            
            # print(f"在目录 {dirpath} 中找到 a.out 文件")
            # print(f"绝对路径: {aout_absolute_path}")
            
            # 详细检查文件状态
            # try:
            #     file_stat = os.stat(aout_absolute_path)
            #     # print(f"文件状态: 大小={file_stat.st_size} 字节, 权限={oct(file_stat.st_mode & 0o777)}")
                
            #     # 检查是否为符号链接
            #     if os.path.islink(aout_absolute_path):
            #         link_target = os.readlink(aout_absolute_path)
            #         print(f"注意: {aout_absolute_path} 是一个符号链接，指向 {link_target}")
                    
            #         # 检查符号链接是否有效
            #         if not os.path.exists(aout_absolute_path):
            #             print(f"错误: 符号链接 {aout_absolute_path} 指向的目标不存在")
            #             continue
            # except Exception as e:
            #     print(f"无法获取文件状态: {aout_absolute_path}, 错误: {e}")
            #     continue
            
            try:
                # 使用绝对路径运行a.out并捕获输出
                result = subprocess.run(
                    [aout_absolute_path],
                    cwd=dirpath,
                    capture_output=True,
                    text=True,
                    timeout=300  # 设置超时时间为5分钟
                )
                
                # 打印执行结果
                # print(f"执行成功，返回代码: {result.returncode}")
                if result.stdout:
                    # print("标准输出:")
                    # print(result.stdout)
                    numbers = re.findall(r'-?\d+\.\d+|-?\d+', result.stdout)
                    result_tmp.append(float(numbers[-1]))
                    

                # if result.stderr:
                    # print("标准错误:")
                    # print(result.stderr)
                    
                    
            except subprocess.TimeoutExpired:
                print(f"执行超时: {aout_absolute_path}")
            except Exception as e:
                print(f"执行出错: {aout_absolute_path}, 错误: {e}")
                
    print('gflops:', max(result_tmp))

if __name__ == "__main__":
    # 设置根目录为data_source
    root_directory = "./data_source"
    
    # 检查目录是否存在
    if not os.path.exists(root_directory):
        print(f"wrong:  {root_directory} not exsit")
    elif not os.path.isdir(root_directory):
        print(f"wrong: {root_directory} no a directory")
    else:
        print(f"start to execute a.out file on data_source...")
        run_aout_in_subdirectories(root_directory)
        print("finish processing")