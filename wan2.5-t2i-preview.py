import pandas as pd
import requests
import os
import time
import base64
import json
from io import BytesIO
from PIL import Image
from tqdm import tqdm

class SimpleImageGenerator:
    def __init__(self, api_key, api_url=None, output_dir=None, model="wan2.5-t2i-preview", size="1024*1024"):
        """
        初始化图片生成器
        """
        self.api_key = api_key
        self.model = model
        self.size = size
        
        # API地址 - 文生图异步接口
        self.api_url = api_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
        
        # 任务查询API地址
        self.task_api_url = "https://dashscope.aliyuncs.com/api/v1/tasks"
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = "wan2.5-t2i-preview"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 请求头
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable"  # 启用异步调用
        }
    
    def read_texts_from_excel(self, excel_path, column_index=0, sheet_name=0, has_header=False):
        """
        从Excel读取单列文本
        """
        try:
            print(f"正在读取Excel文件: {excel_path}")
            
            if has_header:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            
            texts = []
            
            # 确定起始行号
            start_row = 2 if has_header else 1
            
            for idx, row in df.iterrows():
                row_num = idx + start_row  # Excel行号（从1开始）
                
                # 确保列索引不越界
                if column_index < len(row):
                    text = str(row.iloc[column_index]).strip()
                    
                    # 跳过空文本
                    if text and text.lower() != 'nan' and text != 'None':
                        texts.append((row_num, text))
                        print(f"读取到第 {row_num} 行: {text[:30]}...")
            
            print(f"从Excel读取到 {len(texts)} 个有效文本")
            return texts
            
        except Exception as e:
            print(f"读取Excel文件失败: {str(e)}")
            raise
    
    def create_async_task(self, prompt, n=1):
        """
        创建异步任务
        """
        # 准备请求数据 - 根据新的官方文档格式
        data = {
            "model": self.model,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "size": self.size,
                "n": n
            }
        }
        
        try:
            print(f"创建异步任务...")
            print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"任务创建成功，响应: {result}")
                
                # 检查是否包含任务ID
                if "output" in result and "task_id" in result["output"]:
                    task_id = result["output"]["task_id"]
                    return task_id
                else:
                    print(f"响应中未找到任务ID: {result}")
                    return None
            else:
                print(f"任务创建失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"创建异步任务异常: {str(e)}")
            return None
    
    def get_task_result(self, task_id, max_retries=30, retry_interval=30):
        """
        获取异步任务结果
        """
        task_url = f"{self.task_api_url}/{task_id}"
        
        for attempt in range(max_retries):
            try:
                print(f"查询任务状态 ({attempt+1}/{max_retries})...")
                
                response = requests.get(
                    task_url,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 保存调试信息
                    debug_file = os.path.join(self.output_dir, f"task_debug_{task_id}_{attempt}.json")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    # 检查任务状态
                    if "output" in result:
                        task_status = result["output"].get("task_status", "UNKNOWN")
                        print(f"任务状态: {task_status}")
                        
                        if task_status == "SUCCEEDED":
                            print(f"✓ 任务成功完成")
                            return result
                        elif task_status == "FAILED":
                            print(f"✗ 任务失败")
                            if "message" in result["output"]:
                                print(f"失败原因: {result['output']['message']}")
                            return None
                        elif task_status == "RUNNING":
                            print(f"任务正在运行，等待 {retry_interval} 秒后重试...")
                            time.sleep(retry_interval)
                        else:
                            print(f"未知任务状态: {task_status}")
                            time.sleep(retry_interval)
                    else:
                        print(f"响应中未找到output字段")
                        time.sleep(retry_interval)
                else:
                    print(f"查询任务状态失败: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    time.sleep(retry_interval)
                    
            except Exception as e:
                print(f"查询任务状态异常: {str(e)}")
                time.sleep(retry_interval)
        
        print(f"✗ 任务查询超时，已达到最大重试次数: {max_retries}")
        return None
    
    def extract_image_from_result(self, result):
        """
        从任务结果中提取图片数据
        """
        try:
            # 尝试不同的响应格式
            image_data = None
            image_url = None
            
            if "output" in result:
                output = result["output"]
                
                # 检查是否有results字段
                if "results" in output and output["results"]:
                    results = output["results"]
                    if len(results) > 0:
                        first_result = results[0]
                        
                        # 检查是否有url字段
                        if "url" in first_result:
                            image_url = first_result["url"]
                            print(f"✓ 获取到图片URL: {image_url[:100]}...")
                        # 检查是否有image字段（base64编码）
                        elif "image" in first_result:
                            try:
                                image_data = base64.b64decode(first_result["image"])
                                print("✓ 获取到base64编码的图片数据")
                            except:
                                print("✗ base64解码失败")
                
                # 如果没有results，检查是否有其他字段
                elif "image" in output:
                    try:
                        image_data = base64.b64decode(output["image"])
                        print("✓ 从output.image获取base64图片")
                    except:
                        print("✗ output.image不是有效的base64数据")
            
            # 如果有图片URL，下载图片
            if image_url:
                try:
                    print(f"正在下载图片...")
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        image_data = img_response.content
                        print("✓ 图片下载成功")
                    else:
                        print(f"✗ 图片下载失败: {img_response.status_code}")
                except Exception as e:
                    print(f"✗ 图片下载异常: {str(e)}")
            
            return image_data
            
        except Exception as e:
            print(f"提取图片数据异常: {str(e)}")
            return None
    
    def generate_image(self, prompt, row_num=None, n=1):
        """
        生成单张图片（异步方式）
        row_num: 可选参数，如果提供会用于文件名
        """
        # 如果没有提供行号，使用时间戳作为标识
        if row_num is None:
            row_num = int(time.time())
            print(f"测试模式：使用时间戳 {row_num} 作为标识")
        
        print(f"开始生成第 {row_num} 行图片...")
        print(f"模型: {self.model}")
        print(f"图片尺寸: {self.size}")
        print(f"生成数量: {n}")
        
        # 1. 创建异步任务
        task_id = self.create_async_task(prompt, n)
        
        if not task_id:
            print(f"✗ 第 {row_num} 行创建异步任务失败")
            return None
        
        print(f"✓ 异步任务创建成功，任务ID: {task_id}")
        
        # 2. 查询任务结果
        task_result = self.get_task_result(task_id)
        
        if not task_result:
            print(f"✗ 第 {row_num} 行任务执行失败")
            return None
        
        # 3. 提取图片数据
        image_data = self.extract_image_from_result(task_result)
        
        if not image_data:
            print(f"✗ 第 {row_num} 行无法提取图片数据")
            return None
        
        # 4. 保存图片
        filename = f"row_{row_num:04d}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            image = Image.open(BytesIO(image_data))
            image.save(filepath, "PNG")
            print(f"✓ 第 {row_num} 行图片保存成功: {filename}")
            print(f"  保存路径: {filepath}")
            print(f"  图片尺寸: {image.size}")
            return filepath
        except Exception as e:
            print(f"✗ 第 {row_num} 行图片保存失败: {str(e)}")
            return None
    
    def batch_generate(self, texts, delay=2.0, n=1):
        """
        批量生成图片
        """
        success_rows = []
        failed_rows = []
        
        print(f"开始批量生成，共 {len(texts)} 个文本")
        print(f"参数配置:")
        print(f"  模型: {self.model}")
        print(f"  图片尺寸: {self.size}")
        print(f"  生成数量: {n}")
        print(f"  请求间隔: {delay}秒")
        
        for row_num, text in tqdm(texts, desc="生成进度"):
            result = self.generate_image(text, row_num, n=n)
            
            if result:
                success_rows.append(row_num)
            else:
                failed_rows.append(row_num)
            
            # 控制请求频率
            if delay > 0 and len(texts) > 1:
                time.sleep(delay)
        
        return success_rows, failed_rows
    
    def generate_from_excel(self, excel_path, column_index=0, has_header=False, delay=5.0, n=1):
        """
        从Excel文件批量生成图片
        """
        # 读取文本
        texts = self.read_texts_from_excel(excel_path, column_index, has_header=has_header)
        
        if not texts:
            print("未找到有效文本，请检查Excel文件")
            return
        
        # 批量生成
        success, failed = self.batch_generate(
            texts, 
            delay,
            n=n
        )
        
        # 生成报告
        self._generate_report(success, failed, excel_path)
        
        return {
            "total": len(texts),
            "success": len(success),
            "failed": len(failed),
            "success_rows": success,
            "failed_rows": failed
        }
    
    def _generate_report(self, success_rows, failed_rows, excel_path):
        """生成简单的报告"""
        report_path = os.path.join(self.output_dir, "生成报告.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("wan2.5-t2i-preview API 批量图片生成报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"源文件: {excel_path}\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {self.output_dir}\n")
            f.write(f"模型: {self.model}\n")
            f.write(f"图片尺寸: {self.size}\n")
            f.write(f"生成方式: 异步调用\n")
            f.write("-" * 50 + "\n")
            f.write(f"总任务数: {len(success_rows) + len(failed_rows)}\n")
            f.write(f"成功: {len(success_rows)} 个\n")
            f.write(f"失败: {len(failed_rows)} 个\n")
            f.write("-" * 50 + "\n")
            
            if success_rows:
                f.write("成功生成的行号:\n")
                for row in success_rows:
                    f.write(f"  行 {row}: row_{row:04d}.png\n")
            
            if failed_rows:
                f.write("失败的行号:\n")
                for row in failed_rows:
                    f.write(f"  行 {row}\n")
        
        print(f"生成报告已保存: {report_path}")


# 直接运行版本（无需命令行参数）
def main():
    """主函数 - 直接运行版本"""
    
    # ==================== 请修改以下配置 ====================
    
    # 1. 您的API密钥（必填）
    API_KEY = "sk-7f542cb20b1c485d9ca4bc5c645b2898"  # 请替换为您的真实API密钥
    
    # 2. API地址（必填）
    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    
    # 3. Excel文件路径（必填）
    EXCEL_FILE = r"C:\item2\666.xlsx"
    
    # 4. 输出目录（可选）
    OUTPUT_DIR = r"C:\item2\wan2.5-t2i-preview"
    
    # 5. 模型配置
    MODEL = "wan2.5-t2i-preview"  # 使用官方文档指定的模型
    SIZE = "1024*1024"         # 图片尺寸
    
    # 6. Excel配置
    COLUMN_INDEX = 0           # 文本在第几列（0=第一列，1=第二列...）
    HAS_HEADER = True          # Excel是否有标题行
    DELAY = 5.0                # 请求间隔（秒），异步任务需要更长时间
    
    # 7. 生成参数
    N = 1                      # 生成数量
    
    # ==================== 配置结束 ====================
    
    print("开始批量图片生成（异步模式）...")
    print(f"Excel文件: {EXCEL_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型: {MODEL}")
    print(f"图片尺寸: {SIZE}")
    print(f"生成数量: {N}")
    print(f"注意: 使用异步模式，每张图片生成可能需要更长时间")
    
    # 检查文件是否存在
    if not os.path.exists(EXCEL_FILE):
        print(f"错误: Excel文件不存在 - {EXCEL_FILE}")
        print("请检查文件路径是否正确")
        return
    
    # 创建生成器
    generator = SimpleImageGenerator(
        api_key=API_KEY,
        api_url=API_URL,
        output_dir=OUTPUT_DIR,
        model=MODEL,
        size=SIZE
    )
    
    # 批量生成
    result = generator.generate_from_excel(
        excel_path=EXCEL_FILE,
        column_index=COLUMN_INDEX,
        has_header=HAS_HEADER,
        delay=DELAY,
        n=N
    )
    
    if result:
        print("\n" + "=" * 50)
        print("批量生成完成！")
        print(f"总任务数: {result['total']}")
        print(f"成功: {result['success']} 个")
        print(f"失败: {result['failed']} 个")
        
        if result['failed'] > 0:
            print(f"失败行号: {result['failed_rows']}")
        
        print(f"图片保存在: {os.path.abspath(OUTPUT_DIR)}")
        print("=" * 50)


# 测试模式（不实际调用API，只测试文件读取）
def test_mode():
    """测试模式 - 只测试文件读取，不调用API"""
    
    print("=== 测试模式 ===")
    print("此模式只测试Excel文件读取，不调用API生成图片")
    
    EXCEL_FILE = r"C:\item2\文言文11.xlsx"
    OUTPUT_DIR = r"C:\item2\test_output"
    
    # 创建测试生成器（不需要API密钥）
    class TestGenerator:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        
        def read_texts_from_excel(self, excel_path, column_index=0, has_header=False):
            try:
                print(f"测试读取: {excel_path}")
                
                if has_header:
                    df = pd.read_excel(excel_path)
                else:
                    df = pd.read_excel(excel_path, header=None)
                
                texts = []
                start_row = 2 if has_header else 1
                
                for idx, row in df.iterrows():
                    row_num = idx + start_row
                    if column_index < len(row):
                        text = str(row.iloc[column_index]).strip()
                        if text and text.lower() != 'nan' and text != 'None':
                            texts.append((row_num, text))
                            print(f"第 {row_num} 行: {text}")
                
                print(f"共读取到 {len(texts)} 个文本")
                return texts
            except Exception as e:
                print(f"读取失败: {e}")
                return []
    
    # 测试
    test_gen = TestGenerator(OUTPUT_DIR)
    texts = test_gen.read_texts_from_excel(EXCEL_FILE, column_index=0, has_header=False)
    
    if texts:
        print("\n测试成功！Excel文件读取正常。")
        print("接下来，请确保：")
        print("1. 已获取有效的API密钥")
        print("2. 已获得正确的API地址")
        print("3. 将配置填入main()函数中")
    else:
        print("\n测试失败，请检查Excel文件路径和格式。")


def test_single_image():
    """测试单张图片生成功能"""
    print("=== 测试单张图片生成 ===")
    print("此模式用于测试API是否正常工作，仅生成一张图片")
    
    # ==================== 配置 ====================
    # 使用与main函数相同的配置
    API_KEY = "sk-119949c0056240c2beb6858765ca947d"  # 请替换为您的真实API密钥
    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
    OUTPUT_DIR = r"C:\item2\wan2.5-t2i-preview"
    MODEL = "wan2.5-t2i-preview"
    SIZE = "1024*1024"
    
    # 创建测试输出目录
    test_output_dir = os.path.join(OUTPUT_DIR, "test_single")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 创建生成器
    generator = SimpleImageGenerator(
        api_key=API_KEY,
        api_url=API_URL,
        output_dir=test_output_dir,
        model=MODEL,
        size=SIZE
    )
    
    # 选择提示词来源
    print("\n请选择提示词来源:")
    print("1. 手动输入提示词")
    print("2. 从Excel文件读取第一行")
    print("3. 使用官方文档示例提示词")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    prompt = ""
    
    if choice == "1":
        # 手动输入提示词
        prompt = input("请输入测试用的提示词: ").strip()
        if not prompt:
            print("提示词不能为空，使用默认提示词")
            prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
    
    elif choice == "2":
        # 从Excel文件读取
        excel_file = r"C:\item2\文言文11.xlsx"
        
        if not os.path.exists(excel_file):
            print(f"Excel文件不存在: {excel_file}")
            print("请手动输入提示词:")
            prompt = input("提示词: ").strip()
        else:
            try:
                # 读取Excel第一行
                df = pd.read_excel(excel_file, header=None)
                if len(df) > 0 and len(df.columns) > 0:
                    prompt = str(df.iloc[0, 0]).strip()
                    print(f"从Excel读取到提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                else:
                    print("Excel文件为空或格式不正确")
                    prompt = input("请手动输入提示词: ").strip()
            except Exception as e:
                print(f"读取Excel失败: {e}")
                prompt = input("请手动输入提示词: ").strip()
    
    elif choice == "3":
        # 使用官方文档示例提示词
        prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
        print(f"使用官方文档示例提示词: {prompt}")
    
    else:
        print("无效选择，使用默认提示词")
        prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
    
    if not prompt:
        prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
    
    print(f"\n开始生成单张测试图片...")
    print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"模型: {MODEL}")
    print(f"图片尺寸: {SIZE}")
    print(f"输出目录: {test_output_dir}")
    print(f"生成方式: 异步调用")
    
    # 生成参数配置
    print("\n请配置生成参数:")
    
    n_input = input("生成数量 (默认1): ").strip()
    try:
        n = int(n_input) if n_input else 1
    except:
        print("输入无效，使用默认值1")
        n = 1
    
    # 生成单张图片
    print("\n正在调用API生成图片（异步模式）...")
    print(f"注意: 异步生成可能需要更长时间，请耐心等待")
    print(f"请求参数:")
    print(f"  生成数量: {n}")
    
    start_time = time.time()
    
    result = generator.generate_image(
        prompt, 
        row_num="test_single",
        n=n
    )
    
    end_time = time.time()
    
    if result:
        print(f"\n✓ 测试成功！")
        print(f"图片生成总耗时: {end_time - start_time:.2f} 秒")
        print(f"图片保存位置: {result}")
        
        # 显示文件信息
        if os.path.exists(result):
            file_size = os.path.getsize(result) / 1024  # KB
            print(f"文件大小: {file_size:.2f} KB")
            
            try:
                img = Image.open(result)
                print(f"图片尺寸: {img.size}")
                
                # 显示图片（可选）
                show_img = input("\n是否显示图片? (y/n, 默认n): ").strip().lower()
                if show_img in ['y', 'yes']:
                    img.show()
            except:
                pass
    else:
        print(f"\n✗ 测试失败！")
        print("请检查:")
        print("1. API密钥是否正确")
        print("2. API地址是否正确")
        print("3. 网络连接是否正常")
        print("4. 账户是否有足够的额度")
        print("5. 查看task_debug_*.json文件了解详细错误信息")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("wan2.5-t2i-preview API 批量图片生成器")
    print("=" * 60)
    
    # 选择模式
    choice = input("请选择模式:\n1. 完整模式（批量生成图片）\n2. 测试模式（只测试文件读取）\n3. 测试单张图片生成\n请输入 1, 2 或 3: ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_mode()
    elif choice == "3":
        test_single_image()
    else:
        print("无效选择，退出程序。")