import pandas as pd
import requests
import os
import time
import base64
from io import BytesIO
from PIL import Image
import argparse
from tqdm import tqdm

class SimpleImageGenerator:
    def __init__(self, api_key, api_url=None, output_dir=None):
        """
        初始化图片生成器
        """
        self.api_key = api_key
        
        # 请替换为实际的SeedDream 4.0 API地址
        # 注意：这里需要您提供正确的SeedDream API地址
        self.api_url = api_url or "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = "seedream4.5"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 请求头
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
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
    
    def generate_image(self, prompt, row_num, retry_count=3):
        """
        根据文本生成单张图片
        """
        # 准备请求数据 - 根据SeedDream API文档调整
        data = {
            "model": "ep-20251208104935-ccblz",  # 模型名称
            "prompt": prompt,
            "size": "2k",       # 图片尺寸
            "num_images": 1,           # 生成数量
            "steps": 30,               # 生成步数
            "guidance_scale": 7.5,     # 指导强度
            "negative_prompt": "低质量, 模糊, 失真"  # 负面提示词
        }
        
        for attempt in range(retry_count):
            try:
                print(f"正在生成第 {row_num} 行图片...")
                
                # 打印调试信息
                print(f"API地址: {self.api_url}")
                print(f"提示词: {prompt[:50]}...")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=60
                )
                
                print(f"响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"API响应: {result}")
                    
                    # 处理API响应
                    image_data = None
                    
                    # 调试：打印响应结构
                    print(f"响应数据结构: {result.keys() if isinstance(result, dict) else '非字典'}")
                    
                    # 情况1: 返回base64编码
                    if isinstance(result, dict) and "images" in result and result["images"]:
                        image_data = base64.b64decode(result["images"][0])
                    elif isinstance(result, dict) and "data" in result and result["data"]:
                        if "b64_json" in result["data"][0]:
                            image_data = base64.b64decode(result["data"][0]["b64_json"])
                        elif "url" in result["data"][0]:
                            img_response = requests.get(result["data"][0]["url"], timeout=30)
                            image_data = img_response.content
                    
                    if image_data:
                        # 保存图片
                        filename = f"row_{row_num:04d}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # 确保是有效的图片数据
                        try:
                            image = Image.open(BytesIO(image_data))
                            image.save(filepath, "PNG")
                            print(f"✓ 第 {row_num} 行图片保存成功: {filename}")
                            return filepath
                        except Exception as img_error:
                            print(f"✗ 第 {row_num} 行图片数据无效: {img_error}")
                    else:
                        print(f"✗ 第 {row_num} 行无法获取图片数据")
                        print(f"完整响应: {result}")
                    
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 10
                    print(f"⚠ 达到频率限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"✗ 第 {row_num} 行API请求失败: {response.status_code}")
                    print(f"错误信息: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"⚠ 第 {row_num} 行请求超时，重试 {attempt+1}/{retry_count}")
            except Exception as e:
                print(f"✗ 第 {row_num} 行生成失败: {str(e)}")
            
            # 重试前等待
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)
        
        print(f"✗ 第 {row_num} 行生成失败（超过最大重试次数）")
        return None
    
    def batch_generate(self, texts, delay=2.0):
        """
        批量生成图片
        """
        success_rows = []
        failed_rows = []
        
        print(f"开始批量生成，共 {len(texts)} 个文本")
        
        for row_num, text in tqdm(texts, desc="生成进度"):
            result = self.generate_image(text, row_num)
            
            if result:
                success_rows.append(row_num)
            else:
                failed_rows.append(row_num)
            
            # 控制请求频率
            if delay > 0 and len(texts) > 1:
                time.sleep(delay)
        
        return success_rows, failed_rows
    
    def generate_from_excel(self, excel_path, column_index=0, has_header=False, delay=2.0):
        """
        从Excel文件批量生成图片
        """
        # 读取文本
        texts = self.read_texts_from_excel(excel_path, column_index, has_header=has_header)
        
        if not texts:
            print("未找到有效文本，请检查Excel文件")
            return
        
        # 批量生成
        success, failed = self.batch_generate(texts, delay)
        
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
            f.write("SeedDream 3.0 批量图片生成报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"源文件: {excel_path}\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输出目录: {self.output_dir}\n")
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
    
    # 1. 您的SeedDream API密钥（必填）
    API_KEY = "954585ce-0056-45f0-be23-889f869b96ed"  # 请替换为您的真实API密钥
    
    # 2. SeedDream API地址（必填）
    # 请咨询SeedDream官方文档获取正确的API地址
    API_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"  # 示例地址，请替换
    
    # 3. Excel文件路径（必填）
    # 注意：Windows路径需要使用双反斜杠或原始字符串
    EXCEL_FILE = r"C:\item2\原文1.xlsx"  # 使用原始字符串
    # 或者 EXCEL_FILE = "C:\\item2\\单文言文.xlsx"
    
    # 4. 输出目录（可选）
    OUTPUT_DIR = r"C:\item2\seedream4.5"  # 图片将保存在这里
    
    # 5. Excel配置
    COLUMN_INDEX = 0     # 文本在第几列（0=第一列，1=第二列...）
    HAS_HEADER = True   # Excel是否有标题行
    DELAY = 2.0          # 请求间隔（秒），避免API限制
    
    # ==================== 配置结束 ====================
    
    print("开始批量图片生成...")
    print(f"Excel文件: {EXCEL_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 检查文件是否存在
    if not os.path.exists(EXCEL_FILE):
        print(f"错误: Excel文件不存在 - {EXCEL_FILE}")
        print("请检查文件路径是否正确")
        return
    
    # 创建生成器
    generator = SimpleImageGenerator(
        api_key=API_KEY,
        api_url=API_URL,
        output_dir=OUTPUT_DIR
    )
    
    # 批量生成
    result = generator.generate_from_excel(
        excel_path=EXCEL_FILE,
        column_index=COLUMN_INDEX,
        has_header=HAS_HEADER,
        delay=DELAY
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
    
    EXCEL_FILE = r"C:\item2\原文1.xlsx"
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
        print("1. 已获取有效的SeedDream API密钥")
        print("2. 已获得正确的API地址")
        print("3. 将配置填入main()函数中")
    else:
        print("\n测试失败，请检查Excel文件路径和格式。")


if __name__ == "__main__":
    print("=" * 60)
    print("SeedDream 3.0 批量图片生成器")
    print("=" * 60)
    
    # 选择模式
    choice = input("请选择模式:\n1. 完整模式（生成图片）\n2. 测试模式（只测试文件读取）\n请输入 1 或 2: ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_mode()
    else:
        print("无效选择，退出程序。")