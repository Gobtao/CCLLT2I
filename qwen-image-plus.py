import pandas as pd
import requests
import os
import time
import base64
import json
import re
from io import BytesIO
from PIL import Image
from tqdm import tqdm

class SimpleImageGenerator:
    def __init__(self, api_key, api_url=None, output_dir=None, model="qwen-image-plus", size="1328*1328"):
        """
        初始化图片生成器
        """
        self.api_key = api_key
        self.model = model
        self.size = size
        
        # API地址
        self.api_url = api_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = "qwen-image-plus"
        
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
    
    def generate_image(self, prompt, row_num=None, retry_count=3, negative_prompt="", prompt_extend=True, watermark=False):
        """
        根据文本生成单张图片
        row_num: 可选参数，如果提供会用于文件名
        """
        # 准备请求数据 - 根据官方文档格式
        data = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "negative_prompt": negative_prompt,
                "prompt_extend": prompt_extend,
                "watermark": watermark,
                "size": self.size
            }
        }
        
        # 如果没有提供行号，使用时间戳作为标识
        if row_num is None:
            row_num = int(time.time())
            print(f"测试模式：使用时间戳 {row_num} 作为标识")
        
        for attempt in range(retry_count):
            try:
                print(f"正在生成第 {row_num} 行图片...")
                
                # 打印调试信息
                print(f"API地址: {self.api_url}")
                print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                print(f"模型: {self.model}")
                print(f"图片尺寸: {self.size}")
                
                # 打印请求数据（调试用）
                if attempt == 0:  # 只在第一次尝试时打印
                    print(f"请求数据预览: {json.dumps(data, ensure_ascii=False)[:200]}...")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=120  # 增加超时时间
                )
                
                print(f"响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"API响应结构: {list(result.keys()) if isinstance(result, dict) else '非字典'}")
                    
                    # 保存原始响应用于调试
                    debug_file = os.path.join(self.output_dir, f"debug_response_{row_num}.json")
                    with open(debug_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"原始响应已保存到: {debug_file}")
                    
                    # 处理API响应 - 根据实际返回结构提取图片URL或base64数据
                    image_data = None
                    image_url = None
                    
                    # 根据提供的debug_response_2.json结构，图片URL在output.choices[0].message.content[0].image
                    if isinstance(result, dict):
                        # 第一种可能：返回base64编码的图片
                        if "images" in result and result["images"]:
                            # 如果是base64编码的图片
                            if isinstance(result["images"][0], str):
                                try:
                                    image_data = base64.b64decode(result["images"][0])
                                    print("✓ 从base64数据获取图片")
                                except:
                                    print("✗ base64数据解码失败")
                        
                        # 第二种可能：返回图片URL（根据debug_response_2.json）
                        elif "output" in result:
                            output = result.get("output", {})
                            if isinstance(output, dict) and "choices" in output:
                                choices = output.get("choices", [])
                                if choices and len(choices) > 0:
                                    choice = choices[0]
                                    if isinstance(choice, dict) and "message" in choice:
                                        message = choice.get("message", {})
                                        if isinstance(message, dict) and "content" in message:
                                            contents = message.get("content", [])
                                            if contents and len(contents) > 0:
                                                content = contents[0]
                                                if isinstance(content, dict) and "image" in content:
                                                    image_url = content.get("image")
                                                    if image_url and image_url.startswith("http"):
                                                        print(f"✓ 获取到图片URL: {image_url[:100]}...")
                                                    else:
                                                        # 可能是base64数据
                                                        try:
                                                            image_data = base64.b64decode(content.get("image", ""))
                                                            print("✓ 从content.image获取base64图片")
                                                        except:
                                                            print("✗ content.image不是有效的base64数据")
                        
                        # 第三种可能：返回data字段（其他API格式）
                        elif "data" in result and result["data"]:
                            data_list = result.get("data", [])
                            if data_list and len(data_list) > 0:
                                item = data_list[0]
                                if "url" in item:
                                    image_url = item.get("url")
                                    print(f"✓ 从data.url获取图片URL: {image_url[:100]}...")
                                elif "b64_json" in item:
                                    try:
                                        image_data = base64.b64decode(item.get("b64_json", ""))
                                        print("✓ 从data.b64_json获取base64图片")
                                    except:
                                        print("✗ b64_json不是有效的base64数据")
                    
                    # 如果有图片URL，下载图片
                    if image_url:
                        try:
                            print(f"正在下载图片: {image_url[:100]}...")
                            img_response = requests.get(image_url, timeout=30)
                            if img_response.status_code == 200:
                                image_data = img_response.content
                                print("✓ 图片下载成功")
                            else:
                                print(f"✗ 图片下载失败: {img_response.status_code}")
                        except Exception as download_error:
                            print(f"✗ 图片下载失败: {download_error}")
                    
                    # 如果有图片数据，保存图片
                    if image_data:
                        # 保存图片
                        if row_num is None:
                            filename = f"test_{int(time.time())}.png"
                        else:
                            filename = f"row_{row_num:04d}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # 确保是有效的图片数据
                        try:
                            image = Image.open(BytesIO(image_data))
                            image.save(filepath, "PNG")
                            print(f"✓ 第 {row_num} 行图片保存成功: {filename}")
                            print(f"  保存路径: {filepath}")
                            print(f"  图片尺寸: {image.size}")
                            return filepath
                        except Exception as img_error:
                            print(f"✗ 第 {row_num} 行图片数据无效: {img_error}")
                    else:
                        print(f"✗ 第 {row_num} 行无法获取图片数据")
                        print(f"完整响应结构已保存到: {debug_file}")
                    
                elif response.status_code == 429:
                    wait_time = (attempt + 1) * 10
                    print(f"⚠ 达到频率限制，等待 {wait_time} 秒...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 401:
                    print(f"✗ API密钥无效或已过期")
                    break
                elif response.status_code == 400:
                    print(f"✗ 请求参数错误: {response.text[:200]}")
                    break
                else:
                    print(f"✗ 第 {row_num} 行API请求失败: {response.status_code}")
                    print(f"错误信息: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                print(f"⚠ 第 {row_num} 行请求超时，重试 {attempt+1}/{retry_count}")
            except json.JSONDecodeError as e:
                print(f"✗ 第 {row_num} 行响应JSON解析失败: {str(e)}")
                print(f"原始响应: {response.text[:500]}")
            except Exception as e:
                print(f"✗ 第 {row_num} 行生成失败: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 重试前等待
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        
        print(f"✗ 第 {row_num} 行生成失败（超过最大重试次数）")
        return None
    
    def batch_generate(self, texts, delay=2.0, negative_prompt="", prompt_extend=True, watermark=False):
        """
        批量生成图片
        """
        success_rows = []
        failed_rows = []
        
        print(f"开始批量生成，共 {len(texts)} 个文本")
        print(f"参数配置:")
        print(f"  模型: {self.model}")
        print(f"  图片尺寸: {self.size}")
        print(f"  负面提示词: {negative_prompt[:50]}{'...' if len(negative_prompt) > 50 else ''}")
        print(f"  提示词扩展: {prompt_extend}")
        print(f"  水印: {watermark}")
        
        for row_num, text in tqdm(texts, desc="生成进度"):
            result = self.generate_image(
                text, 
                row_num, 
                negative_prompt=negative_prompt,
                prompt_extend=prompt_extend,
                watermark=watermark
            )
            
            if result:
                success_rows.append(row_num)
            else:
                failed_rows.append(row_num)
            
            # 控制请求频率
            if delay > 0 and len(texts) > 1:
                time.sleep(delay)
        
        return success_rows, failed_rows
    
    def generate_from_excel(self, excel_path, column_index=0, has_header=False, delay=2.0, 
                           negative_prompt="", prompt_extend=True, watermark=False):
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
            negative_prompt=negative_prompt,
            prompt_extend=prompt_extend,
            watermark=watermark
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
    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    
    # 3. Excel文件路径（必填）
    EXCEL_FILE = r"C:\item2\文言文14.xlsx"
    
    # 4. 输出目录（可选）
    OUTPUT_DIR = r"C:\item2\qwen-image-plus"
    
    # 5. 模型配置
    MODEL = "qwen-image-plus"  # 使用官方文档指定的模型
    SIZE = "1328*1328"         # 图片尺寸
    
    # 6. Excel配置
    COLUMN_INDEX = 0           # 文本在第几列（0=第一列，1=第二列...）
    HAS_HEADER = True          # Excel是否有标题行
    DELAY = 2.0                # 请求间隔（秒），避免API限制
    
    # 7. 生成参数
    NEGATIVE_PROMPT = ""       # 负面提示词
    PROMPT_EXTEND = True       # 是否扩展提示词
    WATERMARK = False          # 是否添加水印
    
    # ==================== 配置结束 ====================
    
    print("开始批量图片生成...")
    print(f"Excel文件: {EXCEL_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"模型: {MODEL}")
    print(f"图片尺寸: {SIZE}")
    
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
        negative_prompt=NEGATIVE_PROMPT,
        prompt_extend=PROMPT_EXTEND,
        watermark=WATERMARK
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
    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    OUTPUT_DIR = r"C:\item2\qwen-image-plus"
    MODEL = "qwen-image-plus"
    SIZE = "1328*1328"
    
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
            prompt = "一只可爱的小猫在草地上玩耍"
    
    elif choice == "2":
        # 从Excel文件读取
        excel_file = r"C:\item2\文言文12.xlsx"
        
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
        prompt = "一副典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联上左书'义本生知人机同道善思新'，右书'通云赋智乾坤启数高志远'，横批'智启通义'，字体飘逸，在中间挂着一幅中国风的画作，内容是岳阳楼。"
        print(f"使用官方文档示例提示词: {prompt[:100]}...")
    
    else:
        print("无效选择，使用默认提示词")
        prompt = "一只可爱的小猫在草地上玩耍"
    
    if not prompt:
        prompt = "一只可爱的小猫在草地上玩耍"
    
    print(f"\n开始生成单张测试图片...")
    print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"模型: {MODEL}")
    print(f"图片尺寸: {SIZE}")
    print(f"输出目录: {test_output_dir}")
    
    # 生成参数配置
    print("\n请配置生成参数:")
    
    negative_prompt = input("负面提示词（直接回车使用默认值）: ").strip()
    if not negative_prompt:
        negative_prompt = ""
    
    prompt_extend_input = input("是否扩展提示词 (y/n, 默认y): ").strip().lower()
    prompt_extend = True if prompt_extend_input in ['y', 'yes', ''] else False
    
    watermark_input = input("是否添加水印 (y/n, 默认n): ").strip().lower()
    watermark = True if watermark_input in ['y', 'yes'] else False
    
    # 生成单张图片
    print("\n正在调用API生成图片...")
    print(f"请求参数:")
    print(f"  负面提示词: {negative_prompt}")
    print(f"  提示词扩展: {prompt_extend}")
    print(f"  水印: {watermark}")
    
    start_time = time.time()
    
    result = generator.generate_image(
        prompt, 
        row_num="test_single",
        negative_prompt=negative_prompt,
        prompt_extend=prompt_extend,
        watermark=watermark
    )
    
    end_time = time.time()
    
    if result:
        print(f"\n✓ 测试成功！")
        print(f"图片生成耗时: {end_time - start_time:.2f} 秒")
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
        print("5. 查看debug_response_test_single.json文件了解详细错误信息")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("qwen-image-plus API 批量图片生成器")
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