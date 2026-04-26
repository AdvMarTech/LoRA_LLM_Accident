#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA 3模型批量推理程序
适配LLaMA 3模型进行推理
"""

import json
import argparse
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLaMA3Inferencer:
    def __init__(self, model_path: str, device: str = "auto"):
        """
        加载LLaMA 3模型进行推理
        
        Args:
            model_path: 模型路径
            device: 设备 (auto, cuda, cpu)
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        
        logger.info(f"🔄 正在加载LLaMA 3模型: {model_path}")
        logger.info(f"📱 使用设备: {self.device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 设置pad_token，LLaMA通常需要特别设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if "cuda" in str(self.device) else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if "cpu" in str(self.device):
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("✅ LLaMA 3模型加载完成")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _build_prompt(self, instruction: str, input_text: str) -> str:
        """构建LLaMA 3格式的提示词"""
        system_msg = "你是一个专业的事故信息提取助手。请严格按照用户要求的格式提取信息。"
        
        # LLaMA 3 Instruct格式
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n文本内容：\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def inference_single(self, instruction: str, input_text: str) -> str:
        """单条推理"""
        try:
            # 构建提示词
            prompt = self._build_prompt(instruction, input_text)
            
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # 解码输出（只取新生成的部分）
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # 清理LLaMA 3特殊标记
            special_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
            for token in special_tokens:
                if token in response:
                    response = response.split(token)[0].strip()
            
            return response if response else "未能生成有效回答"
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return f"ERROR: 推理失败 - {str(e)}"
    
    def batch_inference(self, data_list: List[Dict[str, Any]], output_file: str) -> bool:
        """批量推理"""
        logger.info(f"🚀 开始批量推理，共 {len(data_list)} 条数据")
        
        success_count = 0
        error_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, data in enumerate(tqdm(data_list, desc="推理进度")):
                try:
                    instruction = data.get('instruction', '')
                    input_text = data.get('input', '')
                    
                    if not instruction or not input_text:
                        raise ValueError("缺少instruction或input字段")
                    
                    # 执行推理
                    output = self.inference_single(instruction, input_text)
                    
                    # 判断是否成功
                    is_success = not output.startswith("ERROR")
                    if is_success:
                        success_count += 1
                    else:
                        error_count += 1
                    
                    # 构造结果
                    result = {
                        "id": i + 1,
                        "instruction": instruction,
                        "input": input_text,
                        "output": output,
                        "status": "success" if is_success else "error",
                        "model": f"LLaMA3-{self.model_path.split('/')[-1]}"
                    }
                    
                    # 保存结果
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    # 显示进度
                    if (i + 1) % 10 == 0 or i == len(data_list) - 1:
                        logger.info(f"已完成 {i+1}/{len(data_list)} 条，成功率: {success_count/(i+1)*100:.1f}%")
                
                except Exception as e:
                    error_count += 1
                    logger.error(f"第 {i+1} 条数据处理失败: {e}")
                    
                    error_result = {
                        "id": i + 1,
                        "instruction": data.get('instruction', ''),
                        "input": data.get('input', ''),
                        "output": f"ERROR: {str(e)}",
                        "status": "error",
                        "model": f"LLaMA3-{self.model_path.split('/')[-1]}"
                    }
                    
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    f.flush()
        
        # 输出最终统计
        total = len(data_list)
        logger.info(f"🎉 推理完成!")
        logger.info(f"📊 总计: {total}, 成功: {success_count}, 失败: {error_count}")
        logger.info(f"📈 成功率: {success_count/total*100:.1f}%")
        logger.info(f"💾 结果已保存到: {output_file}")
        
        return True

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL数据"""
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'instruction' not in data or 'input' not in data:
                            logger.warning(f"第{line_num}行缺少必要字段，跳过")
                            continue
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"第{line_num}行JSON格式错误: {e}")
                        continue
        
        logger.info(f"✅ 成功加载 {len(data_list)} 条数据")
        return data_list
    
    except FileNotFoundError:
        logger.error(f"❌ 文件不存在: {file_path}")
        return []
    except Exception as e:
        logger.error(f"❌ 加载数据失败: {e}")
        return []

def create_sample_data(output_file: str = "sample_data.jsonl"):
    """创建示例数据"""
    sample_data = [
        {
            "instruction": "请从给定文本中提取出事故发生的时间、地点、结果，并将事故原因按 1. 2. 3. ...的格式依次列出。",
            "input": "2016年8月6日0204时左右，中国籍干货船\"永泽1\"轮在长江上海段吴淞口警戒区内64号灯浮下游水域，与中国籍干货船\"金柏5\"轮发生碰撞。直接原因是\"永泽1\"轮未履行让路义务且瞭望疏忽，\"金柏5\"轮未履行直航船义务且避让不及时。\"永泽1\"轮右舷NO.1货舱破损后右倾沉没，12名船员中11人获救、1人失踪，\"金柏5\"轮球鼻艏破损、艏尖舱进水，构成一般等级水上交通事故。"
        },
        {
            "instruction": "请从给定文本中提取出事故发生的时间、地点、结果，并将事故原因按 1. 2. 3. ...的格式依次列出。",
            "input": "2015年7月23日0102时左右，中国籍油船\"百池\"轮在长江上海段A60灯浮上游吴淞口警戒区内，与中国籍散货船\"江夏祥\"轮发生碰撞（概位31°23.4′N、121°33.9′E）。直接原因是\"百池\"轮向左转向导致紧迫局面，双方均未使用安全航速且应急操纵不当。\"百池\"轮艏楼右侧轻微受损，\"江夏祥\"轮左舷船中破损进水后在吴淞口10号锚区侧翻，未造成人员伤亡和水域污染，构成一般等级水上交通事故。"
        },
        {
            "instruction": "请从给定文本中提取出事故发生的时间、地点、结果，并将事故原因按 1. 2. 3. ...的格式依次列出。",
            "input": "2017年3月15日14时30分左右，在东海某海域，中国籍货船\"海丰号\"与韩国籍油轮\"釜山星\"发生碰撞事故。事故原因主要包括：海丰号船员疲劳驾驶，未及时发现目标船舶；釜山星号在大雾天气中航行速度过快；双方船舶均未按规定使用雷达和AIS设备进行避让。事故造成海丰号右舷破损进水，釜山星号轻微受损，幸无人员伤亡。"
        }
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in sample_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ 示例数据已创建: {output_file}")

def check_environment():
    """检查运行环境"""
    logger.info("🔍 检查运行环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🎮 发现 {gpu_count} 个GPU: {gpu_name}")
    else:
        logger.info("💻 将使用CPU运行（速度较慢）")
    
    # 检查内存
    import psutil
    memory = psutil.virtual_memory()
    logger.info(f"💾 系统内存: {memory.total // (1024**3)}GB (可用: {memory.available // (1024**3)}GB)")

def main():
    parser = argparse.ArgumentParser(description="LLaMA 3模型批量推理程序")
    parser.add_argument("--model_path", type=str, required=True, help="LLaMA 3模型路径")
    parser.add_argument("--data_file", type=str, help="输入JSONL文件")
    parser.add_argument("--output_file", type=str, default="llama3_results.jsonl", help="输出文件")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="计算设备")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    parser.add_argument("--check_env", action="store_true", help="检查运行环境")
    
    args = parser.parse_args()
    
    # 检查环境
    if args.check_env:
        check_environment()
        return
    
    # 创建示例数据
    if args.create_sample:
        create_sample_data("sample_data.jsonl")
        return
    
    # 检查输入文件
    if not args.data_file:
        logger.error("❌ 请指定输入数据文件")
        logger.info("💡 使用 --create_sample 创建示例数据")
        logger.info("💡 使用 --check_env 检查运行环境")
        return
    
    try:
        # 检查环境
        check_environment()
        
        # 加载数据
        data_list = load_data(args.data_file)
        if not data_list:
            logger.error("❌ 未能加载有效数据")
            return
        
        # 初始化推理器
        inferencer = LLaMA3Inferencer(args.model_path, args.device)
        
        # 执行批量推理
        success = inferencer.batch_inference(data_list, args.output_file)
        
        if success:
            logger.info("🎉 推理任务完成!")
            
            # 显示结果预览
            logger.info("📋 结果预览:")
            with open(args.output_file, 'r', encoding='utf-8') as f:
                first_result = json.loads(f.readline())
                print(f"输入: {first_result['input'][:100]}...")
                print(f"输出: {first_result['output'][:200]}...")
        else:
            logger.error("❌ 推理任务失败!")
            
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()