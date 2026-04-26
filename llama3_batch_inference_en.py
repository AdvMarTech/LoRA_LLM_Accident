#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA 3 batch inference script
Adapted for inference with LLaMA 3 models.
"""

import json
import argparse
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLaMA3Inferencer:
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Load a LLaMA 3 model for inference.

        Args:
            model_path: Path to the model.
            device: Device to use (auto, cuda, cpu).
        """
        self.model_path = model_path
        self.device = self._setup_device(device)

        logger.info(f"Loading LLaMA 3 model: {model_path}")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        # Set pad_token. LLaMA models usually require this to be set explicitly.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
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
        logger.info("LLaMA 3 model loaded successfully")

    def _setup_device(self, device: str) -> torch.device:
        """Set up the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _build_prompt(self, instruction: str, input_text: str) -> str:
        """Build a prompt in the LLaMA 3 chat format."""
        system_msg = "You are a professional accident information extraction assistant. Please extract information strictly according to the user-specified format."

        # LLaMA 3 Instruct format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\nText content:\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    def inference_single(self, instruction: str, input_text: str) -> str:
        """Run inference for a single sample."""
        try:
            # Build prompt
            prompt = self._build_prompt(instruction, input_text)

            # Encode input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False
            ).to(self.device)

            # Generate response
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

            # Decode output. Keep only the newly generated tokens.
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Remove LLaMA 3 special tokens
            special_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]
            for token in special_tokens:
                if token in response:
                    response = response.split(token)[0].strip()

            return response if response else "Failed to generate a valid response"

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return f"ERROR: Inference failed - {str(e)}"

    def batch_inference(self, data_list: List[Dict[str, Any]], output_file: str) -> bool:
        """Run batch inference."""
        logger.info(f"Starting batch inference on {len(data_list)} samples")

        success_count = 0
        error_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            for i, data in enumerate(tqdm(data_list, desc="Inference progress")):
                try:
                    instruction = data.get('instruction', '')
                    input_text = data.get('input', '')

                    if not instruction or not input_text:
                        raise ValueError("Missing instruction or input field")

                    # Run inference
                    output = self.inference_single(instruction, input_text)

                    # Check whether inference succeeded
                    is_success = not output.startswith("ERROR")
                    if is_success:
                        success_count += 1
                    else:
                        error_count += 1

                    # Build result
                    result = {
                        "id": i + 1,
                        "instruction": instruction,
                        "input": input_text,
                        "output": output,
                        "status": "success" if is_success else "error",
                        "model": f"LLaMA3-{self.model_path.split('/')[-1]}"
                    }

                    # Save result
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()

                    # Show progress
                    if (i + 1) % 10 == 0 or i == len(data_list) - 1:
                        logger.info(f"Completed {i+1}/{len(data_list)} samples, success rate: {success_count/(i+1)*100:.1f}%")

                except Exception as e:
                    error_count += 1
                    logger.error(f"Failed to process sample {i+1}: {e}")

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

        # Print final statistics
        total = len(data_list)
        logger.info("Inference completed!")
        logger.info(f"Total: {total}, succeeded: {success_count}, failed: {error_count}")
        logger.info(f"Success rate: {success_count/total*100:.1f}%")
        logger.info(f"Results saved to: {output_file}")

        return True

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data."""
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'instruction' not in data or 'input' not in data:
                            logger.warning(f"Line {line_num} is missing required fields. Skipping.")
                            continue
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON format error on line {line_num}: {e}")
                        continue

        logger.info(f"Successfully loaded {len(data_list)} samples")
        return data_list

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return []

def create_sample_data(output_file: str = "sample_data.jsonl"):
    """Create sample data."""
    sample_data = [
        {
            "instruction": "Extract the time, location, and outcome of the accident from the given text, and list the accident causes in the format 1. 2. 3. ...",
            "input": "At around 02:04 on August 6, 2016, the Chinese dry cargo vessel 'Yongze 1' collided with the Chinese dry cargo vessel 'Jinbai 5' in the waters downstream of Light Buoy No. 64 in the Wusongkou precautionary area of the Shanghai section of the Yangtze River. The direct causes were that 'Yongze 1' failed to fulfill its give-way obligation and maintained an improper lookout, while 'Jinbai 5' failed to fulfill its obligation as the stand-on vessel and did not take timely avoiding action. After the collision, the No. 1 cargo hold on the starboard side of 'Yongze 1' was damaged; the vessel listed to starboard and sank. Of the 12 crew members, 11 were rescued and 1 went missing. The bulbous bow of 'Jinbai 5' was damaged, and the forepeak tank was flooded. The accident constituted a general water traffic accident."
        },
        {
            "instruction": "Extract the time, location, and outcome of the accident from the given text, and list the accident causes in the format 1. 2. 3. ...",
            "input": "At around 01:02 on July 23, 2015, the Chinese tanker 'Baichi' collided with the Chinese bulk carrier 'Jiangxiaxiang' in the Wusongkou precautionary area upstream of Light Buoy A60 in the Shanghai section of the Yangtze River, at an approximate position of 31°23.4'N, 121°33.9'E. The direct causes were that 'Baichi' turned to port and created a close-quarters situation, both vessels failed to proceed at a safe speed, and both vessels handled the emergency maneuver improperly. The starboard side of the forecastle of 'Baichi' was slightly damaged. The port midship section of 'Jiangxiaxiang' was damaged and flooded, and the vessel capsized in Wusongkou Anchorage No. 10. No casualties or water pollution occurred. The accident constituted a general water traffic accident."
        },
        {
            "instruction": "Extract the time, location, and outcome of the accident from the given text, and list the accident causes in the format 1. 2. 3. ...",
            "input": "At around 14:30 on March 15, 2017, in a certain sea area of the East China Sea, the Chinese cargo vessel 'Haifeng' collided with the South Korean tanker 'Busan Star'. The main causes of the accident included fatigue driving by the crew of 'Haifeng', failure to detect the target vessel in time, excessive speed by 'Busan Star' in foggy weather, and failure by both vessels to use radar and AIS equipment for collision avoidance as required. The accident caused flooding through damage to the starboard side of 'Haifeng' and minor damage to 'Busan Star'. Fortunately, there were no casualties."
        }
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in sample_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    logger.info(f"Sample data created: {output_file}")

def check_environment():
    """Check the runtime environment."""
    logger.info("Checking runtime environment...")

    # Check CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Found {gpu_count} GPU(s): {gpu_name}")
    else:
        logger.info("Running on CPU. This may be slow.")

    # Check memory
    import psutil
    memory = psutil.virtual_memory()
    logger.info(f"System memory: {memory.total // (1024**3)} GB (available: {memory.available // (1024**3)} GB)")

def main():
    parser = argparse.ArgumentParser(description="LLaMA 3 batch inference script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LLaMA 3 model")
    parser.add_argument("--data_file", type=str, help="Input JSONL file")
    parser.add_argument("--output_file", type=str, default="llama3_results.jsonl", help="Output file")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Computation device")
    parser.add_argument("--create_sample", action="store_true", help="Create sample data")
    parser.add_argument("--check_env", action="store_true", help="Check the runtime environment")

    args = parser.parse_args()

    # Check environment
    if args.check_env:
        check_environment()
        return

    # Create sample data
    if args.create_sample:
        create_sample_data("sample_data.jsonl")
        return

    # Check input file
    if not args.data_file:
        logger.error("Please specify an input data file")
        logger.info("Use --create_sample to create sample data")
        logger.info("Use --check_env to check the runtime environment")
        return

    try:
        # Check environment
        check_environment()

        # Load data
        data_list = load_data(args.data_file)
        if not data_list:
            logger.error("No valid data could be loaded")
            return

        # Initialize inferencer
        inferencer = LLaMA3Inferencer(args.model_path, args.device)

        # Run batch inference
        success = inferencer.batch_inference(data_list, args.output_file)

        if success:
            logger.info("Inference task completed!")

            # Show result preview
            logger.info("Result preview:")
            with open(args.output_file, 'r', encoding='utf-8') as f:
                first_result = json.loads(f.readline())
                print(f"Input: {first_result['input'][:100]}...")
                print(f"Output: {first_result['output'][:200]}...")
        else:
            logger.error("Inference task failed!")

    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
