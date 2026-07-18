# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
from pathlib import Path
from hunyuan_image_3 import HunyuanImage3ForCausalMM
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument(
        "--prompt", type=str, required=True,
        help=(
            "Prompt to run. Either a single string, multiple prompts separated by '|||' "
            "(e.g., 'prompt1|||prompt2|||prompt3'), or a path to a JSON/JSONL file "
            "containing a list of prompts."
        )
    )
    parser.add_argument(
        "--image-size-list",
        type=str,
        default=None,
        help=(
            "Per-sample image sizes for batch mode, separated by '|||'. "
            "Each entry follows the same format as --image-size (e.g., '1024x1024|||1280x720'). "
            "Length must match the number of prompts."
        )
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with a list of prompt strings, OR a JSONL file with one "
            "prompt per line. Mutually exclusive with the '|||' separator in --prompt."
        )
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help=(
            "Image to run. For multiple images, use comma-separated paths "
            "(e.g., 'img1.png,img2.png,img3.png'). In batch mode (when multiple prompts "
            "are supplied), images are loaded per-sample using the same convention: each "
            "prompt's image list is separated by '|||' and individual images by ','."
        )
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--image-size", type=str, default="auto",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument(
        "--use-system-prompt",
        type=str,
        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"],
        help=(
            "Use system prompt. 'None' means no system prompt; 'dynamic' means "
            "the system prompt is determined by --bot-task; 'en_vanilla', "
            "'en_recaption', 'en_think_recaption' and 'en_unified' are four "
            "predefined system prompts; 'custom' means using the custom system "
            "prompt. When using 'custom', --system-prompt must be provided. "
            "Default to load from the model generation config."
        )
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt. Used when --use-system-prompt is 'custom'."
    )
    parser.add_argument(
        "--bot-task",
        type=str,
        choices=["image", "auto", "recaption", "think_recaption"],
        help=(
            "Type of task for the model. 'image' for direct image generation; "
            "'auto' for text generation; 'recaption' for re-write->image; "
            "'think_recaption' for think->re-write->image. "
            "Default to load from the model generation config."
        )
    )
    parser.add_argument(
        "--save", type=str, default="image.png",
        help=(
            "Path to save the generated image. In batch mode, use a directory path or a "
            "template with '{idx}' (e.g., 'outputs/img_{idx}.png') to save each sample."
        )
    )
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level")
    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    parser.add_argument(
        "--infer-align-image-size",
        action="store_true",
        help="Whether to align the target image size to the src image size."
    )

    # ======================== Taylor Cache ========================
    parser.add_argument("--use-taylor-cache", action="store_true", help="Use Taylor Cache when sampling.")
    parser.add_argument("--taylor-cache-interval", type=int, default=5, help="Interval of Taylor Cache.")
    parser.add_argument("--taylor-cache-order", type=int, default=2, help="Order of Taylor Cache.")
    parser.add_argument(
        "--taylor-cache-enable-first-enhance",
        action="store_true",
        help="Enable first enhance when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-first-enhance-steps",
        type=int,
        default=3,
        help="First enhance steps when using Taylor Cache (>2)."
    )
    parser.add_argument(
        "--taylor-cache-enable-tailing-enhance",
        action="store_true",
        help="Enable tailing enhance when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-tailing-enhance-steps",
        type=int,
        default=1,
        help="Tailing enhance steps when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-low-freqs-order",
        type=int,
        default=2,
        help="Low freqs order when using Taylor Cache."
    )
    parser.add_argument(
        "--taylor-cache-high-freqs-order",
        type=int,
        default=2,
        help="High freqs order when using Taylor Cache."
    )    
    

    return parser.parse_args()


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def parse_prompt_input(args):
    """Parse --prompt / --prompt-file into a list of prompt strings."""
    if args.prompt_file:
        # Load prompts from file
        path = Path(args.prompt_file)
        if not path.exists():
            raise ValueError(f"Prompt file {args.prompt_file} does not exist")
        text = path.read_text(encoding="utf-8").strip()
        if path.suffix.lower() == ".jsonl":
            prompts = [line.strip() for line in text.splitlines() if line.strip()]
        else:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                prompts = [str(p) for p in data]
            elif isinstance(data, dict) and "prompts" in data:
                prompts = [str(p) for p in data["prompts"]]
            else:
                raise ValueError(
                    f"Prompt JSON must be a list or contain a 'prompts' key, got {type(data)}."
                )
        return prompts
    if "|||" in args.prompt:
        return [p.strip() for p in args.prompt.split("|||") if p.strip()]
    return [args.prompt]


def parse_image_size_list(args, batch_size):
    """Parse --image-size-list into a per-sample list aligned with `batch_size`."""
    if not args.image_size_list:
        return None
    sizes = [s.strip() for s in args.image_size_list.split("|||") if s.strip()]
    if len(sizes) != batch_size:
        raise ValueError(
            f"--image-size-list has {len(sizes)} entries but there are {batch_size} prompts."
        )
    return sizes


def parse_image_input(args, batch_size):
    """Parse --image into a per-sample list aligned with `batch_size`.

    Convention:
        - In single-prompt mode: comma-separated paths -> list of images for that sample.
        - In batch mode: '|||' separates per-sample images; within each, ',' separates paths.
    """
    if not args.image:
        return [None] * batch_size
    if batch_size == 1:
        # Backwards-compatible: comma-separated list -> list of images for the single sample
        paths = [p.strip() for p in args.image.split(",") if p.strip()]
        if len(paths) == 0:
            return [None]
        if len(paths) == 1:
            return [paths[0]]
        return [paths]
    # Batch mode
    per_sample = args.image.split("|||")
    if len(per_sample) != batch_size:
        raise ValueError(
            f"--image has {len(per_sample)} per-sample entries but there are {batch_size} prompts."
        )
    result = []
    for entry in per_sample:
        entry = entry.strip()
        if not entry:
            result.append(None)
            continue
        paths = [p.strip() for p in entry.split(",") if p.strip()]
        if len(paths) == 0:
            result.append(None)
        elif len(paths) == 1:
            result.append(paths[0])
        else:
            result.append(paths)
    return result


def make_save_paths(save_arg: str, batch_size: int) -> list[str]:
    """Resolve a save argument into one path per batch sample."""
    if batch_size == 1:
        return [save_arg]
    # If the save arg contains {idx}, use template substitution
    if "{idx}" in save_arg:
        return [save_arg.replace("{idx}", str(i)) for i in range(batch_size)]
    # Otherwise treat the path as a directory
    save_dir = Path(save_arg)
    # If the user passed a file-like path (with an extension), still create a directory of
    # that name and number the files inside.
    if save_dir.suffix:
        save_dir = save_dir.parent / save_dir.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    return [str(save_dir / f"image_{i}.png") for i in range(batch_size)]


def main(args):
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    # 1. Resolve prompt(s)
    prompts = parse_prompt_input(args)
    batch_size = len(prompts)
    print(f"Batch size: {batch_size}")

    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    # 2. Resolve per-sample images
    image_inputs = parse_image_input(args, batch_size)

    # 3. Resolve per-sample image_size (optional)
    image_size_list = parse_image_size_list(args, batch_size)
    if image_size_list is None:
        # Fall back to single shared image_size for all samples
        image_size_arg = args.image_size
    else:
        image_size_arg = image_size_list

    # 4. Generate
    if batch_size == 1:
        # Backwards-compatible single-sample call
        cot_text, samples = model.generate_image(
            prompt=prompts[0],
            seed=args.seed,
            image_size=image_size_arg,
            use_system_prompt=args.use_system_prompt,
            system_prompt=args.system_prompt,
            bot_task=args.bot_task,
            diff_infer_steps=args.diff_infer_steps,
            verbose=args.verbose,
            max_new_tokens=args.max_new_tokens,
            image=image_inputs[0],
            infer_align_image_size=args.infer_align_image_size,
            use_taylor_cache=args.use_taylor_cache,
            taylor_cache_interval=args.taylor_cache_interval,
            taylor_cache_order=args.taylor_cache_order,
            taylor_cache_enable_first_enhance=args.taylor_cache_enable_first_enhance,
            taylor_cache_first_enhance_steps=args.taylor_cache_first_enhance_steps,
            taylor_cache_enable_tailing_enhance=args.taylor_cache_enable_tailing_enhance,
            taylor_cache_tailing_enhance_steps=args.taylor_cache_tailing_enhance_steps,
            taylor_cache_low_freqs_order=args.taylor_cache_low_freqs_order,
            taylor_cache_high_freqs_order=args.taylor_cache_high_freqs_order,
        )
    else:
        # Broadcast system_prompt to match batch size
        system_prompt = [args.system_prompt] * batch_size if args.system_prompt is not None else None
        cot_text, samples = model.generate_image(
            prompt=prompts,
            seed=args.seed,
            image_size=image_size_arg,
            use_system_prompt=args.use_system_prompt,
            system_prompt=system_prompt,
            bot_task=args.bot_task,
            diff_infer_steps=args.diff_infer_steps,
            verbose=args.verbose,
            max_new_tokens=args.max_new_tokens,
            image=image_inputs,
            infer_align_image_size=args.infer_align_image_size,
            use_taylor_cache=args.use_taylor_cache,
            taylor_cache_interval=args.taylor_cache_interval,
            taylor_cache_order=args.taylor_cache_order,
            taylor_cache_enable_first_enhance=args.taylor_cache_enable_first_enhance,
            taylor_cache_first_enhance_steps=args.taylor_cache_first_enhance_steps,
            taylor_cache_enable_tailing_enhance=args.taylor_cache_enable_tailing_enhance,
            taylor_cache_tailing_enhance_steps=args.taylor_cache_tailing_enhance_steps,
            taylor_cache_low_freqs_order=args.taylor_cache_low_freqs_order,
            taylor_cache_high_freqs_order=args.taylor_cache_high_freqs_order,
        )

    # 6. Save outputs
    save_paths = make_save_paths(args.save, batch_size)
    if not isinstance(samples, list):
        samples = [samples]
    assert len(samples) == batch_size, (
        f"Generated {len(samples)} samples but expected {batch_size}.")
    for path, img in zip(save_paths, samples):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        print(f"Image saved to {path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
