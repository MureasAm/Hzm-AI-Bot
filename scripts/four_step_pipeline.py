import json
import os
from openai import OpenAI
from pathlib import Path

# 项目根目录（scripts/ 的上一级），所有路径基于它构建
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env.prod"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

# 输入文件路径
TRANSCRIPT_FILE = PROJECT_ROOT / "data" / "input_transcript.json"      # 你的转写 JSON
OLD_SYSTEM_PROMPT = PROJECT_ROOT / "persona" / "system_prompt.txt"     # 线上最新版人设提示词

# 提示词模板路径
PROMPT_STEP1 = PROJECT_ROOT / "prompts" / "prompt_step1.txt"
PROMPT_STEP2 = PROJECT_ROOT / "prompts" / "prompt_step2.txt"
PROMPT_STEP3 = PROJECT_ROOT / "prompts" / "prompt_step3.txt"
PROMPT_FUSION = PROJECT_ROOT / "prompts" / "prompt_fusion.txt"

# 惰性客户端：只有真正调用时才读取 key 并创建（避免 run_tool 一 import 就 raise）
_client = None


def get_deepseek_key():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=")[1].replace('"', '').strip()
    return None


def get_client():
    global _client
    if _client is not None:
        return _client
    api_key = get_deepseek_key()
    if not api_key:
        raise ValueError("❌ 未能在 .env.prod 中找到 OPENAI_API_KEY，请检查文件！")
    _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client

def read_file_safe(filepath):
    """
    安全读取文件，自动尝试常见编码：
    UTF-8 with BOM、UTF-8、GBK、GB2312
    若全部失败则抛出异常
    """
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            # 如果读取成功，检查内容是否为空（可能是编码错误导致乱码）
            if content.strip():  # 非空
                return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # 其他异常直接抛出
            raise e
    raise ValueError(f"无法使用任何已知编码读取文件：{filepath}，请检查文件编码是否为 UTF-8 或 GBK。")

def load_transcript(filepath):
    """加载转写 JSON（JSON 必须为 UTF-8，不做自动编码检测）"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif "segments" in data:
        return data["segments"]
    else:
        raise ValueError("转写 JSON 结构无法识别，请检查格式")

def call_api(prompt, temperature=0.7, max_tokens=4096):
    """统一调用 API，返回文本内容"""
    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def extract_json_from_response(content):
    """从模型回复中提取 JSON 数组"""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return json.loads(content)

# ---------- 核心步骤 ----------
def step1_generate_qa(transcript):
    print("[Step 1] 生成 QA 对话对...")
    prompt_template = read_file_safe(PROMPT_STEP1)
    transcript_str = json.dumps(transcript, ensure_ascii=False, indent=2)
    full_prompt = prompt_template + "\n\n【转写文本】\n" + transcript_str

    content = call_api(full_prompt)
    try:
        qa_pairs = extract_json_from_response(content)
        print(f"[Step 1] 成功生成 {len(qa_pairs)} 对 QA")
        return qa_pairs
    except json.JSONDecodeError as e:
        print(f"[Step 1] JSON 解析失败: {e}")
        with open(PROJECT_ROOT / "outputs" / "step1_raw_output.txt", "w", encoding="utf-8") as f:
            f.write(content)
        raise

def step2_analyze_persona(qa_pairs):
    print("[Step 2] 生成人格切面分析...")
    template = read_file_safe(PROMPT_STEP2)
    qa_json_str = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
    prompt = template.replace("{qa_json}", qa_json_str)

    analysis = call_api(prompt, max_tokens=4096)
    print("[Step 2] 人格分析完成")
    return analysis

def step3_generate_system_prompt(qa_pairs, persona_analysis):
    print("[Step 3] 生成系统提示词草案...")
    template = read_file_safe(PROMPT_STEP3)
    qa_json_str = json.dumps(qa_pairs, ensure_ascii=False, indent=2)
    prompt = template.replace("{qa_json}", qa_json_str).replace("{persona_analysis}", persona_analysis)

    suggestion = call_api(prompt, max_tokens=4096)
    print("[Step 3] 系统提示词草案完成")
    return suggestion

def step4_fusion(old_prompt_path, new_prompt_path, output_path):
    print("[Step 4] 融合新旧提示词...")
    # 检查旧提示词是否存在
    if not os.path.exists(old_prompt_path):
        print(f"[Step 4] 警告：旧版提示词 {old_prompt_path} 不存在，将直接使用新草案作为最终版本。")
        new_content = read_file_safe(new_prompt_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return new_content

    old_prompt = read_file_safe(old_prompt_path)
    new_prompt = read_file_safe(new_prompt_path)
    template = read_file_safe(PROMPT_FUSION)

    full_prompt = template.replace("{old_system_prompt}", old_prompt).replace("{new_system_prompt}", new_prompt)
    merged = call_api(full_prompt, max_tokens=4096)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(merged)
    print(f"[Step 4] 升级版提示词已保存至 {output_path}")
    return merged

# ---------- 主流程 ----------
def run(transcript_file=None, old_prompt=None, out_dir=None, do_fusion=True):
    """参数化入口（供 run_tool 调用）。空参数回落默认路径。"""
    transcript_path = Path(transcript_file) if transcript_file else TRANSCRIPT_FILE
    old_prompt_path = Path(old_prompt) if old_prompt else OLD_SYSTEM_PROMPT
    out = Path(out_dir) if out_dir else Path("outputs/pipeline")

    qa_out = out / "qa_pairs.json"
    analysis_out = out / "persona_analysis.md"
    draft_out = out / "system_prompt_suggestion.md"
    merged_out = out / "system_prompt_upgraded.txt"
    out.mkdir(parents=True, exist_ok=True)

    print("===== 四步人格蒸馏流水线启动 =====")

    # 1. 检查转写文件
    if not os.path.exists(transcript_path):
        print(f"错误：找不到转写文件 {transcript_path}")
        return

    transcript = load_transcript(transcript_path)

    # 第一步
    qa_pairs = step1_generate_qa(transcript)
    with open(qa_out, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    # 第二步
    persona_analysis = step2_analyze_persona(qa_pairs)
    with open(analysis_out, "w", encoding="utf-8") as f:
        f.write(persona_analysis)

    # 第三步
    new_system_prompt = step3_generate_system_prompt(qa_pairs, persona_analysis)
    with open(draft_out, "w", encoding="utf-8") as f:
        f.write(new_system_prompt)

    # 第四步（可选）
    if do_fusion:
        step4_fusion(
            old_prompt_path=old_prompt_path,
            new_prompt_path=draft_out,
            output_path=merged_out
        )

    print("\n===== 流水线完成 =====")
    print("生成文件：")
    print(f"  - QA 对：{qa_out}")
    print(f"  - 人格分析：{analysis_out}")
    print(f"  - 新草案：{draft_out}")
    if do_fusion:
        print(f"  - 升级版提示词：{merged_out}")
        print("\n建议：请人工审核升级版提示词，确认无误后覆盖 persona/system_prompt.txt。")


def main():
    run()


if __name__ == "__main__":
    main()