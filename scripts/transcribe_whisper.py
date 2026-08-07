from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import os
import json
import time
import sys
from pathlib import Path

# ===== 模型下载路径配置（修改此处可更改盘符）=====
MODEL_DOWNLOAD_ROOT = "D:/my_ai_models/faster-whisper"

def transcribe_audio_faster(audio_path, output_json_path, language="zh", device="cuda", model="medium"):
    print(f"\n{'='*50}")
    print(f"正在初始化 faster-whisper 模型 ({model}, {device})...")

    model_path = model
    model = WhisperModel(
        model_path,
        device=device,
        compute_type="float16",
        download_root=MODEL_DOWNLOAD_ROOT
    )
    
    print(f"开始转写：{audio_path} ...")
    start_time = time.time()
    
    print("正在将音频提取到内存...")
    try:
        audio_array = decode_audio(audio_path, sampling_rate=16000)
    except Exception as e:
        print(f"❌ 音频解码失败: {e}")
        return
    
    sample_rate = 16000
    chunk_minutes = 20
    chunk_samples = chunk_minutes * 60 * sample_rate
    total_samples = len(audio_array)
    total_minutes = total_samples / sample_rate / 60
    
    print(f"音频总时长: {total_minutes:.1f} 分钟。将按 {chunk_minutes} 分钟/段处理。")
    
    output_segments = []
    
    for i in range(0, total_samples, chunk_samples):
        chunk = audio_array[i : i + chunk_samples]
        offset_seconds = i / sample_rate
        
        start_min = offset_seconds / 60
        end_min = min(total_minutes, (i + len(chunk)) / sample_rate / 60)
        print(f"\n--> 处理片段: {start_min:.1f} 分钟 ~ {end_min:.1f} 分钟 ...")
        
        segments, info = model.transcribe(
            chunk,
            beam_size=5,
            best_of=5,
            language=language,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.4,
                min_speech_duration_ms=100,
                max_speech_duration_s=20
            ),
            temperature=0.0,
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            repetition_penalty=1.1,
            without_timestamps=False,
            word_timestamps=True
        )
        
        for segment in segments:
            actual_start = segment.start + offset_seconds
            actual_end = segment.end + offset_seconds
            output_segments.append({
                "start": round(actual_start, 2),
                "end": round(actual_end, 2),
                "text": segment.text.strip()
            })
            print(f"[{round(actual_start, 2)}s -> {round(actual_end, 2)}s]: {segment.text}")
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_segments, f, ensure_ascii=False, indent=2)
    
    elapsed = time.time() - start_time
    print(f"\n✅ 完成！耗时 {elapsed:.1f} 秒。")
    print(f"结果已保存至：{output_json_path}")

# 支持的音频格式
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac")


def process_folder(input_folder, out_dir=None, language="zh", device="cuda", model="medium"):
    folder_path = Path(input_folder)
    if not folder_path.exists():
        print(f"❌ 文件夹不存在：{input_folder}")
        return

    audio_files = [f for f in folder_path.iterdir() if f.suffix.lower() in AUDIO_EXTS]
    if not audio_files:
        print(f"❌ 文件夹中没有找到音频文件（支持 {AUDIO_EXTS}）：{input_folder}")
        return

    output_folder = Path(out_dir) if out_dir else (folder_path / "transcribed")
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"📁 找到 {len(audio_files)} 个音频文件，开始逐个处理...")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'#'*50}")
        print(f"正在处理第 {i}/{len(audio_files)} 个文件：{audio_file.name}")
        output_path = output_folder / (audio_file.stem + "_transcribed.json")
        try:
            transcribe_audio_faster(str(audio_file), str(output_path),
                                    language=language, device=device, model=model)
        except Exception as e:
            print(f"❌ 处理失败：{e}")
            continue

    print(f"\n🎉 全部完成！共处理 {len(audio_files)} 个文件。")
    print(f"结果保存在：{output_folder}")


def run(input_path=None, out_dir=None, language="zh", device="cuda", model="medium"):
    """参数化入口（供 run_tool 调用）。"""
    path = Path(input_path) if input_path else Path("audio")
    if path.is_dir():
        process_folder(str(path), out_dir=out_dir, language=language, device=device, model=model)
    elif path.is_file() and path.suffix.lower() in AUDIO_EXTS:
        out_dir_p = Path(out_dir) if out_dir else path.parent
        out_dir_p.mkdir(parents=True, exist_ok=True)
        output_path = out_dir_p / (path.stem + "_transcribed.json")
        transcribe_audio_faster(str(path), str(output_path),
                                language=language, device=device, model=model)
    else:
        print(f"❌ 输入路径无效或不是音频文件：{path}")


if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else None
    run(input_path)