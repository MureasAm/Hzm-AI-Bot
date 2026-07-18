from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
import os
import json
import time
import sys
from pathlib import Path

# ===== 模型下载路径配置（修改此处可更改盘符）=====
MODEL_DOWNLOAD_ROOT = "D:/my_ai_models/faster-whisper"

def transcribe_audio_faster(audio_path, output_json_path):
    print(f"\n{'='*50}")
    print("正在初始化 faster-whisper 模型 (medium, GPU 高精度模式)...")
    
    model_path = "medium"
    model = WhisperModel(
        model_path,
        device="cuda",
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
            language="zh",
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

def process_folder(input_folder):
    folder_path = Path(input_folder)
    if not folder_path.exists():
        print(f"❌ 文件夹不存在：{input_folder}")
        return
    
    mp3_files = list(folder_path.glob("*.mp3"))
    if not mp3_files:
        print(f"❌ 文件夹中没有找到 .mp3 文件：{input_folder}")
        return
    
    output_folder = folder_path / "transcribed"
    output_folder.mkdir(exist_ok=True)
    
    print(f"📁 找到 {len(mp3_files)} 个 mp3 文件，开始逐个处理...")
    
    for i, mp3_file in enumerate(mp3_files, 1):
        print(f"\n{'#'*50}")
        print(f"正在处理第 {i}/{len(mp3_files)} 个文件：{mp3_file.name}")
        output_path = output_folder / (mp3_file.stem + "_transcribed.json")
        try:
            transcribe_audio_faster(str(mp3_file), str(output_path))
        except Exception as e:
            print(f"❌ 处理失败：{e}")
            continue
    
    print(f"\n🎉 全部完成！共处理 {len(mp3_files)} 个文件。")
    print(f"结果保存在：{output_folder}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "audio"
    
    input_path = Path(input_path)
    
    if input_path.is_dir():
        process_folder(str(input_path))
    elif input_path.is_file() and input_path.suffix.lower() == ".mp3":
        output_path = input_path.stem + "_transcribed.json"
        transcribe_audio_faster(str(input_path), str(output_path))
    else:
        print(f"❌ 输入路径无效或不是 mp3 文件：{input_path}")