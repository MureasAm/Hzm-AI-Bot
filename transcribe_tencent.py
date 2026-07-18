import requests
import json
import time
import hmac
import hashlib
import datetime
import re

# ================== 配置信息 ==================
SECRET_ID = "此处为腾讯云"
SECRET_KEY = "此处为腾讯云API密钥"
AUDIO_URL = "在存储里的URL地址"   # 比如 https://xxx.cos.ap-guangzhou.myqcloud.com/xxx.mp3
OUTPUT_JSON = "tencent_result.json"

# 腾讯云语音识别服务域名
ASR_ENDPOINT = "asr.tencentcloudapi.com"
# 地域
REGION = "ap-guangzhou"

 #================== 签名函数（保持不变） ==================
def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

def get_authorization(action, payload):
    timestamp = int(time.time())
    date = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
    
    service = "asr"
    host = ASR_ENDPOINT
    algorithm = "TC3-HMAC-SHA256"
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    content_type = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{content_type}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = f"{http_request_method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{hashed_request_payload}"
    
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical_request}"
    
    secret_date = sign(("TC3" + SECRET_KEY).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    
    authorization = f"{algorithm} Credential={SECRET_ID}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
    return authorization, timestamp

# ================== 提交识别任务（✅ 修改点1：增加说话人分离参数） ==================
def submit_recognition():
    action = "CreateRecTask"
    payload = json.dumps({
        "EngineModelType": "16k_zh",
        "ChannelNum": 1,
        "ResTextFormat": 2,
        "SourceType": 0,
        "Url": AUDIO_URL,
        "SpeakerDiarization": 1,       # 开启说话人分离
        "SpeakerNumber": 0            # 自动判断说话人数
       
    })
    
    authorization, timestamp = get_authorization(action, payload)
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": ASR_ENDPOINT,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": "2019-06-14",
        "X-TC-Region": REGION
    }
    
    resp = requests.post(f"https://{ASR_ENDPOINT}", headers=headers, data=payload)
    result = resp.json()
    if "Response" in result and "Data" in result["Response"]:
        task_id = result["Response"]["Data"]["TaskId"]
        print(f"✅ 任务提交成功，TaskId: {task_id}")
        return task_id
    else:
        print(f"❌ 提交失败: {result}")
        return None

# ================== 解析内嵌时间戳的纯文本（支持说话人ID） ==================
def parse_inline_timestamps(text):
    """
    解析格式:
    - 老格式: [分钟:秒.毫秒,分钟:秒.毫秒] 文本
    - 新格式 (说话人分离): [分钟:秒.毫秒,分钟:秒.毫秒,说话人ID] 文本
    自动识别并过滤出说话时长最长的主播语音。
    """
    raw_segments = []
    
    # 修改正则：第三个捕获组 (?:,(\d+))? 可选地匹配说话人ID
    pattern = r'\[(\d+:\d+\.\d+),(\d+:\d+\.\d+)(?:,(\d+))?\]\s*(.*?)(?=\n?\[|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for start_str, end_str, speaker_str, content in matches:
        # 解析开始和结束时间
        start_m, start_s = start_str.split(':')
        start_sec = int(start_m) * 60 + float(start_s)
        end_m, end_s = end_str.split(':')
        end_sec = int(end_m) * 60 + float(end_s)
        
        content = content.strip().replace('\n', ' ').replace('\r', ' ')
        if content:
            # 如果有说话人ID则记录，否则设为 -1
            speaker_id = int(speaker_str) if speaker_str else -1
            raw_segments.append({
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "text": content,
                "speaker_id": speaker_id
            })
    
    if not raw_segments:
        return []
    
    # === 说话人过滤逻辑 ===
    # 统计每个 speaker_id 的说话总时长
    speaker_duration = {}
    for seg in raw_segments:
        sid = seg["speaker_id"]
        duration = seg["end"] - seg["start"]
        speaker_duration[sid] = speaker_duration.get(sid, 0) + duration
    
    # 如果只有一个说话人，或者全是 -1（无说话人信息），则保留所有
    if len(speaker_duration) <= 1:
        print("🎤 未检测到多个说话人，保留所有语句。")
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in raw_segments]
    
    # 选择说话时长最长的 speaker 作为主播
    main_speaker = max(speaker_duration, key=speaker_duration.get)
    print(f"🎤 检测到 {len(speaker_duration)} 个说话人，已将 Speaker {main_speaker} 选为主播。")
    
    # 只提取主播的句子
    final_segments = []
    for seg in raw_segments:
        if seg["speaker_id"] == main_speaker:
            final_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })
    
    return final_segments

# ================== 轮询获取结果（✅ 修改点2：加入说话人过滤的 JSON 解析） ==================
def poll_result(task_id, max_wait=300):
    action = "DescribeTaskStatus"
    payload = json.dumps({"TaskId": task_id})
    
    count = 0
    while count < max_wait:
        authorization, timestamp = get_authorization(action, payload)
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": ASR_ENDPOINT,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": "2019-06-14",
            "X-TC-Region": REGION
        }
        resp = requests.post(f"https://{ASR_ENDPOINT}", headers=headers, data=payload)
        result = resp.json()
        status = result["Response"]["Data"]["Status"]
        print(f"⏳ 状态: {status} ({count}s)")
        
        if status == 2:
            result_str = result["Response"]["Data"].get("Result", "")
            print("🔍 原始返回结果（前500字符）:", result_str[:500])
            if not result_str:
                print("⚠️ 任务成功但结果为空。")
                return None
            
            # 1. 优先尝试解析内嵌时间戳格式（你的稳定方案）
            segments = parse_inline_timestamps(result_str)
            if segments:
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                print(f"🎉 解析成功（内嵌时间戳格式）！共 {len(segments)} 句。")
                print(f"结果已保存至: {OUTPUT_JSON}")
                return segments
            
            # 2. 否则尝试 JSON 解析（标准 SentenceList，加入说话人过滤）
            try:
                task_result = json.loads(result_str)
                raw_segments = []
                
                if "SentenceList" in task_result:
                    for sentence in task_result["SentenceList"]:
                        start = sentence["StartTime"] / 1000.0
                        end = sentence["EndTime"] / 1000.0
                        text = sentence.get("WordList", "").strip()
                        speaker_id = sentence.get("SpeakerId", -1)  # 获取说话人ID
                        raw_segments.append({
                            "start": start,
                            "end": end,
                            "text": text,
                            "speaker_id": speaker_id
                        })
                elif "Result" in task_result:
                    # 纯文本兜底（如果直接给了一大段文字）
                    segments = [{"start": 0, "end": 0, "text": task_result["Result"]}]
                    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                        json.dump(segments, f, ensure_ascii=False, indent=2)
                    print(f"🎉 解析成功（纯文本兜底）！共 1 句。")
                    return segments
                
                if not raw_segments:
                    print("❌ 无法从 JSON 中提取句子。")
                    return None
                
                # === 说话人过滤：自动选择说话时长最长的主播 ===
                speaker_duration = {}
                for seg in raw_segments:
                    sid = seg["speaker_id"]
                    duration = seg["end"] - seg["start"]
                    speaker_duration[sid] = speaker_duration.get(sid, 0) + duration
                
                if speaker_duration:
                    main_speaker = max(speaker_duration, key=speaker_duration.get)
                    print(f"🎤 检测到 {len(speaker_duration)} 个说话人，已将 Speaker {main_speaker} 选为主播。")
                else:
                    main_speaker = -1  # 无 SpeakerId，保留所有
                    print("🎤 未检测到说话人标识，将保留所有语句。")
                
                # 提取主播句子
                final_segments = []
                for seg in raw_segments:
                    if main_speaker == -1 or seg["speaker_id"] == main_speaker:
                        final_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"]
                        })
                
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(final_segments, f, ensure_ascii=False, indent=2)
                print(f"🎉 说话人分离完成！主播共有 {len(final_segments)} 句话。")
                print(f"结果已保存至: {OUTPUT_JSON}")
                return final_segments
                
            except json.JSONDecodeError:
                pass
            
            # 3. 如果上述都失败，打印提示（不再保存原始文本）
            print("❌ 无法解析返回结果，请检查音频或联系腾讯云。")
            return None
        
        elif status == 3:
            print(f"❌ 识别失败: {result}")
            return None
        else:
            time.sleep(5)
            count += 5
    print("⏰ 等待超时")
    return None

if __name__ == "__main__":
    tid = submit_recognition()
    if tid:
        poll_result(tid)
