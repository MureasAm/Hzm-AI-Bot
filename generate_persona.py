import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI


# ==========================
# 读取 RAW_CORPUS
# ==========================

# 直接从你的 generate_vectors.py 导入
from generate_vectors import RAW_CORPUS


# ==========================
# 配置
# ==========================

def get_deepseek_key():
    env_path = Path(".env.prod")

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY"):
                    return line.split("=")[1].replace('"', '').strip()

    return None


client = None


# ==========================
# 人格分类 Prompt
# ==========================

PERSONA_PROMPT = """
你是一名专业的虚拟角色人格分析专家。
在分析时，请特别关注并提取：
1. 她是如何主动推进对话的？（反问、自爆、抛梗、开启新话题）
2. 她如何维持与他人的连接？（不只是如何保护自己）
3. 她的进攻性表现在哪些地方？（不只是防御）

请确保提取的规则中，防御型行为（如被夸时嘴硬、被质疑时心虚、被越界时推拉）与进攻型行为（如主动抛梗、反问观众、调侃对方提问动机、自爆糗事填补冷场、预判对方意图并抢先调侃）的数量比例大致为 1:1。如果素材中进攻型行为充足，进攻型规则不得少于防御型规则。


你的任务：
分析下面的角色行为素材，并拆分为三个层级：

1. trait（稳定人格特质）
回答：
“这个角色长期是什么样的人？”

例如：
- 回避型亲近者
- 高敏感
- 害怕麻烦别人
- 嘴硬心软

注意：角色的核心特质通常是矛盾的（如"渴望亲密却又推开他人"、"清醒悲观却又选择乐观面对"）。如果你的分析中所有 traits 都是单一方向的描述，请重新审视素材，寻找那些互相拉扯、彼此矛盾的行为模式——那才是角色最真实的内核。

2. style（语言表达风格）
回答：
“这个角色怎么说话？”

注意区分 style 和 behavior：style 是"她无论什么情境都这么说话"的固定语言习惯（如括号、自称、重复），behavior 是"遇到特定情境才会触发"的反应模式（如被夸时嘴硬、冷场时主动自爆）。如果某个语言特征只在特定情境下出现，应归入 behavior 而非 style。

例如：
- 高频使用括号自嘲
- 省略号很多
- 喜欢重复
- 使用hzm自称
请特别关注并提取：
1. 她在“推进对话”时的特定句式（反问句、追加细节的转折词、开启新话题的标志词）
2. 她在“主动逗乐对方”时的语言结构
不要只关注括号、自称等防御型风格。
3. 她在互动中主动出击的时刻有哪些？（不只是被动回应）例如：反问观众、调侃弹幕的提问动机、主动自爆糗事、预判对方想说什么并抢先调侃、用玩笑把话题抛回去。这些都属于"主动引导对话"而非"被动防御"。


3. behavior（触发式行为模式）
回答：
“遇到某种情况，她会怎么反应？”

格式：
trigger:
触发事件

response:
典型反应流程


要求：

- 不要心理诊断
- 不要创造素材不存在的人格
- 必须基于提供内容
- 同一个特征如果多个片段出现，合并
- 输出 JSON
- 部分素材的结尾没有标注人格标签。请根据行为本身（而非标签词）来判断其属于哪种特质或行为模式。如果一个行为在多个素材中反复出现，即使没有被显式命名，也应提取为规则。



输出格式：

{
 "traits":[
   {
    "name":"",
    "description":"",
    "evidence":[]
   }
 ],

 "styles":[
   {
    "name":"",
    "description":"",
    "evidence":[]
   }
 ],

 "behaviors":[
   {
    "name":"",
    "trigger":"",
    "response":"",
    "evidence":[]
   }
 ]
}


素材：

"""


# ==========================
# 调用模型分析
# ==========================

async def analyze_persona():

    global client

    key = get_deepseek_key()

    if not key:
        print("❌ 未找到 OPENAI_API_KEY")
        return


    client = AsyncOpenAI(
        api_key=key,
        base_url="https://api.deepseek.com/v1"
    )


    # 合并素材

    corpus_text = "\n\n".join(
        [
            f"【事件{i+1}】\n{x['statement']}"
            for i, x in enumerate(RAW_CORPUS)
        ]
    )


    print("🧠 正在分析人格结构...")


    response = await client.chat.completions.create(
        model="deepseek-chat",

        messages=[
            {
                "role":"system",
                "content":PERSONA_PROMPT
            },
            {
                "role":"user",
                "content":corpus_text
            }
        ],

        temperature=0.2
    )


    content = response.choices[0].message.content


    # 清理 markdown

    content = content.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()


    data = json.loads(content)


    # ======================
    # 保存三个文件
    # ======================


    with open(
        "persona_traits.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data["traits"],
            f,
            ensure_ascii=False,
            indent=2
        )


    with open(
        "persona_styles.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data["styles"],
            f,
            ensure_ascii=False,
            indent=2
        )


    with open(
        "persona_behaviors.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data["behaviors"],
            f,
            ensure_ascii=False,
            indent=2
        )


    print("✅ 人格拆分完成")
    print("")
    print("生成:")
    print(" - persona_traits.json")
    print(" - persona_styles.json")
    print(" - persona_behaviors.json")



if __name__ == "__main__":

    asyncio.run(analyze_persona())