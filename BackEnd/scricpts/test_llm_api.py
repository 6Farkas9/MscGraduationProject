import requests
import json

# ======= 需要你自己填写的部分 =======
API_KEY = "sk-TyyUwhwE1LpVBr24swMwKWhWq9oAdosQGYjI5qumbC6DsDoa"  # 在这里填入你的 Aizex API Key（sk- 开头那串）

# 使用主站
BASE_URL = "https://aizex.top/v1"

# 如果你主站访问不了，也可以尝试国内优化路由：
# BASE_URL = "https://a1.aizex.me/v1"

MODEL = "gpt-4.1-nano"  # 确认 Aizex 面板里的模型调用名
# ==================================


def call_aizex_chat(prompt: str):
    url = f"{BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }

    # 方式一：显式禁用代理
    proxies = {
        "http": None,
        "https": None,
    }

    # 如果你确定需要通过某个代理访问（比如 http://127.0.0.1:7890），可以改成：
    # proxies = {
    #     "http": "http://127.0.0.1:7890",
    #     "https": "http://127.0.0.1:7890",
    # }

    try:
        response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=30)
    except Exception as e:
        print("请求过程中出现异常：", repr(e))
        return

    if response.status_code != 200:
        print("请求失败：", response.status_code)
        print("响应内容：", response.text)
        return

    try:
        data = response.json()
    except json.JSONDecodeError:
        print("响应不是合法的 JSON：")
        print(response.text)
        return

    # 解析 OpenAI 兼容格式
    try:
        content = data["choices"][0]["message"]["content"]
        print("模型回复：")
        print(content)
    except Exception as e:
        print("解析返回值时出错：", e)
        print("原始返回：", json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_prompt = "用简短中文介绍一下你是谁，并说一句冷笑话。"
    call_aizex_chat(test_prompt)
