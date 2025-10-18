from openai import OpenAI
import os

api_para = {
    "ali": {
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-r1"
    },
    "silicon": {
        "api_key": "输出你的api_key",
        "base_url": "https://api.siliconflow.cn/v1",
        "model_name": "Pro/deepseek-ai/DeepSeek-R1"
    },
    "baidu": {
        "api_key": "输出你的api_key",
        "base_url": "https://qianfan.baidubce.com/v2",
        "model_name": "deepseek-r1"
    },
    "huoshan": {
        "api_key": "输出你的api_key",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name": "ep-20250219090715-xsthv"
    },
    "tengxunyun": {
        "api_key": "输出你的api_key",
        "base_url": "https://api.lkeap.cloud.tencent.com/v1",
        "model_name": "deepseek-r1"
    },
}


def deepseek_stream(prompt, api_key, base_url, model_name):
    # 初始化OpenAI客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""  # 定义完整回复
    is_answering = False  # 判断是否结束思考过程并开始回复

    # 创建聊天并完成请求
    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in stream:
        if not getattr(chunk, "choices", None):
            print("\n" + "=" * 20 + "Token 使用情况" + "=" * 20 + "\n")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta

        # 检查是否有reasoning_content属性
        if not hasattr(delta, 'reasoning_content'):
            continue

        # 处理空内容情况
        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content', None):
            continue

        # 处理开始回答的情况
        if not getattr(delta, 'reasoning_content', None) and not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            is_answering = True

        # 处理思考过程
        if getattr(delta, 'reasoning_content', None):
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        # 处理回复内容
        elif getattr(delta, 'content', None):
            print(delta.content, end='', flush=True)
            answer_content += delta.content


    # 如果需要打印完整内容，解除以下的注释
    """
    print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
    print(reasoning_content)
    print("=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(answer_content)
    """

if __name__ == '__main__':
    prompt = '''
    解释一下为什么根号2是无理数
    '''
    api_name = 'ali'  # ali,silicon，huoshan，tengxunyun，baidu

    try:
        deepseek_stream(prompt, api_para[api_name]['api_key'], api_para[api_name]['base_url'],
                        api_para[api_name]['model_name'])
    except Exception as e:
        print(f"发生错误：{e}")
