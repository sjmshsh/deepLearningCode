import fitz  # PyMuPDF库，用于处理PDF文件
import os  # 操作系统相关功能
import numpy as np  # NumPy库，用于数值计算
import json  # JSON数据处理
from openai import OpenAI  # OpenAI API客户端
from dotenv import load_dotenv
import os

load_dotenv()  # 加载.env文件
api_key = os.getenv("OPENAI_API_KEY")  # 读取密钥
print(api_key)
# 初始化 OpenAI 客户端，设置基础 URL 和 API 密钥
client = OpenAI(
    base_url="https://api.openai.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥
)



def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本并打印前`num_chars`个字符。

    参数:
    pdf_path (str): PDF文件的路径。

    返回:
    str: 从PDF中提取的文本。
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""  # 初始化一个空字符串用于存储提取的文本

    # 遍历PDF中的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # 获取页面
        text = page.get_text("text")  # 从页面提取文本
        all_text += text  # 将提取的文本追加到all_text字符串中

    return all_text  # 返回提取的文本


def chunk_text(text, n, overlap):
    """
    将给定的文本分割为长度为 n 的段，并带有指定的重叠字符数。

    参数:
    text (str): 需要分割的文本。
    n (int): 每个片段的字符数量。
    overlap (int): 段与段之间的重叠字符数量。

    返回:
    List[str]: 一个包含文本片段的列表。
    """
    chunks = []  # 初始化一个空列表用于存储片段

    # 使用 (n - overlap) 的步长遍历文本
    for i in range(0, len(text), n - overlap):
        # 将从索引 i 到 i + n 的文本片段添加到 chunks 列表中
        chunks.append(text[i:i + n])

    return chunks  # 返回包含文本片段的列表


def create_embeddings(text, model="text-embedding-ada-002"):
    """
    使用指定的OpenAI模型为给定文本创建嵌入。

    参数:
    text (str): 需要为其创建嵌入的输入文本。
    model (str): 用于创建嵌入的模型。

    返回:
    dict: 包含嵌入结果的OpenAI API回复。
    """
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=text
    )

    return response  # 返回包含嵌入结果的回复


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数:
    vec1 (np.ndarray): 第一个向量。
    vec2 (np.ndarray): 第二个向量。

    返回:
    float: 两个向量之间的余弦相似度。
    """
    # 计算两个向量的点积，并除以它们范数的乘积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query, text_chunks, embeddings, k=5):
    """
    使用给定的查询和嵌入对文本块执行语义搜索。

    参数:
    query (str): 语义搜索的查询。
    text_chunks (List[str]): 要搜索的文本块列表。
    embeddings (List[dict]): 文本块的嵌入列表。
    k (int): 返回的相关文本块数量。默认值为5。

    返回:
    List[str]: 基于查询的前k个最相关文本块列表。
    """
    # 为查询创建嵌入
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []  # 初始化一个用于存储相似度分数的列表

    # 计算查询嵌入与每个文本块嵌入之间的相似度分数
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))  # 将索引和相似度分数追加到列表中

    # 按降序对相似度分数进行排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 获取前k个最相似文本块的索引
    top_indices = [index for index, _ in similarity_scores[:k]]
    # 返回前k个最相关的文本块
    return [text_chunks[index] for index in top_indices]

def generate_response(system_prompt, user_message, model="gpt-3.5-turbo"):
    """
    根据系统提示和用户消息生成AI模型的回复。

    参数:
    system_prompt (str): 指导AI行为的系统提示。
    user_message (str): 用户的消息或查询。
    model (str): 用于生成回复的模型。

    返回:
    dict: AI模型的回复。
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

def get_embedding(text, model="text-embedding-ada-002"):
    """
    使用指定模型为给定文本生成嵌入向量。

    参数:
    text (str): 输入文本。
    model (str): 嵌入模型名称。

    返回:
    np.ndarray: 嵌入向量。
    """
    response = client.embeddings.create(model=model, input=text)
    return np.array(response.data[0].embedding)

def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    根据相似度下降计算分块断点。

    参数:
    similarities (List[float]): 句子之间的相似度分数列表。
    method (str): 'percentile', 'standard_deviation' 或 'interquartile'。
    threshold (float): 阈值（对于 'percentile' 是百分位数，对于 'standard_deviation' 是标准差的数量）。

    返回:
    List[int]: 应该发生分块分裂的索引位置列表。
    """
    # 根据所选方法确定阈值
    if method == "percentile":
        # 计算相似度分数的第X百分位
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度分数的平均值和标准差
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 将阈值设置为平均值减去X个标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算第一和第三四分位数（Q1 和 Q3）
        q1, q3 = np.percentile(similarities, [25, 75])
        # 使用IQR规则设置阈值以检测异常值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        # 如果提供了无效方法，则引发错误
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的位置索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

def split_into_chunks(sentences, breakpoints):
    """
    将句子分割为语义块。

    参数:
    sentences (List[str]): 句子列表。
    breakpoints (List[int]): 应该发生分割的索引位置。

    返回:
    List[str]: 文本块列表。
    """
    chunks = []  # 初始化一个空列表来存储块
    start = 0  # 初始化起始索引

    # 遍历每个断点以创建块
    for bp in breakpoints:
        # 将从起始到当前断点的句子片段加入块中
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1  # 更新起始索引到断点后的下一个句子

    # 将剩余的句子作为最后一个块加入
    chunks.append(". ".join(sentences[start:]))
    return chunks  # 返回块列表


def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    查找与查询最相关的文本片段。

    参数:
    query (str): 搜索查询。
    text_chunks (List[str]): 文本片段列表。
    chunk_embeddings (List[np.ndarray]): 文本片段嵌入列表。
    k (int): 返回的最相关结果数量。

    返回:
    List[str]: 前k个最相关的文本片段。
    """
    # 为查询生成一个嵌入向量
    query_embedding = get_embedding(query)

    # 计算查询嵌入与每个片段嵌入之间的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取最相似的k个片段的索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回前k个最相关的文本片段
    return [text_chunks[i] for i in top_indices]


