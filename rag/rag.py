import fitz  # PyMuPDF库，用于处理PDF文件
import os  # 操作系统相关功能
import numpy as np  # NumPy库，用于数值计算
import json  # JSON数据处理
from openai import OpenAI  # OpenAI API客户端
from dotenv import load_dotenv
import os
from comm import extract_text_from_pdf, chunk_text, create_embeddings, semantic_search, generate_response


load_dotenv()  # 加载.env文件
api_key = os.getenv("OPENAI_API_KEY")  # 读取密钥
print(api_key)
# 初始化 OpenAI 客户端，设置基础 URL 和 API 密钥
client = OpenAI(
    base_url="https://api.openai.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥
)

# 定义PDF文件的路径
pdf_path = "data/AI_Information.pdf"

# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 将提取的文本分割为每段1000个字符并带有200个字符重叠的片段
text_chunks = chunk_text(extracted_text, 1000, 200)

# 打印生成的文本片段数量
print("Number of text chunks:", len(text_chunks))

# 打印第一个文本片段
print("\nFirst text chunk:")
print(text_chunks[0])

# 为文本块创建嵌入
response = create_embeddings(text_chunks)

# 从JSON文件中加载验证数据
with open('data/val.json') as f:
    data = json.load(f)

# 从验证数据中提取第一个查询
query = data[0]['question']

# 执行语义搜索以找到与查询最相关的前2个文本片段
top_chunks = semantic_search(query, text_chunks, response.data, k=2)

# 打印查询
print("Query:", query)

# 打印前2个最相关的文本片段
for i, chunk in enumerate(top_chunks):
    print(f"Context {i + 1}:\n{chunk}\n=====================================")

# 定义AI助手的系统提示
system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"


# 基于top片段创建用户提示
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成AI回复
ai_response = generate_response(system_prompt, user_prompt)
print(ai_response.choices[0].message.content)

# 定义评估系统的系统提示
evaluate_system_prompt = "You are an intelligent evaluation system tasked with assessing the AI assistant's responses. If the AI assistant's response is very close to the true response, assign a score of 1. If the response is incorrect or unsatisfactory in relation to the true response, assign a score of 0. If the response is partially aligned with the true response, assign a score of 0.5."
# 通过组合用户查询、AI回复、真实回复和评估系统提示创建评估提示
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示和评估提示生成评估回复
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# 打印评估回复
print(evaluation_response.choices[0].message.content)


