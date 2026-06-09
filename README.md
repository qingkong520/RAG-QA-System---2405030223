# RAG 智能问答系统

基于本地知识库的检索增强生成（RAG）智能问答系统，使用 Ollama 本地大模型、LangChain 框架和 Streamlit 开发。

## 项目简介

本项目实现了一个能够基于本地私有知识库的智能问答系统，支持上传 PDF 和 DOCX 文档，通过检索增强生成技术，让大模型基于您提供的文档回答问题，有效缓解模型幻觉问题。

## 环境要求与安装步骤

### 1. Python 环境
- Python 3.9 或更高版本

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 安装 Ollama
- 访问 [Ollama 官网](https://ollama.com/) 下载并安装 Windows 版本

### 4. 下载模型
```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

### 5. 验证安装
```bash
python test_ollama.py
```

## 使用说明

### 运行 Web 应用
```bash
streamlit run app.py
```

### 使用步骤
1. 启动应用后，在侧边栏上传 PDF 或 DOCX 文档
2. 点击"构建/更新知识库"按钮处理文档
3. 在对话区输入问题并提问
4. 系统会基于知识库回答问题

### 命令行版本
```bash
# 先构建知识库
python build_kb.py

# 运行问答
python rag_chain.py
```

## 关键技术点说明

### RAG 流程
1. **文档加载**：支持 PDF 和 DOCX 格式文档读取
2. **文本分块**：使用 RecursiveCharacterTextSplitter，chunk_size=1000，chunk_overlap=200
3. **向量化存储**：使用 nomic-embed-text 嵌入模型，Chroma 向量数据库
4. **检索**：相似性检索返回 Top 3 相关文档块
5. **生成**：基于检索到的文档生成回答

### 所用模型
- 大模型：deepseek-r1:7b（或 qwen2:7b）
- 嵌入模型：nomic-embed-text

### 嵌入方式
- Ollama 本地嵌入 + Chroma 向量数据库

## 项目效果截图

（使用时请自行添加截图）

## 已知问题与改进方向

- 支持更多文档格式
- 优化向量检索性能
- 添加文档预览功能
