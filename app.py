import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from vector_db import (
    load_vector_db, create_vector_db,
    add_documents_to_vector_db, get_vector_db_stats,
    split_documents, retrieve_similar_documents
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


SYSTEM_PROMPT = """你是一个问答助手，必须完全基于提供的参考文档回答。

你的回答必须严格遵守以下规则：
1. 仅使用参考文档中的信息回答问题
2. 如果参考文档中没有相关信息，必须直接说"文档中未找到相关答案"
3. 绝对不要编造任何参考文档中没有的内容
4. 不要使用任何你自己的知识，只使用提供的文档

参考文档：
{context}

问题：{question}

回答："""


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0


def load_or_create_vector_db():
    try:
        vector_db = load_vector_db()
        if vector_db:
            stats = get_vector_db_stats(vector_db)
            st.session_state.vector_db = vector_db
            st.session_state.chunk_count = stats["chunk_count"]
            return True
    except Exception:
        pass  # 如果加载失败（比如 Ollama 未启动），也不要崩溃
    return False


def create_chain(vector_db):
    llm = Ollama(model="deepseek-r1:7b")
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return chain


def ask_question_with_check(chain, vector_db, question):
    """先检查检索结果，再决定如何回答"""
    # 先直接检索看看有没有相关文档
    docs = retrieve_similar_documents(vector_db, question, k=3)
    
    # 如果没有检索到任何文档，直接返回固定答案，不调用模型！
    if len(docs) == 0:
        return {
            "answer": "文档中未找到相关答案",
            "source_documents": []
        }
    
    # 调用大模型
    result = chain({"question": question})
    answer = result["answer"]
    
    # 终极保障：检查回答
    target_phrase = "文档中未找到相关答案"
    
    # 如果回答里包含了我们要求的那句话，就只返回那句话
    if target_phrase in answer:
        return {
            "answer": target_phrase,
            "source_documents": result["source_documents"]
        }
    
    # 否则返回模型的回答
    return {
        "answer": answer,
        "source_documents": result["source_documents"]
    }


def process_uploaded_files(uploaded_files):
    documents = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(tmp_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        
        os.unlink(tmp_path)
    
    return documents


def main():
    st.set_page_config(page_title="RAG 智能问答系统", page_icon="🤖")
    st.title("🤖 RAG 智能问答系统")
    
    init_session_state()
    
    # 检查 Ollama 是否可用
    ollama_available = True
    try:
        if not st.session_state.vector_db:
            load_or_create_vector_db()
    except Exception:
        ollama_available = False
        st.warning("⚠️ Ollama 服务未启动！请先启动 Ollama 服务才能使用知识库功能。")
    
    with st.sidebar:
        st.header("📚 知识库管理")
        
        if ollama_available:
            uploaded_files = st.file_uploader(
            "上传文档 (PDF/DOCX/TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
            
            if uploaded_files and st.button("🔨 构建/更新知识库"):
                try:
                    with st.spinner("正在处理文档..."):
                        documents = process_uploaded_files(uploaded_files)
                        
                        if documents:
                            if not st.session_state.vector_db:
                                vector_db, chunk_count = create_vector_db(documents)
                                st.session_state.vector_db = vector_db
                                st.session_state.chunk_count = chunk_count
                            else:
                                chunk_count = add_documents_to_vector_db(st.session_state.vector_db, documents)
                                st.session_state.chunk_count += chunk_count
                            
                            st.session_state.chain = create_chain(st.session_state.vector_db)
                            st.success(f"知识库已更新！新增 {chunk_count} 个文本块")
                        else:
                            st.error("未找到有效的文档！")
                except Exception as e:
                    st.error(f"构建知识库失败：{e}\n请确保 Ollama 服务已启动并已下载 nomic-embed-text 模型！")
            
            st.divider()
            
            if st.session_state.vector_db:
                st.success(f"✅ 知识库已加载")
                st.info(f"📄 文本块数量: {st.session_state.chunk_count}")
            else:
                st.warning("⚠️ 知识库未初始化，请先上传文档")
            
            if st.button("🗑️ 清空对话历史"):
                st.session_state.messages = []
                if st.session_state.vector_db:
                    try:
                        st.session_state.chain = create_chain(st.session_state.vector_db)
                    except Exception:
                        pass
                st.success("对话历史已清空")
        else:
            st.error("❌ 请先启动 Ollama 服务！")
            st.info("启动步骤：")
            st.code("""1. 在开始菜单打开 Ollama 应用
2. 或在命令行运行：ollama serve
3. 下载模型：
   ollama pull deepseek-r1:7b
   ollama pull nomic-embed-text""")
    
    st.header("💬 对话")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("请输入您的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if not ollama_available:
            with st.chat_message("assistant"):
                st.warning("请先启动 Ollama 服务！")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "请先启动 Ollama 服务！"
            })
        elif not st.session_state.vector_db:
            with st.chat_message("assistant"):
                st.warning("请先上传文档并构建知识库！")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "请先上传文档并构建知识库！"
            })
        else:
            try:
                if not st.session_state.chain:
                    st.session_state.chain = create_chain(st.session_state.vector_db)
                
                with st.chat_message("assistant"):
                    with st.spinner("正在思考..."):
                        # 使用带检查的问答函数
                        result = ask_question_with_check(
                            st.session_state.chain, 
                            st.session_state.vector_db, 
                            prompt
                        )
                        answer = result["answer"]
                        st.markdown(answer)
                        
                        if result["source_documents"]:
                            with st.expander("📖 查看参考文档"):
                                for i, doc in enumerate(result["source_documents"], 1):
                                    st.markdown(f"**[{i}]** {doc.page_content}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"问答失败：{e}\n请确保 Ollama 服务正常运行并已下载所需模型！")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"问答失败：{e}"
                })


if __name__ == "__main__":
    main()
