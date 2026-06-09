from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from vector_db import load_vector_db, retrieve_similar_documents


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


def create_rag_chain(vector_db, model_name="deepseek-r1:7b"):
    llm = Ollama(model=model_name)
    
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


def ask_question(chain, vector_db, question):
    """先检查检索结果，再决定如何回答"""
    # 先直接检索看看有没有相关文档
    docs = retrieve_similar_documents(vector_db, question, k=3)
    
    # 如果没有检索到任何文档，直接返回固定答案，不调用模型！
    if len(docs) == 0:
        return {
            "answer": "文档中未找到相关答案",
            "source_documents": []
        }
    
    # 只要检索到了文档，就直接调用大模型，相信系统提示词！
    result = chain({"question": question})
    
    return {
        "answer": result["answer"],
        "source_documents": result["source_documents"]
    }


def main():
    print("="*60)
    print("RAG 智能问答系统（命令行版）")
    print("="*60)
    
    vector_db = load_vector_db()
    if not vector_db:
        print("\n未找到向量数据库！请先构建知识库。")
        return
    
    chain = create_rag_chain(vector_db)
    
    print("\n知识库已加载，开始对话！输入 'quit' 退出。\n")
    
    while True:
        question = input("你: ").strip()
        
        if question.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break
        
        if not question:
            continue
        
        result = ask_question(chain, vector_db, question)
        
        print(f"\n助手: {result['answer']}\n")
        
        print("参考文档：")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"  [{i}] {doc.page_content[:100]}...")
        print("-"*60)


if __name__ == "__main__":
    main()
