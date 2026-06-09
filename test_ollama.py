from langchain_community.llms import Ollama

def test_ollama_connection():
    print("="*50)
    print("测试 Ollama 连接...")
    print("="*50)
    
    try:
        llm = Ollama(model="deepseek-r1:7b")
        response = llm.invoke("你好，请简单介绍一下你自己。")
        print("\nOllama 连接成功！")
        print("\n模型回复：")
        print("-"*50)
        print(response)
        print("-"*50)
        return True
    except Exception as e:
        print(f"\n连接失败：{e}")
        print("\n请确保：")
        print("1. Ollama 服务已启动")
        print("2. 已下载 deepseek-r1:7b 或 qwen2:7b 模型")
        print("3. 模型名称正确")
        return False

if __name__ == "__main__":
    test_ollama_connection()
