import os
from vector_db import load_documents_from_folder, create_vector_db


def main():
    print("="*60)
    print("构建本地知识库")
    print("="*60)
    
    folder_path = input("\n请输入文档文件夹路径: ").strip()
    
    if not os.path.exists(folder_path):
        print(f"\n错误：路径 '{folder_path}' 不存在！")
        return
    
    print("\n正在加载文档...")
    documents = load_documents_from_folder(folder_path)
    
    if not documents:
        print("\n未找到PDF或DOCX文档！")
        return
    
    print(f"已加载 {len(documents)} 个文档")
    
    print("\n正在构建向量数据库...")
    vector_db, chunk_count = create_vector_db(documents)
    
    print(f"\n知识库构建完成！")
    print(f"共生成 {chunk_count} 个文本块")
    print(f"数据库保存在 ./chroma_db")


if __name__ == "__main__":
    main()
