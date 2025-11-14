from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
result =llm.invoke("hi")

hf_llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you)
)
chat_model = ChatHuggingFace(llm= hf_llm)

r = chat_model.invoke("Hi")
print(r.content) 