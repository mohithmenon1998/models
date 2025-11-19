from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash-lite")

promtpt1 = PromptTemplate(template="You are an harvard professor write a detailed content in {topic}", input_variables=["subject","topic"])

promtpt2 = PromptTemplate(template="You are a summarizer, summarize this {content} in one line.", input_variables=["content"])

parser = StrOutputParser()

chain = promtpt1 | model | parser | promtpt2 | model | parser

print(chain.get_graph().draw_ascii())

result = chain.invoke({"topic": "march 2008 market crash"})

print(result)
