from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

t1 = PromptTemplate(template="detailed analysis of {topic}", input_variables=["topic"])

t2 = PromptTemplate(template="summary of {text} in 10 words", input_variables=["text"])

parser = StrOutputParser()

chain = t1 | llm | parser | t2 | llm | parser

# result = chain.invoke({"topic": "godrej"})
result = llm.invoke("write a few lines about f1")

print(result)
# print(chain.get_graph().draw_ascii())