from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash-lite")

parser = StrOutputParser()

# topic fetcher

class NoteTopic(BaseModel):
    topic : str = Field(description="Fetch the medical topic mentioned in the text prompt")

# parallel chains
topic_prompt = model.with_structured_output(NoteTopic)

text = """Surgery: This is easier to revise closer to the exam due to its factual nature, but many consider it challenging to master all the detailed procedural aspects and surgical conditions."""

pll_promtpt1 = PromptTemplate(template="You are a harvard medical professor with years of experience in FMGE and NEET exam topics in depth, also you are well versed in prepladder study materials, based on your expertise write detailed notes on {topic}", input_variables=["topic"])

pll_promtpt2 = PromptTemplate(template="You are a FMGE/NEET medical exam question setter, based on the topic and with format of previous years question, create a test with easy, medium, hard and extreme level questions which will cover all the point of the topic, also add a solutions section with short explanation, The aim is to create a test that touch all important topic and is in the FMGE or NEET exam format and will be a great revision for the candidate. Topic --> {topic}", input_variables=["topic"])

pll_promtpt3 = PromptTemplate(template="combine the notes and quiz into one single document. notes --> {notes}, quiz --> {quiz}", input_variables=["quiz", "notes"])

notes_chain = pll_promtpt1 | model | parser 
quiz_chain = pll_promtpt2 | model | parser

pll_chain = RunnableParallel({"notes": notes_chain, "quiz": quiz_chain})

final_pll_chain = topic_prompt | pll_chain | pll_promtpt3 | model | parser 

def medical_assistant(prompt: str):
    return final_pll_chain.invoke(prompt)

    