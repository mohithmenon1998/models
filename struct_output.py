from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

class Review(BaseModel):
    """ This is used for structuring a Movie review"""
    summary: str = Field(description="The short summary of the movie review in 10 words")
    sentiment: Literal["blockbuster", "housefull", "superhit", "flop"] = Field(description="Sentiment of the review")
    movie_name: str = Field(description="Name of the movie from the review")
    actor: list[str] = Field(description="name of the actor if mentioned in the review")

structured = llm.with_structured_output(Review)

review = '''Social media has been flooded with rave reviews for Dulquer’s intense portrayal in the film. One user wrote, “One hell of a performance! Truly outstanding! @dulQuer #Kaantha Must Watch.”

Another praised the actor’s exceptional effort, tweeting, “WHAT AN ACTOR PEAK PERFORMANCE from @dulQuer Deserved a National Award for his Stunning Performance !! #Kaantha #Kaanthareview.”

'''

result = structured.invoke(review)

print(result)