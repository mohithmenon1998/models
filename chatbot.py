from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

Messages = [SystemMessage(content="You are a helpful AI Assistant")]

while True:
    print("chat with me or press q to go away\n")
    user_input = input("YOU: ")
    Messages.append(HumanMessage(content=user_input))
    
    if user_input == 'q':
        print("bye see u again\n")
        break
    else:
        result = llm.invoke(Messages)
        Messages.append(AIMessage(content=result.content))
        print("AI: ", result.content, "\n")
        
print(Messages)