# Assignment 1: LangChain + Gemini Wikipedia agent
A small AI agent that answers a question by looking it up on Wikipedia. It uses LangChain to build the agent and Google's Gemini API as the language model. It is the base for Assignment 1.

[!DISCLAIMER]. The notebook was written for Google Colab and runs well there. LangChain changes quickly, so running it in PyCharm (or any local Python setup) can need changes, mainly in how the API key is loaded and which LangChain version is installed. 


#UPDATES FROM ORIGINAL NOTEBOOK

#	Area	Original	Updated
1.	Install	google-generativeai	google-genai, plus langgraph
2.	API setup	genai.configure(api_key=...)	Removed; the key is passed straight to the model. The key is read from Colab secrets or an environment variable.
3.	Agent import	from langchain.agents import initialize_agent, AgentType	from langchain.agents import create_agent, with a fallback for older versions
4.	Model	gemini-1.5-flash-latest	gemini-3.8-flash
5.	Building the agent	initialize_agent(..., agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)	create_agent(llm, tools=[wikipedia_tool])
6.	Running the agent	agent.run("...")	agent.invoke(...) with a system message, then a printed trace of each message


# 1. Imports and libraries
- %pip install langchain wikipedia google-generativeai langchain-community langchain-google-genai
+ %pip install langchain wikipedia google-genai langchain-community langchain-google-genai langgraph

Google has two Python SDKs. The old one, google-generativeai, is end-of-life. 
The new one is google-genai, imported as from google import genai which have different functions
langgraph is added so the fallback import in change 3 works on older LangChain versions. 

### LangChain 1.x installs it automatically.
- import google.generativeai as genai
- from google.colab import userdata

### Load the API key from Colab secrets
- GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

### Configure the Gemini API
- genai.configure(api_key=GOOGLE_API_KEY)
+ import os

### Load the API key: from Colab secrets when running in Colab, otherwise from the GOOGLE_API_KEY environment variable
+ try:
+     from google.colab import userdata
+     GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
+ except ImportError:
+     GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
genai.configure(...) belongs to the old SDK. With the new SDK it fails with AttributeError: module 'google.genai' has no attribute 'configure'.

It is not needed anyway: the key is passed straight to the model in the agent cell (google_api_key=GOOGLE_API_KEY).
google.colab only exists in Colab. The try/except uses the Colab secret there and the GOOGLE_API_KEY environment variable everywhere else, so the same cell runs in both places.

# 2. Model selection: Gemini 3.8 Flash
- llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GOOGLE_API_KEY)
+ llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
The 1.5 family is retired. If you get a 404, list the models your key can use:
from google import genai
client = genai.Client(api_key=GOOGLE_API_KEY)
for m in client.models.list():
    if "flash" in m.name:
        print(m.name)

A 503 UNAVAILABLE ("high demand") is a different problem: the model name is fine and Google's servers are busy. Retry later.

# 3. LangChain's new call for building an AI agent
- from langchain.agents import initialize_agent, AgentType
try:
+     from langchain.agents import create_agent  # LangChain / LangGraph 1.x
+ except ImportError:
+     from langgraph.prebuilt import create_react_agent as create_agent  # older versions

- agent = initialize_agent(
-     tools=[wikipedia_tool],
-     llm=llm,
-     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
-     verbose=True
- )
+ agent = create_agent(llm, tools=[wikipedia_tool])
initialize_agent and AgentType were deprecated, and they are no longer in langchain.agents in LangChain 1.x. They moved to a separate package, langchain-classic.
create_agent is LangChain 1.x's replacement and is built on LangGraph. In earlier LangGraph versions the same function was create_react_agent in langgraph.prebuilt; LangGraph 1.x now warns that it moved to langchain.agents.

The try/except imports whichever exists, so the cell runs on both version families.
The old agent made the model write "Action:" text that LangChain had to parse. The new one uses the model's native tool calling: the model returns a structured tool call.
 
# 4. Running the agent and reading the result
- agent.run("What is the capital of France?")
+ messages = [
+     ("system", "Answer using Wikipedia. Always call the wikipedia tool before answering, and name the page you used."),
+     ("user", "What is the capital of France?"),
+ ]
+ result = agent.invoke({"messages": messages})
+
### Gemini can return its text as a list of content blocks, so pull out just the text
+ def message_text(m):
+     if isinstance(m.content, str):
+         return m.content
+     return "".join(b if isinstance(b, str) else b.get("text", "") for b in m.content)

### Print the trace (what verbose=True used to show), then the final answer
+ for m in result["messages"]:
+     print(f"[{m.type}] {message_text(m)[:300]}")
+     for call in getattr(m, "tool_calls", []):
+         print(f"    tool call: {call['name']}({call['args']})")
+ message_text(result["messages"][-1])

Chain.run was deprecated. The new agent is called with .invoke, and its input and output are a list of messages.
There is no verbose=True. The returned list of messages is the trace of what the agent did, so the cell prints one line per message and each tool call.
System message. The new agent lets the model decide whether to call a tool. In a Colab run, Gemini answered "capital of France" from memory and never called Wikipedia. The system message tells it to use the tool.

message_text. Gemini returns its reply as a list of content blocks (with a long signature string attached), not a plain string. Printing .content directly shows the whole raw list, and the cell's last line would print it a second time. The helper keeps just the text.
Expected output (the wording of the last line will vary with Gemini):

[system] Answer using Wikipedia. Always call the wikipedia tool before answering, and name the page you used.
[human] What is the capital of France?
[ai]
    tool call: wikipedia({'query': 'capital of France'})
[tool] Page: List of capitals of France
Summary: This is a chronological list of capitals of France. The capital of France has been Paris since its liberation in 1944.
[ai] The capital of France has been Paris since its liberation in 1944.
