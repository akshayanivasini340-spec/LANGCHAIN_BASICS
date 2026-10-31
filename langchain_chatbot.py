from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from datetime import datetime

# Connect to Ollama
llm = ChatOllama(model="llama3.2:latest", temperature=0.7)

# Five Prompt Templates
prompts = {
    "Admission": "Explain college admission details about {question}.",
    "Course": "Explain course information about {question}.",
    "Examination": "Help students with examination details about {question}.",
    "Library": "Answer the library-related question: {question}.",
    "Activities": "Explain student activities about {question}."
}

# Save prompts
pd.DataFrame(
    list(prompts.items()),
    columns=["Category", "Prompt"]
).to_csv("dataset/prompts.csv", index=False)

# Get user input
print("\nCOLLEGE ASSISTANT CHATBOT")
print("Categories:", list(prompts.keys()))

category = input("\nEnter category: ").strip().title()
question = input("Enter your question: ")

if category in prompts:

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful College Assistant. " + prompts[category]
    )

    chain = prompt | llm

    response = chain.invoke({"question": question})

    answer = response.content

    print("\nChatbot Response:")
    print(answer)

    # Save user query
    query_data = pd.DataFrame([{
        "Category": category,
        "Question": question,
        "Date": datetime.now()
    }])

    query_data.to_csv(
        "dataset/user_queries.csv",
        mode="a",
        header=not pd.io.common.file_exists("dataset/user_queries.csv"),
        index=False
    )

    # Save chatbot response
    response_data = pd.DataFrame([{
        "Category": category,
        "Response": answer
    }])

    response_data.to_csv(
        "dataset/chatbot_responses.csv",
        mode="a",
        header=not pd.io.common.file_exists("dataset/chatbot_responses.csv"),
        index=False
    )

    print("\nData saved successfully!")

else:
    print("Invalid category!")
# SUMMARIZATION CHAIN

summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following text in simple words:\n{text}"
)

summary_chain = summary_prompt | llm

text = """
A college library provides books, newspapers,
journals, computers and a quiet study environment.
Students can borrow books and use digital resources
for their academic activities.
"""

summary_result = summary_chain.invoke({"text": text})

print("\nSUMMARIZATION RESULT:")
print(summary_result.content)


# QUESTION ANSWERING CHAIN

qa_prompt = ChatPromptTemplate.from_template(
    "Answer the question using the given context.\n"
    "Context: {context}\n"
    "Question: {question}"
)

qa_chain = qa_prompt | llm

qa_result = qa_chain.invoke({
    "context": text,
    "question": "What facilities does the college library provide?"
})

print("\nQUESTION ANSWERING RESULT:")
print(qa_result.content)


# SAVE CHAIN RESULTS

import pandas as pd

results = pd.DataFrame([
    {
        "Chain": "Summarization",
        "Result": summary_result.content
    },
    {
        "Chain": "Question Answering",
        "Result": qa_result.content
    }
])

results.to_csv("dataset/chain_results.csv", index=False)

print("\nChain results saved successfully!")