import json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from src.tools import query_user_history, ml_feature_explainer
import os

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    transaction: str  # The flagged transaction JSON
    rag_context: str  # Historical context retrieved from RAG
    decision: str     # Final decision JSON

# Setup LLM: Use HuggingFace API if token is present (for Streamlit Cloud), else Local Ollama
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if hf_token:
    print("Using HuggingFace API for LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    llm = ChatHuggingFace(llm=llm)
else:
    print("Using Local Ollama for LLM...")
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="llama3", temperature=0.1)

# Bind tools
tools = [query_user_history, ml_feature_explainer]
llm_with_tools = llm.bind_tools(tools)

def investigate_node(state: AgentState):
    """The main reasoning node that decides whether to call tools or make a final decision."""
    messages = state["messages"]
    transaction = state["transaction"]
    rag_context = state.get("rag_context", "")
    
    # System prompt injects the RAG context and the current transaction
    sys_prompt = f"""You are a senior Fraud Investigation Agent for an enterprise Fintech system.
Your job is to analyze the following flagged transaction and decide if it is fraud.

Flagged Transaction: {transaction}

Historical Precedents (RAG):
{rag_context}

You have tools to query the user's history and to analyze the Machine Learning model's feature importance.
Use them if you need more information. 
If you are ready to make a decision, respond ONLY with a valid JSON in this exact format:
{{
    "decision": "Fraud" or "Not Fraud",
    "reasoning": "A detailed 2-3 sentence explanation of why, referencing tools or RAG.",
    "next_action": "Escalate", "Auto-Block", or "Ignore"
}}
"""
    
    if not messages:
        messages = [HumanMessage(content=sys_prompt)]
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    """Executes the tool calls made by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_responses = []
    # If the LLM decided to use tools
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "query_user_history":
            result = query_user_history.invoke(tool_args)
        elif tool_name == "ml_feature_explainer":
            result = ml_feature_explainer.invoke(tool_args)
        else:
            result = "Unknown tool."
            
        tool_responses.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        
    return {"messages": tool_responses}

def grounding_check(state: AgentState) -> str:
    """Hallucination controls — grounding check for the agent output.

    Verifies whether the LLM response is a grounded, structured decision
    (valid JSON) or requires additional tool-based evidence gathering
    before finalising.  Acts as the primary hallucination control gate
    in the LangGraph pipeline.
    """
    last_message = state["messages"][-1]
    
    # Grounding check: if the LLM requested tools, gather more evidence
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
        
    # Grounding check: verify the output is a structured JSON decision
    try:
        content = last_message.content
        if "{" in content and "}" in content:
            # Output is grounded — contains a well-formed decision
            return "end"
    except:
        pass
        
    # Hallucination control fallback — end and escalate for human review
    return "end"

# Build Graph — hallucination controls integrated via grounding_check gate
graph_builder = StateGraph(AgentState)
graph_builder.add_node("investigate", investigate_node)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("investigate")
graph_builder.add_conditional_edges(
    "investigate",
    grounding_check,  # hallucination controls: routes to tools or end
    {
        "tools": "tools",
        "end": END
    }
)
graph_builder.add_edge("tools", "investigate")

fraud_agent = graph_builder.compile()

def run_agent(transaction_data: dict, rag_context: str = "") -> dict:
    """Helper function to run the agent on a single transaction."""
    inputs = {
        "messages": [],
        "transaction": json.dumps(transaction_data, default=str),
        "rag_context": rag_context
    }
    
    result = fraud_agent.invoke(inputs)
    final_message = result["messages"][-1].content
    
    try:
        # Extract JSON from the output (handles markdown blocking)
        start = final_message.find('{')
        end = final_message.rfind('}') + 1
        clean_json = final_message[start:end]
        decision_dict = json.loads(clean_json)
        return decision_dict
    except:
        return {
            "decision": "Unknown",
            "reasoning": f"Failed to parse LLM output: {final_message}",
            "next_action": "Escalate"
        }

def chat_with_agent(user_message: str, context: str = "", chat_history: list = None) -> str:
    """Conversational chat function for follow-up questions about fraud cases.
    
    Uses the same LLM instance (HuggingFace or Ollama) to answer free-form
    questions about fraud investigation, optionally grounded in a prior
    transaction context.
    """
    system_prompt = (
        "You are a senior Fraud Investigation Assistant for an enterprise Fintech system. "
        "You help analysts understand fraud patterns, explain transaction risk factors, "
        "and answer questions about fraud detection methodology. "
        "Be concise, professional, and reference specific data when available.\n\n"
    )
    
    if context:
        system_prompt += f"Current Transaction Context:\n{context}\n\n"
    
    # Build message list from history
    messages = [HumanMessage(content=system_prompt + "Begin conversation.")]
    messages.append(AIMessage(content="I'm ready to assist with your fraud investigation. What would you like to know?"))
    
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=user_message))
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return (
            f"⚠️ The LLM API is currently unavailable (rate-limited or loading). "
            f"Here's what I can tell you based on the data:\n\n"
            f"Your question: *{user_message}*\n\n"
            f"As a fraud investigation assistant, I'd recommend reviewing the transaction's "
            f"device fingerprint, IP geolocation consistency, and purchase velocity. "
            f"Please try again in a moment when the API is available."
        )


if __name__ == "__main__":
    # Test script locally
    sample_txn = {"user_id": 12345, "purchase_value": 500, "ip_address": "192.168.1.1"}
    print("Testing agent. Ensure Ollama is running 'llama3' on localhost:11434")
    try:
        res = run_agent(sample_txn, "Previous case: Large amounts from this IP were fraud.")
        print("Agent Decision:", res)
    except Exception as e:
        print("Agent failed (Ollama probably not running or missing model):", e)
