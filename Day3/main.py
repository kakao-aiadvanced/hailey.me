from init import *
from pprint import pprint
from typing import List
import streamlit as st

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

### State

st.set_page_config(
    page_title="Research Assistant",
    page_icon=":orange_heart:",
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        did_web_search: whether web search was used
        hallucinfation_again: whether hallucination check was run again
        was_hallucination: whether hallucination was detected
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]
    did_web_search: bool
    hallucination_again: bool
    was_hallucination: bool


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    did_web_search = False
    hallucination_again = False
    was_hallucination = False
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "did_web_search": did_web_search, "hallucination_again": hallucination_again, "was_hallucination": was_hallucination}



def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    did_web_search = state["did_web_search"]
    hallucination_again = state["hallucination_again"]
    was_hallucination = state["was_hallucination"]

    # Prepare context with source information and collect unique sources
    context_with_sources = []
    unique_sources = set()
    rag_sources = set()
    web_sources = set()
    
    for i, doc in enumerate(documents):
        # Extract source information from document metadata
        source = "Unknown Source"
        if hasattr(doc, 'metadata') and doc.metadata:
            if 'source' in doc.metadata:
                source = doc.metadata['source']
            elif 'url' in doc.metadata:
                source = doc.metadata['url']
        
        # Categorize sources
        if source in RAG_URLS:
            rag_sources.add(source)
        elif source.startswith('http'):
            web_sources.add(source)
        
        unique_sources.add(source)
        
        # Format context with source
        doc_text = f"Document {i+1} (Source: {source}):\n{doc.page_content}\n"
        context_with_sources.append(doc_text)
    
    # Join all contexts
    formatted_context = "\n".join(context_with_sources)

    # RAG generation
    generation = rag_chain.invoke({"context": formatted_context, "question": question})
    
    # Add proper source listing to the generation
    if unique_sources and len(unique_sources) > 0:
        # Remove any existing sources section from generation
        if "Sources:" in generation:
            generation = generation.split("Sources:")[0].strip()
        
        # Add new sources section
        generation += "\n\nSources:\n"
        
        # Add RAG sources (limit to one per unique URL)
        for source in sorted(rag_sources):
            generation += f"- [RAG Source: {source}]\n"
        
        # Add web search sources
        for source in sorted(web_sources):
            generation += f"- [Web Source: {source}]\n"
    
    return {"documents": documents, "question": question, "generation": generation, "did_web_search": did_web_search, "hallucination_again": hallucination_again, "was_hallucination": was_hallucination}


def relevance(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    did_web_search = state["did_web_search"]
    hallucination_again = state["hallucination_again"]
    was_hallucination = state["was_hallucination"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "did_web_search": did_web_search, "hallucination_again": hallucination_again, "was_hallucination": was_hallucination}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = None
    hallucination_again = state["hallucination_again"]
    was_hallucination = state["was_hallucination"]
    if "documents" in state:
      documents = state["documents"]

    # Web search
    docs = tavily.search(query=question)['results']

    # Filter out URLs that are already in RAG system
    filtered_docs = []
    for doc in docs:
        if doc["url"] not in RAG_URLS:
            filtered_docs.append(doc)
        else:
            print(f"---EXCLUDED RAG URL: {doc['url']}---")

    # Create individual Documents for each web search result with metadata (limit to 3)
    web_docs = []
    for doc in filtered_docs[:2]:  # Limit to maximum 2 web search results
        web_doc = Document(
            page_content=doc["content"],
            metadata={
                "url": doc["url"],
                "title": doc["title"],
                "score": doc["score"],
                "source": doc["url"]  # Add source for consistency with existing source handling
            }
        )
        web_docs.append(web_doc)
    
    # Append web search results to existing documents
    if documents is not None:
        documents.extend(web_docs)
    else:
        documents = web_docs
    
    print(web_docs)
    return {"documents": documents, "question": question, "did_web_search": True, "hallucination_again": hallucination_again, "was_hallucination": was_hallucination}


def hallucination(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, was_hallucination, that indicates whether hallucination was detected
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    was_hallucination = state["was_hallucination"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]
    # Check hallucination
    if grade != "yes":
        if was_hallucination:
            return {"documents": documents, "question": question, "generation": generation, "hallucination_again": True, "was_hallucination": True}
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return {"documents": documents, "question": question, "generation": generation, "hallucination_again": False, "was_hallucination": True}
    else:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return {"documents": documents, "question": question, "generation": generation, "hallucination_again": False, "was_hallucination": False}


### Nodes (기존 노드들 유지하고 새로운 노드들 추가)

def fail_not_relevant(state):
    """
    Set generation to failed due to irrelevant documents
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updated state with failure message
    """
    print("---FAIL: NOT RELEVANT---")
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": "failed: not relevant",
        "did_web_search": state["did_web_search"],
        "hallucination_again": state["hallucination_again"],
        "was_hallucination": state["was_hallucination"]
    }

def fail_hallucination(state):
    """
    Set generation to failed due to repeated hallucination
    
    Args:
        state (dict): The current graph state
        
    Returns:
        state (dict): Updated state with failure message
    """
    print("---FAIL: HALLUCINATION---")
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": "failed: hallucination",
        "did_web_search": state["did_web_search"],
        "hallucination_again": state["hallucination_again"],
        "was_hallucination": state["was_hallucination"]
    }


### Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    
    if state["web_search"] == "Yes":
        if state["did_web_search"]:
            print("---DECISION: FAIL - NOT RELEVANT---")
            return "fail_not_relevant"
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


### Conditional edge

def decide_to_generate_after_hallucination(state):
    """
    Determines whether to generate an answer again, or answer to user

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    
    if state["hallucination_again"]:
        print("---DECISION: FAIL - HALLUCINATION---")
        return "fail_hallucination"
    
    if state["was_hallucination"]:
        return "hallucination"
    else:
        return "end"
        

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("relevance", relevance)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("hallucination", hallucination)  # hallucination checker
workflow.add_node("fail_not_relevant", fail_not_relevant)  # failure due to irrelevant docs
workflow.add_node("fail_hallucination", fail_hallucination)  # failure due to hallucination

# Build graph
workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "relevance")
workflow.add_conditional_edges(
    "relevance",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
        "fail_not_relevant": "fail_not_relevant",
    },
)
workflow.add_edge("websearch", "relevance")
workflow.add_edge("generate", "hallucination")
workflow.add_conditional_edges(
    "hallucination",
    decide_to_generate_after_hallucination,
    {
        "hallucination": "generate",
        "end": END,
        "fail_hallucination": "fail_hallucination",
    },
)
workflow.add_edge("fail_not_relevant", END)
workflow.add_edge("fail_hallucination", END)

# Compile
app = workflow.compile()


st.title("Research Assistant powered by OpenAI")

input_topic = st.text_input(
    "Enter a topic",
    value="What is prompt?",
)

generate_report = st.button("Generate Report")

if generate_report:
    with st.spinner("Generating Report"):
        inputs = {"question": input_topic}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Finished running: {key}:")
        final_report = value["generation"]
        
        # Check if final_report starts with "failed" and display in red
        if final_report.startswith("failed"):
            st.markdown(f'<p style="color: red;">{final_report}</p>', unsafe_allow_html=True)
        else:
            st.markdown(final_report)
        print(final_report)

st.sidebar.markdown("---")
if st.sidebar.button("Restart"):
    st.session_state.clear()
    st.rerun()