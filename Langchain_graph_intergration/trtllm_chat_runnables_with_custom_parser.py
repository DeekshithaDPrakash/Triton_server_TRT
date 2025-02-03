from typing import Any, Dict, List, Optional, Union, Literal, Type
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
import requests
import json
import re

class YesNoResponse(BaseModel):
    """Binary yes/no response"""
    binary_score: str  =  Field(description="answer must be 'yes' or 'no'")

class IntResponse(BaseModel):
    """Integer scale response"""
    int_score: int = Field(description="Rating must be between 1 and 5")

class FloatResponse(BaseModel):
    """Float scale response."""
    float_score: float = Field(description="Score must be between 0.0 and 1.0")

class CategoryResponse(BaseModel):
    "Menu section reponse"
    category: str =  Field(decription="Selected menu categories")

class TRTLLMChat(BaseChatModel):
    url: str = Field(..., description="URL of the Triton inference server endpoint")
    temperature: float = Field(0.0, description="Sampling temperature") 
    max_tokens: int = Field(4096, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(None, description="Top-p for nucleus sampling")

    @property
    def _llm_type(self) -> str:
        return "trt-llm-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[Union[str, List[str]]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        print(f"\nSending prompt:\n{prompt}")  # Debug print
        
        payload = {
            "text_input": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        response = requests.post(self.url, json=payload)
        print(f"\nResponse status: {response.status_code}")  # Debug print
        
        if response.status_code != 200:
            raise ValueError(f"Error from Triton server: {response.text}")
            
        try:
            content = response.json()["text_output"]
            # print(f"\nModel output: {content}")  # Debug print
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
        except KeyError as e:
            raise ValueError(f"Unexpected response format: {response.text}") from e

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                parts.append(f"System: {message.content.strip()}")
            elif isinstance(message, HumanMessage):
                parts.append(f"Human: {message.content.strip()}")
            elif isinstance(message, AIMessage):
                parts.append(f"Assistant: {message.content.strip()}")
        return "\n".join(parts)
        
    #----------------------YES NO ---------------------------------------------------
    def with_structured_output_yesno(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
    ) -> Runnable[LanguageModelInput, YesNoResponse]:
        def parse_yesno(output: AIMessage) -> Optional[YesNoResponse]:
            try:
                text = output.content.strip().lower()
                match = re.search(r"\b(yes|no)\b", text)
                # print(match.group(1))
                if match:
                    # return match.group(1)
                    return YesNoResponse(binary_score=match.group(1))
                return None
            except Exception as e:
                print(f"Error parsing yes/no output: {str(e)}")
                return None

        chain = self | parse_yesno
        return chain

    #--------------------------INT 1~5 Scores------------------
    def with_structured_output_scale_int(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
    ) -> Runnable[LanguageModelInput, IntResponse]:
        def parse_scale(output: AIMessage) -> Optional[IntResponse]:
            try:
                text = output.content.strip()
                match = re.search(r"\b[1-5]\b", text)
                # print(match)
                if match:
                    # return {"rating": match.group(0)}
                    # return match.group(0)
                    return IntResponse(int_score=match.group(0))
                return None
            except Exception as e:
                print(f"Error parsing scale output: {str(e)}")
                return None

        chain = self | parse_scale
        return chain

    #-------------Retrieving Categories--------------------
    def with_structured_output_menu(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
    ) -> Runnable[LanguageModelInput, CategoryResponse]:
        def parse_menu(output: AIMessage) -> Optional[CategoryResponse]:
            try:
                # print("OUTPUT",output)
                text = output.content.strip()
                # print("TEXT",text)
                menu_options = [
                    "Introduction to KARI",
                    "News",
                    "Research and Development",
                    "KARI TV & IMAGE",
                    "Aerospace Policy",
                    "Technology Commercialization",
                    "Information Disclosure",
                    "ESG Management"
                ]
                # print("Menu",text)
                for option in menu_options:
                    if option.lower() in text.lower():
                        # return option
                        return CategoryResponse(category=option)
                return CategoryResponse(category="not retrieving")
            except Exception as e:
                print(f"Error parsing menu output: {str(e)}")
                return None

        chain = self | parse_menu
        return chain

    #-------------FLOAT 0~1 Scores----------------

    def with_structured_output_scale_float(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
    ) -> Runnable[LanguageModelInput, FloatResponse]:
        def parse_float(output: AIMessage) -> Optional[FloatResponse]:
            try:
                text = output.content.strip()
                match = re.search(r"(\d+\.\d{1,2})", text)
                if match:
                    value = float(match.group(1))
                    if 0 <= value <= 1:
                        # return value
                        return FloatResponse(float_score=value)
                return None
            except Exception as e:
                print(f"Error parsing float output: {str(e)}")
                return None

        chain = self | parse_float
        return chain


# Initialize LLM
llm = TRTLLMChat(
    url="http://ip:port/v2/models/ensemble/generate",
    temperature=0.0
)



# Define schema
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Create the structured output grader
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Define system prompt
system_prompt = """You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts.  
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts.
ONLY output 'yes' or 'no' without any explanation or additional text. 
Do NOT include any other words or sentences in your answer."""

# Create the prompt template
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n"),
    ("assistant", "")
])

# Create the grading chain
hallucination_grader = hallucination_prompt | structured_llm_grader

# Test the chain
documents = "Fact 1: 지구는 둥글다.\nFact 2: 달은 지구를 공전한다."
generation = "지구는 둥글지 않다."

result = hallucination_grader.invoke({
    "documents": documents, 
    "generation": generation
})

print(result)
print(result.binary_score)


# Define route query schema
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "not_retrieve"] = Field(
        description="Given a user question, choose to route it to a vectorstore or not_retrieve."
    )

# Create structured router
structured_llm_router = llm.with_structured_output_menu(RouteQuery)

# Define routing prompt
system_prompt = """You are an expert at routing a user question to the appropriate datasource.
The datasources contain specific information on the following topics:
- 'Introduction to KARI': General overview of the Korea Aerospace Research Institute (KARI).
- 'News': Recent news and updates about KARI.
- 'Research and Development': Topics related to aircraft, unmanned vehicles, satellites, space launch vehicles, satellite imagery, space exploration, and satellite navigation.
- 'KARI TV & IMAGE': Multimedia content, including videos and images related to KARI's activities.
- 'Aerospace Policy': Policies and strategies concerning aerospace initiatives.
- 'Technology Commercialization': Information about the commercialization of technologies developed by KARI.
- 'Information Disclosure': Publicly available data and reports from KARI.
- 'ESG Management': Environmental, Social, and Governance practices at KARI.

Evaluate the question and match it to the most relevant category. If no category is relevant, output 'not retrieving'."""

# Create prompt template
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Create router chain
question_router = route_prompt | structured_llm_router

# Test examples
test_questions = [
    "Tell me about KARI's R&D",
    "What is KARI's policies?",
    "How does KARI develop satellites?",
    "What's the weather like today?",
    "KARI's overview"
]

for question in test_questions:
    print(f"\n=== Testing: {question} ===")
    result = question_router.invoke({"question": question}) 
    print(result)

    
class GradeDocuments(BaseModel):
    """score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question"
    )

# Create the structured grader
structured_llm_grader = llm.with_structured_output_scale_float(GradeDocuments)

# Define the system prompt
system = """You are a grader tasked with assessing the relevance of a retrieved document to a user question.
Evaluate the relevance on a scale of 0.0 to 1.0, where:

0.0 - The document is completely unrelated to the question.
0.2 - The document is mostly unrelated, with only minor or tangential relevance.
0.4 - The document has some relevance but lacks sufficient depth or completeness.
0.6 - The document is moderately relevant, addressing parts of the question but not fully.
0.8 - The document is largely relevant and addresses most aspects of the question.
1.0 - The document is highly relevant and directly addresses the question comprehensively.

Your evaluation should consider both keyword overlap and semantic alignment with the user's question. The primary goal is to distinguish between relevant and irrelevant documents with a precise and fair score. The output must only include the numerical score."""

# Create the grading prompt template
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

# Create the retrieval grader chain
retrieval_grader = grade_prompt | structured_llm_grader

### Note: if any additional text is added after the prompt in human {question}, it returns None

        

