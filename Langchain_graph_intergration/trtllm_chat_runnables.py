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
            print(f"\nModel output: {content}")  # Debug print
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

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Create a Runnable that returns structured output matching the schema."""
        """
        def parse_output(output: AIMessage) -> Optional[Union[Dict, BaseModel]]:
            try:
                # Get raw text and clean it
                text = output.content.strip().lower()
                print(f"\nParsing text: {text}")  # Debug print
                
                # For Literal type fields
                if hasattr(schema, 'model_fields'):
                    field_info = next(iter(schema.model_fields.values()))
                    if hasattr(field_info.annotation, '__origin__') and field_info.annotation.__origin__ is Literal:
                        allowed_values = field_info.annotation.__args__
                        print(f"Allowed values: {allowed_values}")  # Debug print
                        
                        # Try exact match first
                        if text in allowed_values:
                            return schema(**{next(iter(schema.model_fields.keys())): text})
                            
                        # Then try to find one of the allowed values in the text
                        for value in allowed_values:
                            if value in text:
                                print(f"Found value: {value}")  # Debug print
                                return schema(**{next(iter(schema.model_fields.keys())): value})
                
                print("No matching value found")  # Debug print
                return None
                
            except Exception as e:
                print(f"Error parsing output: {str(e)}")
                print(f"Raw output was: {text}")
                return None
        """

        def parse_output(output: AIMessage) -> Optional[Union[Dict, BaseModel]]:
            try:
                # Get raw text and clean it
                text = output.content.strip().lower()
                print(f"\nParsing text: {text}")  # Debug print
                
                # Use regex to extract the core value (strip prefixes like 'answer:' or 'evaluation:')
                match = re.search(r"\b(yes|no|1|2|3|4|5)\b", text)
                if match:
                    extracted_value = match.group(0)
                    print(f"Extracted value: {extracted_value}")  # Debug print
                    return schema(**{next(iter(schema.model_fields.keys())): extracted_value})
                
                print("No matching value found")  # Debug print
                return None
                
            except Exception as e:
                print(f"Error parsing output: {str(e)}")
                print(f"Raw output was: {text}")
                return None

        

        # Just return the chain that converts LLM output to structured format
        chain = self | parse_output
        
        if include_raw:
            raise NotImplementedError(
                "include_raw=True is not supported. Consider using the structured response "
                "being None when the LLM produces an incomplete response."
            )

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
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n Score:"),
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
print(f"Grading result: {result.binary_score if result else 'Failed to parse'}")

# Define route query schema
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "not_retrieve"] = Field(
        description="Given a user question, choose to route it to a vectorstore or not_retrieve."
    )

# Create structured router
structured_llm_router = llm.with_structured_output(RouteQuery)

# Define routing prompt
system_prompt = """You are an expert at routing user questions.
For questions about KARI (Korea Aerospace Research Institute), including:
- aircraft, satellites, space vehicles
- satellite imagery and navigation
- space exploration
- aerospace research and development
Output EXACTLY 'vectorstore' (no quotes).

For any other topics, output EXACTLY 'not_retrieve' (no quotes).

Output only one word, either 'vectorstore' or 'not_retrieve'. 
Do not include any other text, explanations, or punctuation."""

# Create prompt template
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

# Create router chain
question_router = route_prompt | structured_llm_router

# Test examples
test_questions = [
    "Tell me about KARI",
    "What's your favorite color?",
    "How does KARI develop satellites?",
    "What's the weather like today?"
]

for question in test_questions:
    print(f"\n=== Testing: {question} ===")
    result = question_router.invoke({"question": question})
    if result:
        print(f"Route to: {result.datasource}")
    else:
        print("Failed to parse response")




