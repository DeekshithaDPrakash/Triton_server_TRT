import json
import requests
from typing import List, Optional, Dict, Any, Iterator
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage, ChatResult, ChatGeneration
from langchain.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class TRTLLMChat(BaseChatModel):
    """Chat model for TensorRT-LLM running on Triton Inference Server"""
    
    url: str = Field(..., description="URL of the Triton inference server endpoint")
    temperature: float = Field(0.0, description="Sampling temperature")
    max_tokens: int = Field(4096, description="Maximum number of tokens to generate")
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "trt-llm-chat"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a prompt string"""
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"System: {message.content}\n"
            elif isinstance(message, HumanMessage):
                prompt += f"Human: {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"Assistant: {message.content}\n"
        return prompt.strip()
    
    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> str:
        """Execute the chat completion"""
        prompt = self._convert_messages_to_prompt(messages)
        
        # Keep numeric values as numbers
        payload = {
            "text_input": prompt,
            "parameters": {
                "temperature": float(self.temperature),  # Ensure it's a float
                "max_tokens": int(self.max_tokens)      # Ensure it's an integer
            }
        }
        
        # Only add stop if provided
        if stop and len(stop) > 0:
            payload["parameters"]["stop"] = stop[0]
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Error from Triton server: {response.text}")
            
            result = response.json()
            
            # Debug print
            print(f"Server response: {result}")
            
            return result["text_output"]
            
        except Exception as e:
            print(f"Request payload: {json.dumps(payload, indent=2)}")
            raise e

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion result."""
        text = self._call(messages, stop)
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def with_structured_output(self, cls: type) -> BaseChatModel:
        """Create a structured output parser"""
        parent = self
        
        class StructuredTRTLLMChat(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return parent._llm_type

            def _generate(
                self,
                messages: List[BaseMessage],
                stop: Optional[List[str]] = None,
                run_manager: Optional[Any] = None,
                **kwargs: Any,
            ) -> ChatResult:
                result = parent._call(messages, stop)
                try:
                    parsed = cls.model_validate_json(result)
                except:
                    parsed = cls.model_validate({"binary_score": result.strip().lower()})
                
                message = AIMessage(content=str(parsed))
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])

            @property
            def _identifying_params(self) -> Dict[str, Any]:
                return parent._identifying_params

        return StructuredTRTLLMChat()

# Define system prompt
system_prompt = """You are a grader assessing whether an LLM generation is grounded in supported by a set of retrieved facts.  
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in supported by the set of facts.
ONLY output 'yes' or 'no' without any explanation or additional text. 
Do NOT include any other words or sentences in your answer."""

# Initialize the LLM
llm = TRTLLMChat(
    url="http://ip:port/v2/models/ensemble/generate",
    temperature=0,
    max_tokens=8096
)

# Create the structured output grader
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Create the prompt template
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation} \n\n Score:"),
])

# Create the grading chain
hallucination_grader = hallucination_prompt | structured_llm_grader
