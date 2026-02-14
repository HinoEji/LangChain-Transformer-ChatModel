import json
import re
from typing import Any, Dict, List, Optional, Union, Callable
from abc import ABC, abstractmethod
import inspect
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel,BitsAndBytesConfig
import torch
from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage,ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from collections.abc import Callable, Mapping, Sequence
from langchain_core.language_models import (
    LanguageModelInput
)
from langchain_core.runnables import Runnable

from ..utils import (
    convert_lc_messages_to_hf_messages,
    convert_hf_messages_to_lc_messages,
    create_tool_call,
    ToolCallParser
)

class TransformerChatModel(BaseChatModel):

    # for creating model
    pretrained_model_name_or_path: str
    device : str = "auto"
    attn_implementation : str|None = None
    quantization_config : Any = None
    torch_dtype : Any = "auto"
    hf_token : str|None = None # some model need authorized
    # tools
    bound_tools : List[Any] = []

    # for generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    additional_generation_kwargs: dict[str, Any] = {}

    # other
    model: Any = None
    tokenizer : Any = None
    max_context_length : int = None


    def __init__(self, *, model = None, tokenizer = None, **kwargs: Any):
        # initial pydantic
        super().__init__(**kwargs)

        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        else:
            torch_dtype = self.torch_dtype
            if isinstance(torch_dtype, str):
                if torch_dtype == "float16":
                    torch_dtype = torch.float16
                elif torch_dtype == "float32":
                    torch_dtype = torch.float32
                elif torch_dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif torch_dtype == "auto":
                    torch_dtype = "auto"
                else:
                    raise ValueError(f"Invalid torch_dtype: {torch_dtype}")
            elif not isinstance(torch_dtype, torch.dtype):
                raise ValueError(f"Invalid torch_dtype: {torch_dtype}")

            load_model_config = {
                "pretrained_model_name_or_path" : self.pretrained_model_name_or_path,
                "torch_dtype" : torch_dtype,
                "device_map" : self.device,
                "attn_implementation" : self.attn_implementation,
                "quantization_config" : self.quantization_config,
                "token" : self.hf_token
            }
            print("Loading Model from hugginface...\n")
            self.model = AutoModelForCausalLM.from_pretrained(**load_model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # set max_context_length
        if self.max_context_length is None:
            try:
                self.max_context_length = self.model.config.max_position_embeddings
            except :
                self.max_context_length = 1024

    @property
    def _generation_config(self) -> dict:
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **self.additional_generation_kwargs
        }
        return generation_config

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:

        tool_instruction = self.get_tools_desc()
        if tool_instruction:
            messages = [SystemMessage(content=tool_instruction)] + messages
        # convert messages to prompt
        hf_msg = convert_lc_messages_to_hf_messages(messages)
        # tokenize
        inputs = self.tokenizer.apply_chat_template(
            hf_msg,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length,
            return_dict = True
        ).to(self.model.device)

        generation_config = self._generation_config
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # speed up, tool call parsing is time consuming
        if self.bound_tools:
            # parse tool call
            tool_calls = ToolCallParser.extract_tool_calls(response)
            if tool_calls:
                msg = AIMessage(content="", tool_calls = tool_calls)
            else: 
                msg = AIMessage(content = response)
        else:
            msg = AIMessage(content = response)
        # post-process to extract tool_call here
        # CODE
        return ChatResult(generations=[ChatGeneration(message=msg)])



    def get_tools_desc(self) -> str:
        """
        Return all tools description in a string.
        The tool must be converted into openai format.
        Each description is converted into format:

        name :
        docstring.
        params:
            * param1 (type)
            * param2 (type)

        """

        tools_desc = ""
        for tool in self.bound_tools:
            tool_info = tool["function"]
            tool_name = tool_info["name"]
            tool_desc = tool_info.get("description", "")
            if tool_desc:
                tool_desc = "\n\t"+tool_desc
            params_desc = []
            for param_name, param_info in tool_info["parameters"].get("properties",{}).items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                additional_info = ";".join([f"{k}: {v}" for k,v in param_info.items() if k != "type" and k != "description"])
                if param_desc or additional_info:
                    param_desc = f"\t\t{param_name} ({param_type}) -- {param_desc} {additional_info}"
                else:
                    param_desc = f"\t\t{param_name} ({param_type})"
                params_desc.append(param_desc)

            if not params_desc:
                params_desc = "This tool don't have parameters"
            else:
                params_desc = "\n".join(params_desc)

            tool_desc = f"""**{tool_name}**:{tool_desc}\n\tparams:\n{params_desc}\n\n"""
            tools_desc += tool_desc

        if tools_desc:
            return f"""You have access to the following tools :

    {tools_desc}

    Whenever you need to use a tool, you MUST respond with a JSON code block in this exact format:
    ```tool_call_block
    {{
    "tool_calls": [
        {{
            "name": "tool_name",
            "arguments": {{"param1": "value1", "param2": "value2"}}
        }}
    ]
    }}
    ```
    Only use provided tools, do not invent new ones.
    If you don't need to use any tools, respond normally without the JSON block.
    Tool responses are provided with a tool_call_id.
    Always match each responese to the corresponding tool call using this ID and use the matched results to produce the final answer.
    A user query may contain multiple sub-questions.
    Use tools only for sub-questions that require them, answer the rest directly, and merge all results into the final response.
    Do not explain how you use tools.
    """
        return ""


    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:

        # convert tools to open_ai format
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        # I add tool_choice so as model can work with agent
        # (creating agent with tool will call model.bind_tool with too_choice)
        # I want all my tools can be used
        model_with_tools = self.__class__(
            model = self.model,
            tokenizer = self.tokenizer,
            pretrained_model_name_or_path = self.pretrained_model_name_or_path,
            device = self.device,
            attn_implementation = self.attn_implementation,
            quantization_config = self.quantization_config,
            torch_dtype = self.torch_dtype,
            hf_token = self.hf_token,
            bound_tools = formatted_tools,
            max_new_tokens = self.max_new_tokens,
            temperature = self.temperature,
            do_sample = self.do_sample,
            **self.additional_generation_kwargs
        )

        return model_with_tools


    @property
    def _llm_type(self) -> str:
        return "transformer-chat-model"

