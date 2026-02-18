import json
import re
from typing import Any, Dict, List, Optional, Union
import inspect
from typing_extensions import override

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import torch

from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import Runnable
from langchain_core.messages import (
    BaseMessage, 
    AIMessage, 
    HumanMessage, 
    SystemMessage,
    ToolMessage,
    AIMessageChunk,
)
from collections.abc import(
    Callable,
    Mapping,
    Sequence,
    Iterator
)
from langchain_core.language_models import (
    LanguageModelInput
)

from ..utils import (
    convert_lc_messages_to_hf_messages,
    convert_hf_messages_to_lc_messages,
    create_tool_call,
    ToolCallParser
)

BASE_DIR = Path(__file__).resolve().parent.parent  # transformer_chat_model/
TEMPLATE_DIR = BASE_DIR / "prompts"

ENV = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)

_WELL_KNOWN_GENERATION_PARAMS = [
    # 2. Length Arguments
    "max_length",
    "max_new_tokens",
    "min_length",
    "min_new_tokens",
    "early_stopping",
    "max_time",
    "stop_strings",

    # 3. Generation Strategy
    "do_sample",
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "use_cache",

    # 4. Logits Manipulation / Sampling
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",

    # 5. Penalties & Constraints
    "repetition_penalty",
    "encoder_repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "bad_words_ids",
    "force_words_ids",
    "constraints",
    "renormalize_logits",

    # 6. Special Tokens
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",

    # # 7. Output Variables
    # "return_dict_in_generate",
    # "output_attentions",
    # "output_hidden_states",
    # "output_scores",
    # "output_logits",

    # # 8. Advanced & Streaming
    # "streamer",
    # "assistant_model",
    # "logits_processor",
    # "stopping_criteria",
    # "guidance_scale",
    # "watermarking_config"
]

class StreamBuffer:
    def __init__(self, tokenizer, max_size = 20):
    self.max_size = max_size
    self.tokens = []
    self.size = 0
    self.tokenizer = tokenizer
  def peek(self):
    if self.size == 0:
      raise Exception("Tokens is empty")
    element = self.tokens[0]
    self.tokens = self.tokens[1:]
    self.size -= 1
    return element
  def push(self, element):
    if self.size < self.max_size:
      self.tokens.append(element)
      self.size += 1
    else:
      _ = self.peek()
      self.tokens.append(element)
  @property
  def text(self):
    return self.tokenizer.convert_tokens_to_string(self.tokens) if self.size > 0 else ""

  def is_full(self):
    return self.size == self.max_size


class TransformerChatModel(BaseChatModel):

    # for creating model
    pretrained_model_name_or_path: str
    device: str = "auto"
    attn_implementation: str|None = None
    quantization_config: Any = None
    torch_dtype: Any = "auto"
    hf_token: str|None = None # some model need authorized
    # tools
    bound_tools: List[Any] = []

    # for generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    additional_generation_kwargs: dict[str, Any] = {}

    # other
    model: Any = None
    tokenizer: Any = None
    max_context_length: int = None

    # for template
    template_system_name: str = "base_system.jinja2"
    template_system: Any = None

    # for debug mode
    debug: bool = False


    def __init__(self, *, model = None, tokenizer = None, **kwargs: Any):
        # initial pydantic
        super().__init__(**kwargs)

        # load template
        self.template_system = ENV.get_template(self.template_system_name)

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
            print("Loading Model ...\n")
            self.model = AutoModelForCausalLM.from_pretrained(**load_model_config)
            self.model.eval()
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

    def _generation_config(self, **kwargs) -> dict:
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **self.additional_generation_kwargs
        }

        # check well known generation params
        for param in _WELL_KNOWN_GENERATION_PARAMS:
            if param in kwargs:
                generation_config[param] = kwargs[param]
        return generation_config

    def get_tools_desc(self) -> list[str]:
        """
        Return all tools description in a list.
        Each description is converted into format:

        name :
        docstring.
        params:
            * param1 (type)
            * param2 (type)

        """

        tools_desc = []
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

            tool_desc = f"""**{tool_name}**:{tool_desc}\n\tparams:\n{params_desc}\n"""
            tools_desc.append(tool_desc)

        return tools_desc

    @override
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
            template_system_name = self.template_system_name,
            debug = self.debug,
            **self.additional_generation_kwargs
        )

        return model_with_tools


    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:

        if messages[0].type == "system":
            system_prompt = messages[0].content
            messages = messages[1:]
        else:
            system_prompt = None

        internal_system_prompt = self.template_system.render(
            tools_desc = self.get_tools_desc(),
            system_prompt = system_prompt
        )
        messages = [SystemMessage(content=internal_system_prompt)] + messages

        # convert messages to prompt
        hf_msg = convert_lc_messages_to_hf_messages(messages)
        # prompt
        prompt = self.tokenizer.apply_chat_template(
            hf_msg,
            tokenize = False,
            add_generation_prompt = True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        ).to(self.model.device)

        generation_config = self._generation_config(**kwargs)
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
                if self.debug:
                    print("="*25+"DEBUG - TOOL CALLS"+"="*25)
                    print(response)
                    print("=" * 50)
                msg = AIMessage(content="", tool_calls = tool_calls)
            else: 
                msg = AIMessage(content = response)
        else:
            msg = AIMessage(content = response)
        # post-process to extract tool_call here
        # CODE
        if self.debug:
            messages.append(AIMessage(content = response))
            hf_msg = convert_lc_messages_to_hf_messages(messages)
            # prompt
            prompt = self.tokenizer.apply_chat_template(
                hf_msg,
                tokenize = False,
                add_generation_prompt = True,
            )

            print("="*25+"DEBUG - PROMPT"+"="*25)
            print(prompt)
            print("=" * 50)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream_transformer(
        self,
        messages: List[BaseMessage] | List[Dict[str, Any]],
        **kwargs
    ) -> Iterator[str]:
        """Streaming in transformer model"""
        if messages[0].type == "system":
            system_prompt = messages[0].content
            messages = messages[1:]
        else:
            system_prompt = None

        internal_system_prompt = self.template_system.render(
            tools_desc = self.get_tools_desc(),
            system_prompt = system_prompt
        )
        messages = [SystemMessage(content=internal_system_prompt)] + messages

        # convert messages to prompt
        hf_msg = convert_lc_messages_to_hf_messages(messages)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt = True,
            skip_special_tokens = True
        )

        prompt = self.tokenizer.apply_chat_template(
            hf_msg,
            tokenize = False,
            add_generation_prompt = True,
        )
        # tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length
        ).to(self.model.device)

        generation_config = self._generation_config(**kwargs)

        inputs.update({
            "streamer": streamer,
            **generation_config
        })

        def generate():
            with torch.inference_mode():
                self.model.generate(**inputs)
        
        # create thread
        thread = Thread(target=generate)
        thread.start()
        
        # yield token
        for token in streamer:
            yield token
        
        thread.join()

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens from the model."""

        is_tool_call = False
        tool_chunk_idx = -1
        stream_buffer = StreamBuffer(self.tokenizer)
        
        for token in self._stream_transformer(messages, **kwargs):

            yield ChatGenerationChunk(
                message = AIMessageChunk(content=token)
            )
        
        # yield final chunk
        yield ChatGenerationChunk(
            message = AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})
        )


    @property
    def _llm_type(self) -> str:
        return "transformer-chat-model"

