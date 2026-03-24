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
    ToolCallParser,
    GPTOSSStreamer
)

from .classic_transformer import ClassicTransformerChatModel

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


class GPTOSSChatModel(ClassicTransformerChatModel):
    reasoning_effort: str = "medium" # auto, low, medium, high
    # template_system_name: str = "gptoss_system.jinja2"
    
    def __init__(self, *, model = None, tokenizer = None, **kwargs: Any):
        if quantization_config:
            quantization_config = None
            print("Quantization config is auto for GPTOSS model in MXFP4")
            print("You may try the unsloth models for other quantized models.")

        super().__init__(model = model, tokenizer = tokenizer, **kwargs)

    @override
    def _llm_type(self) -> str:
        return "gptoss-chat-model"


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
            reasoning_effort = self.reasoning_effort,
            **self.additional_generation_kwargs
        )

        return model_with_tools

    def _parse_response(self, response:str) -> dict[str, Any]:
        """Parse response from gpt oss model to components : analysis and final answer"""
        # analysis: có thể có hoặc không có <|end|>
        pattern_analysis = r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)"
        
        # final: có thể có hoặc không có <|return|>
        pattern_final = r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|$)"

        analysis_match = re.search(pattern_analysis, response, re.DOTALL)
        final_match = re.search(pattern_final, response, re.DOTALL)

        analysis = analysis_match.group(1).strip() if analysis_match else None
        final = final_match.group(1).strip() if final_match else None

        return {
            "analysis" : analysis,
            "final" : final
        }
        
    @override
    def _decide_reasoning_effort(self, messages: List[BaseMessage], limit_messages: int =2):
        """Decide reasoning effort based on reasoning_effort"""
        if self.reasoning_effort != "auto":
            return self.reasoning_effort
        
        instruction_prompt = '''You are a helpful assistant with great ability to decide how much reasoning effort should be used to response the next answer.
    Decide the reasoning effort based on the given context.
    ## RULES:
    - Only response : low, medium, high
    - No explanation or any other text
    '''
        instruction_message = SystemMessage(content=instruction_prompt)
        if messages[0].type == "system":
            new_messages = [instruction_message, messages[0]] + messages[-limit_messages:]
        else:
            new_messages = [instruction_message] + messages[-limit_messages:]

        # convert messages to prompt
        hf_msg = convert_lc_messages_to_hf_messages(messages)
        # prompt
        prompt = self.tokenizer.apply_chat_template(
            hf_msg,
            tokenize = False,
            add_generation_prompt = True,
            reasoning_effort = "low" # for the high speed
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128 # for the high speed
        ).to(self.model.device)

        # generate response
        generation_config = self._generation_config(**kwargs)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        ).strip()

        # parse the response
        parsed_response = self._parse_response(response)
        reasoning_effort = parsed_response["final"].lower()
        if reasoning_effort not in ["low", "medium", "high"]:
            reasoning_effort = "medium"
        return reasoning_effort

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

        # decide reasoning effort
        reasoning_effort = self._decide_reasoning_effort(messages)
        # prompt
        prompt = self.tokenizer.apply_chat_template(
            hf_msg,
            tokenize = False,
            add_generation_prompt = True,
            reasoning_effort = reasoning_effort
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
            skip_special_tokens=False
        ).strip()

        # parse response
        response_dict = self._parse_response(response)
        response = response_dict["final"]
        analysis = response_dict["analysis"]
        
        # speed up, tool call parsing is time consuming
        if self.bound_tools:
            # parse tool call
            tool_calls = ToolCallParser.extract_tool_calls(response)
            if tool_calls:
                if self.debug:
                    print("="*25+"DEBUG - TOOL CALLS"+"="*25)
                    print(response)
                    print("=" * 50)
                msg = AIMessage(content="", tool_calls = tool_calls, additional_kwargs={"analysis": analysis})
            else: 
                msg = AIMessage(content = response, additional_kwargs={"analysis": analysis})
        else:
            msg = AIMessage(content = response, additional_kwargs={"analysis": analysis})
        # post-process to extract tool_call here
        # CODE
        if self.debug:
            messages.append(AIMessage(content = response, additional_kwargs={"analysis": analysis}))
            hf_msg = convert_lc_messages_to_hf_messages(messages)
            # prompt
            prompt = self.tokenizer.apply_chat_template(
                hf_msg,
                tokenize = False,
                add_generation_prompt = True,
                reasoning_effort = reasoning_effort
            )

            print("="*25+"DEBUG - PROMPT"+"="*25)
            print(prompt)
            print("=" * 50)
        return ChatResult(generations=[ChatGeneration(message=msg)])
    
    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream tokens from the model."""

        gptoss_streamer = GPTOSSStreamer(
            tokenizer = self.tokenizer,
            buffer_size = 6
        )
        
        iterator = self._stream_transformer(messages, **kwargs)
        yield from gptoss_streamer(iterator)

