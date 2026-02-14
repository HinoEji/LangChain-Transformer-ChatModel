
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage,ToolMessage, ToolCall
from typing import Any, Mapping, List, Optional, Dict

__all__ = [
    "convert_hf_messages_to_lc_messages",
    "convert_lc_messages_to_hf_messages"
]

def _convert_hf_message_to_lc_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """
    Convert message in huggingface style into LangChain Message style
    """
    role = _dict["role"]
    content = _dict["content"]
    message = None

    if role == "system":
        message = SystemMessage(content=content)
    elif role == "user":
        message = HumanMessage(content=content)
    elif role == "assistant":
        message = AIMessage(content=content)
    elif role == "developer":
        # such model like : GPT-OSS
        message = SystemMessage(content=content)
    else:
        raise ValueError(f"Invalid role: {role}")

    return message

def convert_hf_messages_to_lc_messages(messages: List[Mapping[str, Any]]) -> List[BaseMessage]:
    """
    Convert messages in huggingface style into LangChain Message style
    """
    return [_convert_hf_message_to_lc_message(message) for message in messages]


def convert_lc_messages_to_hf_messages(messages:List[BaseMessage]) -> List[Mapping[str, Any]]:
    """
    Convert messages in LangChain Message style into huggingface style
    """
    chat_messages = []
    for message in messages:
        _type = message.type
        if _type == "system":
            content = message.content
            msg = {"role": "system", "content":content}
            chat_messages.append(msg)
        elif _type == "human":
            content = message.content
            msg = {"role": "user", "content" : content}
            chat_messages.append(msg)
        elif _type == "tool":
            role = "assistant"
            tool_call_id = message.tool_call_id
            status = message.status
            content = f"Respone of tool_call_id {tool_call_id} with status {status} and content : {message.content}"
            msg = {"role" : "assistant", "content" : content}
            chat_messages.append(msg)
        elif _type == "ai":
            role = "assistant"
            # check if there is a tool call
            tool_calls = message.tool_calls
            if tool_calls:
                # this is a tool call
                for tool_call in tool_calls:
                    name = tool_call["name"]
                    args = tool_call["args"]
                    id = tool_call["id"]
                    content = f"Tool Calling:\n tool_call_id : {id}\n name : {name}\n args : {args}"
                    msg = {"role" : "assistant", "content" : content}
                    chat_messages.append(msg)
            else:
                content = message.content
                msg = {"role" : "assistant", "content" : content}
                chat_messages.append(msg)
        else:
            raise ValueError(f"Invalid message type: {_type}")
    return chat_messages
