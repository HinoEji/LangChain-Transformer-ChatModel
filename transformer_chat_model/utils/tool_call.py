from langchain_core.messages import ToolCall
from typing import Any, List, Dict, Optional
from uuid import uuid4
import json
import re

__all__ = [
    "create_tool_call",
    "ToolCallParser"
]

LC_AUTO_PREFIX = "lc_"
"""LangChain auto-generated ID prefix for messages and content blocks."""

LC_ID_PREFIX = "lc_run-"
"""Internal tracing/callback system identifier.

Used for:

- Tracing. Every LangChain operation (LLM call, chain execution, tool use, etc.)
    gets a unique run_id (UUID)
- Enables tracking parent-child relationships between operations
"""


def ensure_id(id_val: str | None) -> str:
    """Ensure the ID is a valid string, generating a new UUID if not provided.

    Auto-generated UUIDs are prefixed by `'lc_'` to indicate they are
    LangChain-generated IDs.

    Args:
        id_val: Optional string ID value to validate.

    Returns:
        A string ID, either the validated provided value or a newly generated UUID4.
    """
    return id_val or f"{LC_AUTO_PREFIX}{uuid4()}"


def create_tool_call(
    name: str,
    args: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ToolCall:
    """Create a `ToolCall`.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier for the tool call.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `ToolCall`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    """
    block = ToolCall(
        type="tool_call",
        name=name,
        args=args,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block



class ToolCallParser:
    """
    Strict parser: ONLY extracts tool calls strictly wrapped in <tool_call> tags.
    Format expected:
    <tool_call>
    [
        {"name": "...", "arguments": {...}},
        ...
    ]
    </tool_call>
    """

    @staticmethod
    def extract_tool_calls(text: str) -> List[ToolCall]:
        return ToolCallParser._extract_from_xml_tags(text)

    @staticmethod
    def _extract_from_xml_tags(text: str) -> List[ToolCall]:
        """Extract content ONLY from <tool_call> tags."""
        tool_calls = []
        pattern = r'<tool_call>\s*(.*?)(?:</tool_call>|\Z)'
        
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for match_content in matches:
            json_content = ToolCallParser._extract_json_list_from_text(match_content)
            
            if json_content:
                calls = ToolCallParser._parse_json_content(json_content)
                tool_calls.extend(calls)

        return tool_calls

    @staticmethod
    def _parse_json_content(content: str) -> List[ToolCall]:
        """Parse JSON content expecting a List of tools."""
        tool_calls = []
        try:
            parsed = json.loads(content)
            
            # Format yêu cầu là một List [...]
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        # Lấy tên tool và tham số
                        name = item.get("name") or item.get("tool_name")
                        args = item.get("arguments", item.get("args", {}))
                        
                        if name:
                            if args:
                                tool_call = create_tool_call(name=name, args=args)
                            else:
                                for key, value in item.items():
                                    if key != "name" and key != "tool_name":
                                        args = value
                                        break
                                tool_call = create_tool_call(name=name, args=args)
                            tool_calls.append(tool_call)
                            
            # Nếu lỡ model trả về dict đơn lẻ {...} trong thẻ tag (dù sai format list nhưng vẫn trong tag)
            # Ta vẫn có thể hỗ trợ hoặc bỏ qua.
            elif isinstance(parsed, dict):
                 name = parsed.get("name") or parsed.get("tool_name")
                 if name:
                     tool_call = create_tool_call(
                         name=name, 
                         args=parsed.get("arguments", parsed.get("args", {}))
                     )
                     tool_calls.append(tool_call)

        except json.JSONDecodeError:
            # Nếu JSON lỗi nhẹ, thử sửa
            fixed = ToolCallParser._fix_json_issues(content)
            if fixed != content:
                return ToolCallParser._parse_json_content(fixed)
        
        return tool_calls

    @staticmethod
    def _extract_json_list_from_text(text: str) -> Optional[str]:
        """
        Find a string starting with '[' and try to find a reasonable end.
        """
        start = text.find('[')
        if start == -1:
            # if list is not found, try to find object '{' in case the model forgets to close the square bracket
            start_obj = text.find('{')
            if start_obj != -1:
                return ToolCallParser._extract_json_balance(text, start_obj, '{', '}')
            return None

        return ToolCallParser._extract_json_balance(text, start, '[', ']')

    @staticmethod
    def _extract_json_balance(text: str, start: int, open_char: str, close_char: str) -> str:
        """Helper to get a balanced JSON string."""
        count = 0
        for i, char in enumerate(text[start:], start):
            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1
                if count == 0:
                    return text[start:i+1]
        
        # In case of streaming, return the remaining part
        return text[start:]

    @staticmethod
    def _fix_json_issues(content: str) -> str:
        """Fix basic JSON issues."""
        fixed = content.strip()
        # Remove trailing commas in arrays/objects: , } -> }
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Balance square brackets (for List)
        open_brackets = fixed.count('[') - fixed.count(']')
        if open_brackets > 0: fixed += ']' * open_brackets
        
        # Balance curly braces (for Object)
        open_braces = fixed.count('{') - fixed.count('}')
        if open_braces > 0: fixed += '}' * open_braces

        return fixed