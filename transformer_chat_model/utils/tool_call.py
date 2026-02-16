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
    """Parses model responses to extract tool calls with robust handling."""

    @staticmethod
    def extract_tool_calls(text: str) -> List[ToolCall]:
        """Extract tool calls from model response text with improved robustness."""
        tool_calls = []

        # Try multiple extraction strategies in order of preference
        strategies = [
            ToolCallParser._extract_from_complete_json_blocks,
            ToolCallParser._extract_from_incomplete_json_blocks,
            ToolCallParser._extract_from_json_anywhere,
            # ToolCallParser._extract_from_function_calls
        ]

        for strategy in strategies:
            try:
                calls = strategy(text)
                if calls:
                    tool_calls.extend(calls)
                    # If we found calls with a strategy, we can break or continue to find more
                    # Continue to catch cases where multiple formats exist
            except Exception as e:
                continue

        # Remove duplicates based on name and args
        unique_calls = []
        seen = set()
        for call in tool_calls:
            signature = (call["name"], str(call["args"]))
            if signature not in seen:
                seen.add(signature)
                unique_calls.append(call)

        return unique_calls

    @staticmethod
    def _extract_from_complete_json_blocks(text: str) -> List[ToolCall]:
        """Extract from properly formatted ```json...``` blocks."""
        tool_calls = []
        pattern = r'```tool_call_block\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            calls = ToolCallParser._parse_json_content(match)
            tool_calls.extend(calls)

        return tool_calls

    @staticmethod
    def _extract_from_incomplete_json_blocks(text: str) -> List[ToolCall]:
        """Extract from incomplete JSON blocks (```json... without closing)."""
        tool_calls = []

        # Look for ```json or ```JSON followed by content
        pattern = r'```(?:tool_call_block|TOOL_CALL_BLOCK)\s*(.*?)(?=```|\Z)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            # Try to find where JSON likely ends
            json_content = ToolCallParser._extract_json_from_text(match)
            if json_content:
                calls = ToolCallParser._parse_json_content(json_content)
                tool_calls.extend(calls)

        return tool_calls

    @staticmethod
    def _extract_from_json_anywhere(text: str) -> List[ToolCall]:
        """Extract JSON that looks like tool calls from anywhere in text."""
        tool_calls = []

        # Look for JSON-like structures that contain "tool_calls" or tool patterns
        # This handles cases where there are no markdown code blocks at all
        json_patterns = [
            r'\{[^{}]*"tool_calls"[^{}]*\[[^\]]*\][^{}]*\}',  # Simple tool_calls pattern
            r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',        # Single tool call pattern
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                # Try to expand the match to get complete JSON
                expanded = ToolCallParser._expand_json_match(text, match)
                calls = ToolCallParser._parse_json_content(expanded)
                tool_calls.extend(calls)

        return tool_calls

    # @staticmethod
    # def _extract_from_function_calls(text: str) -> List[ToolCall]:
    #     """Extract from direct function call format like multiply(2, 6)."""
    #     tool_calls = []

    #     # Look for function call patterns
    #     func_pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
    #     matches = re.findall(func_pattern, text)

    #     for func_name, args_str in matches:
    #         # Skip common English words that might match the pattern
    #         if func_name.lower() in ['to', 'in', 'on', 'at', 'by', 'for', 'with', 'from']:
    #             continue

    #         args = ToolCallParser._parse_function_args(args_str)
    #         tool_call = create_tool_call(
    #             name = func_name,
    #             args=args
    #         )
    #         tool_calls.append(tool_call)

    #     return tool_calls

    @staticmethod
    def _parse_json_content(content: str) -> List[ToolCall]:
        """Parse JSON content and extract tool calls."""
        tool_calls = []

        try:
            parsed = json.loads(content)

            if isinstance(parsed, dict):
                if "tool_calls" in parsed:
                    # Multiple tool calls format
                    for call in parsed["tool_calls"]:
                        if isinstance(call, dict) and ("name" in call or "tool_name" in call):
                            tool_call = create_tool_call(
                                name = call.get("name") or call.get("tool_name"),
                                args = call.get("arguments", call.get("args", {}))
                            )
                            tool_calls.append(tool_call)
                elif "name" in parsed or "tool_name" in parsed:
                    # Single tool call format
                    tool_call = create_tool_call(
                        name = parsed.get("name") or parsed.get("tool_name"), 
                        args=parsed.get("arguments", parsed.get("args", {}))
                        )
                    tool_calls.append(tool_call)

        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed_content = ToolCallParser._fix_json_issues(content)
            if fixed_content != content:
                return ToolCallParser._parse_json_content(fixed_content)

        return tool_calls

    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """Extract the most likely JSON content from text."""
        # Find the first { and try to find matching }
        start = text.find('{')
        if start == -1:
            return None

        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start:i+1]

        # If we can't find matching braces, try to take until end of reasonable JSON
        # Look for common JSON ending patterns
        json_text = text[start:]
        for end_pattern in ['\n\n', '```', '\n}', '}']:
            if end_pattern in json_text:
                potential_end = json_text.find(end_pattern)
                if end_pattern == '}':
                    potential_end += 1
                potential_json = json_text[:potential_end]
                if potential_json.count('{') <= potential_json.count('}'):
                    return potential_json

        return json_text

    @staticmethod
    def _expand_json_match(text: str, match: str) -> str:
        """Expand a partial JSON match to try to get complete JSON."""
        start_pos = text.find(match)
        if start_pos == -1:
            return match

        # Look backwards for opening brace
        actual_start = start_pos
        for i in range(start_pos - 1, -1, -1):
            if text[i] == '{':
                actual_start = i
                break
            elif text[i] == '}':
                break

        # Look forwards for closing brace
        actual_end = start_pos + len(match)
        brace_count = 0
        for i, char in enumerate(text[actual_start:]):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    actual_end = actual_start + i + 1
                    break

        return text[actual_start:actual_end]

    @staticmethod
    def _fix_json_issues(content: str) -> str:
        """Try to fix common JSON formatting issues."""
        fixed = content.strip()

        # Remove trailing commas
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)

        # Add missing closing braces/brackets if obvious
        open_braces = fixed.count('{') - fixed.count('}')
        open_brackets = fixed.count('[') - fixed.count(']')

        if open_braces > 0:
            fixed += '}' * open_braces
        if open_brackets > 0:
            fixed += ']' * open_brackets

        # Fix unquoted keys (simple cases)
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)

        return fixed

    @staticmethod
    def _parse_function_args(args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string format."""
        args = {}
        if not args_str.strip():
            return args

        try:
            # Try to handle as JSON-like object
            if args_str.strip().startswith('{'):
                return json.loads(args_str)
        except:
            pass

        # Handle positional and named arguments
        if '=' not in args_str:
            # Positional arguments
            values = [v.strip().strip('"\'') for v in args_str.split(',')]
            for i, value in enumerate(values):
                try:
                    # Try to convert to appropriate type
                    if value.isdigit():
                        args[f'arg_{i}'] = int(value)
                    elif value.replace('.', '').isdigit():
                        args[f'arg_{i}'] = float(value)
                    else:
                        args[f'arg_{i}'] = value
                except:
                    args[f'arg_{i}'] = value
        else:
            # Named arguments
            for arg in args_str.split(','):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    try:
                        if value.isdigit():
                            args[key] = int(value)
                        elif value.replace('.', '').isdigit():
                            args[key] = float(value)
                        elif value.lower() in ['true', 'false']:
                            args[key] = value.lower() == 'true'
                        else:
                            args[key] = value
                    except:
                        args[key] = value

        return args