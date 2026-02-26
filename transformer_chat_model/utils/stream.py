from langchain_core.messages import AIMessageChunk, ToolCallChunk
from langchain_core.outputs import ChatGenerationChunk
import re
from .tool_call import ensure_id

__all__ = [
    "TransformerStreamer"
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
        
    def is_empty(self):
        return self.size == 0

    def reset(self):
        self.tokens = []
        self.size = 0


class ToolCallChunkParser:
    def __init__(self,tokenizer, id, index, init_tokens = None):
        self.tokenizer = tokenizer
        self.id = id
        self.index = index 
        self.buffer = StreamBuffer(tokenizer = tokenizer, max_size=3)
        if init_tokens:
            for token in init_tokens:
                self.buffer.push(token)

        self.data = self.buffer.tokens.copy()

        self.name_state = "find_key"
        self.args_state = None

    def _check_name_key(self,text):
        # Pattern mở rộng
        # Group 1: Dấu nháy của Key
        # Group 2: Tên Key (name/tool_name)
        # Group 3: VALUE (lấy mọi thứ cho đến khi gặp dấu phẩy)
        pattern = r'(["\']?)\b(name|tool_name)\b\1\s*:\s*([^,]*)'
        
        match = re.search(pattern, text)
        if match:
            raw_value = match.group(3).strip()
            
            # Xử lý làm sạch value (nếu muốn)
            # Vì streaming có thể cắt giữa chừng: '"search_we' -> ta có thể muốn lấy 'search_we'
            # Hoặc nếu nó đã có dấu đóng: '"search_web"' -> bỏ dấu nháy đi
            
            clean_value = raw_value
            # Nếu bắt đầu bằng " hoặc ', bỏ nó đi
            if clean_value and clean_value[0] in ('"', "'"):
                clean_value = clean_value[1:]
            # Nếu kết thúc bằng " hoặc ', bỏ nó đi (chỉ xảy ra nếu đã stream xong value)
            if clean_value and clean_value[-1] in ('"', "'"):
                clean_value = clean_value[:-1]
                
            return True, clean_value
        return False, None

    def _extract_name_value(self,text):
        pattern = r'\s*([^,]*)'
        
        match = re.search(pattern, text)
        if match:
            raw_value = match.group(0).strip()
            
            clean_value = raw_value
            # Nếu bắt đầu bằng " hoặc ', bỏ nó đi
            if clean_value and clean_value[0] in ('"', "'"):
                clean_value = clean_value[1:]
            # Nếu kết thúc bằng " hoặc ', bỏ nó đi (chỉ xảy ra nếu đã stream xong value)
            if clean_value and clean_value[-1] in ('"', "'"):
                clean_value = clean_value[:-1]
                
            return clean_value
        return None

    def _check_args_key(self,text):
        # Pattern giải thích:
        # 1. (["\']?)             -> Group 1: Dấu nháy mở (nếu có)
        # 2. \b(args|arguments)\b -> Group 2: Key là 'args' hoặc 'arguments'
        # 3. \1                   -> Dấu nháy đóng khớp với Group 1
        # 4. \s*:\s*              -> Dấu hai chấm và khoảng trắng
        # 5. (.*)                 -> Group 3: LẤY TẤT CẢ (Greedy) cho đến hết chuỗi
        pattern = r'(["\']?)\b(args|arguments)\b\1\s*:\s*(.*)'
        
        #  dùng cờ re.DOTALL để dấu chấm (.) khớp được cả ký tự xuống dòng (\n)
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return True, match.group(3) # Trả về toàn bộ phần đuôi
        return False, None
    
    def __call__(self, token):
        self.buffer.push(token)
        name = None
        args = None

        if self.name_state == "completed" and self.args_state is None:
        
            # 1. ƯU TIÊN: Kiểm tra xem đã đến phần 'args' chưa?
            # Nếu đã đến phần args, chúng ta PHẢI dừng cập nhật name ngay lập tức.
            found_args_key, args_value = self._check_args_key(self.buffer.text)
            
            if found_args_key:
                if self.args_state is None:
                    self.args_state = "find_value"
                    self.name_state = "completed" # Đánh dấu đã xong name
                
                # Nếu regex _check_args_key lấy được phần giá trị sau dấu : (như {"...)
                if args_value:
                    args = args_value
                return self._build_chunk(name, args)

        # 2. Nếu đang trong trạng thái stream nội dung của args
        if self.args_state == "find_value":
            args = token
            return self._build_chunk(name, args)

        # 3. Xử lý phần NAME
        if self.name_state == "find_key":
            found_name_key, name_value = self._check_name_key(self.buffer.text)
            if found_name_key:
                self.name_state = "find_value"
                if name_value:
                    name = name_value
        
        elif self.name_state == "find_value":
            # Chỉ lấy token làm name nếu token này không chứa các ký tự điều hướng (như dấu phẩy)
            # và phải cẩn thận với khoảng trắng/dấu nháy thừa.
            
            # Nếu token chứa dấu phẩy, có nghĩa là kết thúc name
            if "," in token:
                parts = token.split(",", 1)
                name = self._extract_name_value(parts[0])
                self.name_state = "completed" # Chờ đợi đến khi thấy key args
            else:
                # Kiểm tra xem token có phải là rác không (như dấu ngoặc, khoảng trắng dư)
                raw_extracted = self._extract_name_value(token)
                if raw_extracted and raw_extracted not in ['"', "'", ":"]:
                    name = raw_extracted

        return self._build_chunk(name, args)

    def _build_chunk(self, name, args):
        """Helper để format và tạo ToolCallChunk"""
        if name:
            name = self.tokenizer.convert_tokens_to_string([name]).strip()
            name = name.replace("\n","")
            # Loại bỏ dấu nháy dư thừa nếu có
            name = name.strip("\"' :")
            if not name: name = None

        if args:
            args = self.tokenizer.convert_tokens_to_string([args]).strip()
            args = args.replace("\n", "")
            if not args: args = None

        if self.id is None and args is None and name is None:
            return None

        tool_call_chunk = ToolCallChunk(
            name=name,
            args=args,
            id=self.id,
            index=self.index
        )
        if self.id: self.id = None
        return tool_call_chunk



class TransformerStreamer:
    def __init__(self, tokenizer, buffer_size: int = 5):
        self.tokenizer = tokenizer
        self.buffer = StreamBuffer(tokenizer = tokenizer, max_size = buffer_size)

    def _get_behind_tool_call(self, text):
        """Get the rest behind tag <tool_call>"""
        match = re.search(r'<tool_call>(.*)$', text, re.DOTALL)
        if match:
            tail = match.group(1)
            internal_char = re.findall(r'[^\s]', tail)
            if internal_char:
                text = ''.join(internal_char)
                internal_tokens = self.tokenizer.tokenize(text)
                return internal_tokens
            else:
                return None
        else:
            return None

    def _check_state_position(self,depth, inside_tool_call):
        if depth == 0 and inside_tool_call == True:
            return "end_call"
        if depth == 1 and inside_tool_call == False:
            return "start_call"
        
        if depth == 0 and inside_tool_call == False:
            return "outside_call"

    # def _parse_chunk_from_token(self, token: str, index, depth : int, inside_tool_call: bool ,tool_call_chunk_parser = None):
    #     added = False
    #     tool_call_chunks = []
    #     for i,char in enumerate(token):
    #         if char == '{':
    #             depth +=1 
    #         elif char == '}':
    #             depth -= 1

    #         state = self._check_state_position(depth, inside_tool_call)

    #         if state == "start_call":
    #             # gán lại biến inside tool call
    #             inside_tool_call = True
    #             # tạo một tool call chunk parser
    #             index += 1
    #             _id = ensure_id(None)
    #             tool_call_chunk_parser = ToolCallChunkParser(tokenizer = self.tokenizer, id = _id, index = index , init_tokens = token[i:])
    #             added = True
    #             continue
    #         elif state == "end_call":
    #             inside_tool_call = False
    #             tool_call_chunk = tool_call_chunk_parser(token[0:i])
    #             if tool_call_chunk:
    #                 tool_call_chunks.append(tool_call_chunk)
    #                 added = True
    #             tool_call_chunk_parser = None
    #     if inside_tool_call and not added:
    #         tool_call_chunk = tool_call_chunk_parser(token)
    #         if tool_call_chunk:
    #             tool_call_chunks.append(tool_call_chunk)
    #     return index, depth, inside_tool_call, tool_call_chunk_parser, tool_call_chunks

    def _parse_chunk_from_token(self, token: str, index, depth: int, inside_tool_call: bool, tool_call_chunk_parser=None):
        tool_call_chunks = []
        token_already_sent = False # Flag để đảm bảo token chỉ gửi cho parser 1 lần

        for i, char in enumerate(token):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1

            state = self._check_state_position(depth, inside_tool_call)

            if state == "start_call":
                inside_tool_call = True
                index += 1
                _id = ensure_id(None)
                # Khởi tạo parser và gửi phần còn lại của token (từ sau dấu {)
                tool_call_chunk_parser = ToolCallChunkParser(tokenizer=self.tokenizer, id=_id, index=index)
                
                # Gửi phần nội dung sau dấu { vào parser
                trailing_content = token[i+1:]
                if trailing_content:
                    chunk = tool_call_chunk_parser(trailing_content)
                    if chunk: tool_call_chunks.append(chunk)
                
                token_already_sent = True # Đánh dấu token này đã được xử lý xong
                break # Thoát vòng lặp char vì đã xử lý hết phần content sau {

            elif state == "end_call":
                # Gửi phần nội dung trước dấu } vào parser
                if not token_already_sent:
                    leading_content = token[:i]
                    if leading_content:
                        chunk = tool_call_chunk_parser(leading_content)
                        if chunk: tool_call_chunks.append(chunk)
                
                inside_tool_call = False
                token_already_sent = True
                # Không break vì có thể còn <tool_call> tiếp theo trong cùng token
                continue

        # Nếu đang ở trong tool call và token chưa được gửi ở trên
        if inside_tool_call and not token_already_sent:
            chunk = tool_call_chunk_parser(token)
            if chunk:
                tool_call_chunks.append(chunk)

        return index, depth, inside_tool_call, tool_call_chunk_parser, tool_call_chunks

    
    def __call__(self, iterator):
        tool_call_chunk_parser = None
        is_tool_call = False
        index = -1
        depth = 0
        inside_tool_call = False
        chunks = []
        finish_reason = None
        # nếu depth = 0 và inside tool_call = true thì kết thúc một tool_call
        # nếu depth = 1 và inside tool_call = False thì bắt đầu một tool_call và set biến inside tool_call = True
        # khi đưa token vào một toolcallchunk parser sẽ có các trường hợp sau:
        # 1. Chỉ tồn tại 1 tool call -> tool call có thể hoàn chỉnh hoặc chưa
        # 2. Nhiều hơn 1 tool call -> chắc chắc rằng ngoại trừ tool call cuối, thì các tool call phía trước chắn chắc là đã hoàn thành và chỉ việc yield ra ngoài
        # 3. Vì (2) nên ta nên xử lí một token và tách riêng từng tool call với nhau (nếu có nhiều tool call)

        while not self.buffer.is_full():
            token = next(iterator) # lấy từng token
            # bỏ nó vào buffer và check xem có xuất hiện tag <tool_call> chưa
            self.buffer.push(token)

            if "<tool_call>" in self.buffer.text:
                is_tool_call = True
                break
        
        if is_tool_call:
            remains = self._get_behind_tool_call(self.buffer.text)
            # reset the buffer
            self.buffer.reset()
            if remains:
                for token in remains:
                    index, depth, inside_tool_call, tool_call_chunk_parser, tool_call_chunks = self._parse_chunk_from_token(token = token, index = index, depth = depth, inside_tool_call=inside_tool_call ,tool_call_chunk_parser = tool_call_chunk_parser)
                    if tool_call_chunks:
                        chunk = AIMessageChunk(
                            content = '',
                            tool_call_chunks = tool_call_chunks
                        )
                        
                        chat_chunk = ChatGenerationChunk(
                            message = chunk
                        )
                        yield chat_chunk
                
            # xử lí phần còn lại tương tự
            for token in iterator:
                index, depth, inside_tool_call, tool_call_chunk_parser, tool_call_chunks = self._parse_chunk_from_token(token = token, index = index, depth = depth, inside_tool_call=inside_tool_call ,tool_call_chunk_parser = tool_call_chunk_parser)
                if tool_call_chunks:
                    chunk = AIMessageChunk(
                        content = '',
                        tool_call_chunks = tool_call_chunks
                    )
                    
                    chat_chunk = ChatGenerationChunk(
                    message = chunk
                )
                    yield chat_chunk
            finish_reason = "tool_calls"

        else:
            # Nếu không phải Tool Call thì trả về AIMessageChunk bình thường
            # xả hết token trong buffer
            for token in self.buffer.tokens:
                chunk = AIMessageChunk(content = token)
                chat_chunk = ChatGenerationChunk(
                    message = chunk
                )
                yield chat_chunk
            
            for token in iterator:
                chunk = AIMessageChunk(content = token)
                chat_chunk = ChatGenerationChunk(
                    message = chunk
                )
                yield chat_chunk
            finish_reason = "stop"
        
        # xử lí chunk cuối cùng
        last_chunk = AIMessageChunk(
            content = '',
            finish_reason = finish_reason
        )
        
        chat_chunk = ChatGenerationChunk(
            message = last_chunk
        )
        yield chat_chunk