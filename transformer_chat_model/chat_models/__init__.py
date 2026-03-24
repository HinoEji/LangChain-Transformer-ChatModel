from .classic_transformer import ClassicTransformerChatModel
from .gptoss import GPTOSSChatModel

__all__ = ["init_transformer_chat_model"]

def init_transformer_chat_model(pretrained_model_name_or_path: str, **kwargs: Any):
    """
    Initialize a transformer chat model.
    
    Args:
        pretrained_model_name_or_path: The name or path of the pretrained model.
        **kwargs: Additional arguments to pass to the model constructor.
        Includes:
        
        device (str) -- the device to load the model to. Default is "auto".
        attn_implementation (str|None) -- the attention implementation to use. Default is None.
        quantization_config (BitsAndBytesConfig) -- the quantization configuration to use. Default is None.
        torch_dtype (Any) -- the torch dtype to use. Default is "auto".
        hf_token (str|None) -- the hf token to use. Default is None.
        max_new_tokens (int) -- the max new tokens to use. Default is 512.
        temperature (float) -- the temperature to use. Default is 0.7.
        do_sample (bool) -- the do sample to use. Default is True.
        max_context_length (int) -- the max context length to use. Default is None, then it will be set to the model's max_position_embeddings.
        template_system_name (str) -- the name of template system file, must be stored in transformer_chat_model/prompts/. Default is "base_system.jinja2".
        debug (bool) -- the debug mode to use. Default is False.
        additional_generation_kwargs (dict[str, Any]) -- the additional generation kwargs to use. Default is {}.
    
    Returns:
        A transformer chat model.
    """
    if "gpt-oss" in pretrained_model_name_or_path:
        return GPTOSSChatModel(pretrained_model_name_or_path = pretrained_model_name_or_path, **kwargs)
    else:
        return ClassicTransformerChatModel(pretrained_model_name_or_path = pretrained_model_name_or_path, **kwargs)
