import json
import requests
from typing import Dict, List, Any, Optional
import ollama  # Import the ollama Python library


class LLMProvider:
    """
    Abstract base class for LLM providers.
    This defines the interface that all LLM providers must implement.
    """
    def __init__(self):
        pass
        
    def generate_text(self, prompt: str) -> str:
        """Generate text based on a prompt"""
        raise NotImplementedError("Subclasses must implement generate_text")
        
    def generate_json(self, prompt: str) -> Dict:
        """Generate and parse a JSON response"""
        raise NotImplementedError("Subclasses must implement generate_json")
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with details"""
        raise NotImplementedError("Subclasses must implement get_available_models")
        
    def get_running_models(self) -> List[Dict[str, Any]]:
        """Get list of currently running models"""
        raise NotImplementedError("Subclasses must implement get_running_models")
        
    def set_model(self, model_name: str) -> bool:
        """Set the current model"""
        raise NotImplementedError("Subclasses must implement set_model")
        
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        raise NotImplementedError("Subclasses must implement get_model_info")


class OllamaProvider(LLMProvider):
    """
    Ollama implementation of the LLM provider interface using the official ollama Python library.
    """
    def __init__(self, host: str = "http://127.0.0.1:11434", model_name: str = "mistral"):
        """
        Initialize the Ollama provider.
        
        Args:
            host: URL of the Ollama API (default: "http://127.0.0.1:11434")
            model_name: Name of the model to use (default: "mistral")
        """
        super().__init__()
        self.host = host
        self.model_name = model_name
        self.temperature = 0.7  # Default temperature
        
        # Create client with custom host
        self.client = ollama.Client(host=host)
        
        # Try to find an available model if the default one doesn't exist
        try:
            models = self.get_available_models()
            if models:
                # Check if the default model exists
                model_names = [model.get('model', '') for model in models]
                if model_name not in model_names:
                    # Default model not found, use the first available model instead
                    self.model_name = models[0].get('model', '')
                    print(f"Default model not found. Using '{self.model_name}' instead.")
            else:
                # No models available
                self.model_name = None
                print("No models available. Please install at least one model with 'ollama pull <model>'.")
        except Exception as e:
            print(f"Error checking for available models: {e}")
        
    def generate_text(self, prompt: str, temperature: float = None) -> str:
        """
        Generate text from a prompt using Ollama.
        
        Args:
            prompt: Input text prompt
            temperature: Generation temperature (0.0 to 2.0)
            
        Returns:
            Generated text response
        """
        try:
            # Use provided temperature or default to instance temperature
            temp = temperature if temperature is not None else self.temperature
            
            response = self.client.generate(
                model=self.model_name, 
                prompt=prompt,
                options={"temperature": temp}
            )
            return response.get('response', 'No response from model')
        except Exception as e:
            return f"Error: {str(e)}"
            
    def generate_chat(self, messages: List[Dict[str, str]], temperature: float = None) -> str:
        """
        Generate a chat completion using Ollama chat API.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            temperature: Generation temperature (0.0 to 2.0)
            
        Returns:
            Generated response text
        """
        try:
            # Use provided temperature or default to instance temperature
            temp = temperature if temperature is not None else self.temperature
            
            response = self.client.chat(
                model=self.model_name, 
                messages=messages,
                options={"temperature": temp}
            )
            return response.get('message', {}).get('content', 'No response from model')
        except Exception as e:
            return f"Error: {str(e)}"
            
    def generate_json(self, prompt: str) -> Dict:
        """
        Generate and parse a JSON response from the Ollama model.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Parsed JSON response as a dictionary
        """
        formatted_prompt = f"{prompt}\n\nRespond with valid JSON only."
            
        try:
            response_text = self.generate_text(formatted_prompt)
            
            # Extract JSON from response if wrapped in markdown code blocks
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()
                
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw_text": response_text}
            
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models using ollama library.
        
        Returns:
            List of model objects with details
        """
        try:
            # Use ollama client to list models
            models = self.client.list()
            return models.get('models', [])
            
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []
    
    def get_running_models(self) -> List[Dict[str, Any]]:
        """
        Get list of running models.
        
        Returns:
            List of running model objects with details
        """
        try:
            # Use ollama client to get running models
            ps_result = self.client.ps()
            return ps_result.get('models', [])
        except Exception as e:
            print(f"Error fetching running models: {e}")
            return []
            
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model to get information for
            
        Returns:
            Dictionary with model information
        """
        try:
            # Use ollama client to show model info
            return self.client.show(model=model_name)
        except Exception as e:
            print(f"Error fetching model info: {e}")
            return {"error": str(e)}
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            Dictionary with pull status information
        """
        try:
            # Use ollama client to pull model
            self.client.pull(model=model_name)
            return {"status": "success", "model": model_name}
        except Exception as e:
            print(f"Error pulling model: {e}")
            return {"error": str(e)}
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use ollama client to delete model
            self.client.delete(model=model_name)
            return True
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
            
    def set_model(self, model_name: str) -> bool:
        """
        Set the current model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if model exists in available models
            available_models = self.get_available_models()
            model_exists = any(model.get('name') == model_name for model in available_models)
            
            if model_exists:
                self.model_name = model_name
                return True
            else:
                print(f"Model {model_name} not found in available models")
                return False
        except Exception as e:
            print(f"Error setting model: {e}")
            return False