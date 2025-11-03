# -*- coding: utf-8 -*-

import os
import time
from typing import Optional
from openai import OpenAI
import config

class LLMProvider:
    """
    A class that encapsulates interactions with multiple Large Language Models.
    - A High-Performance (HP) model for complex generation tasks.
    - A General-Purpose (GP) model for routine annotation tasks.
    """
    def __init__(self):
        # High-Performance Client (e.g., DeepSeek)
        if not config.HP_API_KEY:
            raise ValueError("High-Performance model API Key (HP_API_KEY) is not set in config.py.")
        self.hp_client = OpenAI(
            api_key=config.HP_API_KEY,
            base_url=config.HP_BASE_URL
        )
        self.hp_model = config.HP_MODEL_NAME

        # General-Purpose Client (e.g., Qwen)
        if not config.GP_API_KEY:
            raise ValueError("General-Purpose model API Key (GP_API_KEY) is not set in config.py.")
        self.gp_client = OpenAI(
            api_key=config.GP_API_KEY,
            base_url=config.GP_BASE_URL,
            timeout=30.0
        )
        self.gp_model = config.GP_MODEL_NAME

        # For error handling in general model calls
        self.error_counts = {"gp_model": 0}
        self.last_error_time = {"gp_model": 0}
        
        # Token usage tracking
        self.hp_total_tokens = 0
        self.gp_total_tokens = 0

    def get_completion(self, prompt: str, output_filename: str, overwrite: bool = False) -> str:
        """
        Gets a completion from the High-Performance model.
        This is used for complex generation tasks (roles, features, tools).
        It ensures a result exists, either by generating it or reading from a cache file.
        """
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        
        filepath = os.path.join(config.OUTPUT_DIR, output_filename)

        if not overwrite and os.path.exists(filepath):
            print(f"File '{output_filename}' already exists. Reading from cache.")
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

        print(f"Calling HP model to generate '{output_filename}'...")
        try:
            response = self.hp_client.chat.completions.create(
                model=self.hp_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            content = response.choices[0].message.content
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                self.hp_total_tokens += response.usage.total_tokens
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"...Successfully generated and saved to '{output_filename}'.")
            return content
        except Exception as e:
            error_message = f"Error calling HP model for '{output_filename}': {e}"
            print(error_message)
            # Still save the error to the file so the process can potentially continue
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(error_message)
            return error_message

    def call_general_model(self, prompt: str, temperature: float = 0, thread_id: Optional[int] = None) -> str:
        """
        Calls the General-Purpose model (e.g., qwen-turbo) with robust error handling.
        Used for high-volume tasks like annotation.
        Returns '7' if the input data fails the content inspection.
        """
        thread_info = f"[Thread {thread_id}] " if thread_id is not None else ""
        
        while True:
            try:
                response = self.gp_client.chat.completions.create(
                    model=self.gp_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=False
                )
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.gp_total_tokens += response.usage.total_tokens
                
                # On success, reset the error count and return the content
                self.error_counts["gp_model"] = 0
                content = response.choices[0].message.content
                return content
                
            except Exception as e:
                error_message_str = str(e)

                # Check if this is the specific data inspection error
                if 'data_inspection_failed' in error_message_str:
                    print(f"{thread_info}Data inspection failed. Returning '7' as per requirements.")
                    return '7' # Return '7' for this specific error and exit the function.

                # For all other errors, use the original retry logic
                self.error_counts["gp_model"] += 1
                current_time = time.time()

                if current_time - self.last_error_time["gp_model"] > 5:
                    self.last_error_time["gp_model"] = current_time
                
                time.sleep(1)
    
    def get_token_usage_summary(self) -> dict:
        """Returns a summary of token usage for both models."""
        return {
            'hp_tokens': self.hp_total_tokens,
            'gp_tokens': self.gp_total_tokens,
            'total_tokens': self.hp_total_tokens + self.gp_total_tokens
        }
    
    def print_token_usage(self):
        """Prints a formatted summary of token usage."""
        summary = self.get_token_usage_summary()
        print("\n" + "="*50)
        print("           Token Usage Summary")
        print("="*50)
        print(f"HP Model ({self.hp_model}): {summary['hp_tokens']:,} tokens")
        print(f"GP Model ({self.gp_model}): {summary['gp_tokens']:,} tokens")
        print(f"Total:                      {summary['total_tokens']:,} tokens")
        print("="*50)

