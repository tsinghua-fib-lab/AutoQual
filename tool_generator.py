# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import prompts
from llm_provider import LLMProvider
from feature_generator import _wait_for_user_confirmation

def _slugify(text: str) -> str:
    """
    Converts a string into a clean, file-safe name.
    Example: "Feature: Readability Score (Flesch-Kincaid)" -> "feature_readability_score_flesch_kincaid"
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'[\s-]+', '_', text.strip()) # Replace spaces and hyphens with underscores
    return text[:50] # Truncate to 50 chars

class ToolGenerator:
    """
    Responsible for generating annotation tools (code or prompts) for a given list of features.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        # Ensure tool directories exist
        if not os.path.exists(config.CODE_TOOLS_DIR):
            os.makedirs(config.CODE_TOOLS_DIR)
        if not os.path.exists(config.PROMPT_TOOLS_DIR):
            os.makedirs(config.PROMPT_TOOLS_DIR)

    def generate_all_tools(self, features_list_str: str, overwrite: bool):
        """
        Main method to orchestrate the entire tool generation process.
        """
        print("\n--- STAGE 4/5: TOOL ASSIGNMENT ---")
        features = [f.strip() for f in features_list_str.strip().split('\n') if f.strip()]
        
        assignments = self._get_tool_assignments(features, overwrite)
        
        assignment_filename = os.path.join(config.OUTPUT_DIR, "07_tool_type_assignments.csv")
        _wait_for_user_confirmation("Tool Type Assignment", [assignment_filename])

        # Read back assignments in case they were manually edited
        assignments_df = pd.read_csv(assignment_filename)
        
        print("\n--- STAGE 5/5: GENERATING TOOLS IN PARALLEL ---")
        generated_tools = self._generate_tools_parallelly(assignments_df, overwrite)
        _wait_for_user_confirmation("Tool Generation", generated_tools)
        
        print("\n--- ✅ Tool Generation Complete ---")

    def _get_tool_assignments(self, features: list, overwrite: bool) -> dict:
        """
        Determines whether to use 'CODE' or 'PROMPT' for each feature.
        In 'manual' mode, it creates a CSV for the user to fill.
        In 'auto' mode, it asks the LLM to decide.
        """
        assignments = {'feature': [], 'tool_type': []}
        assignment_file = os.path.join(config.OUTPUT_DIR, "07_tool_type_assignments.csv")
        
        if config.EXECUTION_MODE == 'manual' and not overwrite:
            print(f"Manual mode: Please define tool types in '{assignment_file}'.")
            # Create a template for the user if it doesn't exist
            if not os.path.exists(assignment_file):
                pd.DataFrame({
                    'feature': features, 
                    'tool_type': ['' for _ in features] # Leave blank for user
                }).to_csv(assignment_file, index=False)
            return

        print("Auto-determining tool types...")
        with ThreadPoolExecutor() as executor:
            future_to_feature = {executor.submit(self._decide_one_tool_type, f): f for f in features}
            for future in as_completed(future_to_feature):
                feature = future_to_feature[future]
                try:
                    tool_type = future.result().strip().upper()
                    if tool_type not in ["CODE", "PROMPT"]:
                        tool_type = "PROMPT" # Default to PROMPT on invalid response
                except Exception:
                    tool_type = "PROMPT" # Default on error
                
                assignments['feature'].append(feature)
                assignments['tool_type'].append(tool_type)
        
        pd.DataFrame(assignments).to_csv(assignment_file, index=False)
        print(f"Tool type assignments saved to '{assignment_file}'.")

    def _decide_one_tool_type(self, feature_description: str) -> str:
        prompt = prompts.DECIDE_TOOL_TYPE_PROMPT.format(feature_description=feature_description)
        # Use a temporary filename; we don't need to persist this decision junk
        filename = f"temp_decision_{_slugify(feature_description)}.txt"
        return self.llm.get_completion(prompt, filename, overwrite=True)

    def _generate_tools_parallelly(self, assignments_df: pd.DataFrame, overwrite: bool):
        """Generates all tool files in parallel based on the assignment dataframe."""
        generated_files = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for _, row in assignments_df.iterrows():
                futures.append(executor.submit(self._create_single_tool, row['feature'], row['tool_type'], overwrite))
            
            for future in as_completed(futures):
                try:
                    result_path = future.result()
                    if result_path:
                        generated_files.append(result_path)
                except Exception as e:
                    print(f"A tool generation task failed: {e}")
        return generated_files

    def _create_single_tool(self, feature: str, tool_type: str, overwrite: bool):
        """Dispatcher function to create a code or prompt tool."""
        feature_slug = _slugify(feature)
        
        if tool_type.upper() == 'CODE':
            return self._create_code_tool(feature_slug, feature, overwrite)
        elif tool_type.upper() == 'PROMPT':
            return self._create_prompt_tool(feature_slug, feature, overwrite)
        else:
            print(f"Warning: Unknown tool type '{tool_type}' for feature '{feature}'. Skipping.")
            return None

    def _create_code_tool(self, feature_slug: str, feature_desc: str, overwrite: bool):
        """Generates and saves a Python code tool with a unique function name."""
        function_name = f"annotate_{feature_slug}"
        filename = f"{feature_slug}.py"
        filepath = os.path.join(config.CODE_TOOLS_DIR, filename)
        
        if not overwrite and os.path.exists(filepath):
            print(f"Code tool '{filename}' already exists. Skipping generation.")
            return filepath
        
        prompt = prompts.GENERATE_CODE_TOOL_PROMPT.format(
            function_name=function_name,
            feature_name=feature_slug,
            feature_description=feature_desc
        )
        code_content = self.llm.get_completion(prompt, f"temp_code_{feature_slug}.txt", overwrite=True)
        
        # Clean up the response to get only the code
        code_match = re.search(r'```python\n(.*?)```', code_content, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"Generated code tool: {filepath}")
        return filepath

    def _create_prompt_tool(self, feature_slug: str, feature_desc: str, overwrite: bool):
        """Generates and saves a specific, reusable prompt template tool by calling the LLM."""
        filename = f"{feature_slug}.txt"
        filepath = os.path.join(config.PROMPT_TOOLS_DIR, filename)

        if not overwrite and os.path.exists(filepath):
            print(f"Prompt tool '{filename}' already exists. Skipping generation.")
            return filepath

        prompt = prompts.GENERATE_PROMPT_TOOL_PROMPT.format(
            feature_description=feature_desc,
        )
        
        prompt_template_content = self.llm.get_completion(prompt, f"temp_prompt_{feature_slug}.txt", overwrite=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt_template_content)
        print(f"Generated prompt tool: {filepath}")
        return filepath

