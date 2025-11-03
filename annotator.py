# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import config
from llm_provider import LLMProvider

class Annotator:
    """
    Handles the entire feature annotation process using a two-stage approach:
    1. Fast annotation using all code-based tools.
    2. Slower, parallel annotation using all prompt-based tools with checkpointing.
    """
    def __init__(self, llm_provider: LLMProvider, data_df: pd.DataFrame):
        self.llm = llm_provider
        self.data_df = data_df
        self.code_tools, self.prompt_tools = self._load_all_tools()
        self.checkpoint_file = os.path.join(config.OUTPUT_DIR, "checkpoint_annotated_data.csv")

    def _load_all_tools(self):
        """Dynamically loads all code and prompt tools from their respective directories."""
        code_tools = {}
        prompt_tools = {}
        if not os.path.exists(config.CODE_TOOLS_DIR) or not os.path.exists(config.PROMPT_TOOLS_DIR):
            print("Warning: Tools directory not found. Skipping tool loading.")
            return {}, {}
        # Load Code Tools
        for filename in os.listdir(config.CODE_TOOLS_DIR):
            if filename.endswith(".py"):
                filepath = os.path.join(config.CODE_TOOLS_DIR, filename)
                feature_slug = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(feature_slug, filepath)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    function_name = f"annotate_{feature_slug}"
                    if hasattr(module, function_name):
                        code_tools[feature_slug] = getattr(module, function_name)
                    else:
                         print(f"Warning: Function '{function_name}' not found in '{filename}'. Skipping.")
                except Exception as e:
                    print(f"Warning: Failed to load code tool from '{filename}'. Error: {e}")
        # Load Prompt Tools
        for filename in os.listdir(config.PROMPT_TOOLS_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(config.PROMPT_TOOLS_DIR, filename)
                feature_slug = filename[:-4]
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompt_tools[feature_slug] = f.read()
        print(f"Loaded {len(code_tools)} code tools and {len(prompt_tools)} prompt tools.")
        return code_tools, prompt_tools

    def annotate_features(self) -> pd.DataFrame:
        """
        Orchestrates the two-stage annotation process.
        """
        # Stage 1: Apply code-based tools
        df_with_code_features = self._apply_all_code_tools()
        df_with_code_features.to_csv(config.CODE_ANNOTATED_FILE, index=False)
        print(f"\n✅ Code-based annotation complete. Intermediate results saved to: {config.CODE_ANNOTATED_FILE}")

        # Stage 2: Apply prompt-based tools
        final_df = self._apply_all_prompt_tools(df_with_code_features)

        # Save the final, complete dataframe
        final_df.to_csv(config.ANNOTATED_DATA_FILE, index=False)
        print(f"\nAnnotation complete. Final data saved to '{config.ANNOTATED_DATA_FILE}'")
        return final_df

    def _apply_all_code_tools(self) -> pd.DataFrame:
        """Applies all loaded code tools to the dataframe. This is a fast, in-memory operation."""
        print("\n--- Applying Code-based Tools ---")
        if not self.code_tools:
            print("No code tools found. Skipping.")
            return self.data_df.copy()

        df_copy = self.data_df.copy()
        for slug, func in tqdm(self.code_tools.items(), desc="Code Tools"):
            df_copy[slug] = df_copy['text'].apply(lambda text: self._apply_code_tool(func, text))
        
        return df_copy

    def _apply_all_prompt_tools(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Applies all prompt tools with high parallelism and checkpointing."""
        print("\n--- Applying Prompt-based Tools (GP Model) ---")
        print(f"-> Intermediate progress will be checkpointed to: {self.checkpoint_file}")
        if not self.prompt_tools:
            print("No prompt tools found. Skipping.")
            return input_df

        # Load from checkpoint if it exists
        if os.path.exists(self.checkpoint_file):
            print(f"Found checkpoint file: '{self.checkpoint_file}'. Resuming...")
            df_to_process = pd.read_csv(self.checkpoint_file)
        else:
            df_to_process = input_df.copy()

        # Identify which rows still need annotation for any of the prompt tools
        # A row needs processing if any prompt tool column is missing or has a null value
        prompt_slugs = list(self.prompt_tools.keys())
        all_cols_exist = all(col in df_to_process.columns for col in prompt_slugs)
        
        if all_cols_exist:
             rows_to_process_indices = df_to_process[df_to_process[prompt_slugs].isnull().any(axis=1)].index
        else:
             rows_to_process_indices = df_to_process.index

        if rows_to_process_indices.empty:
            print("All rows appear to be annotated. Skipping prompt annotation.")
            return df_to_process

        print(f"Found {len(rows_to_process_indices)} rows needing prompt annotations.")
        
        tasks_to_submit = []
        for index in rows_to_process_indices:
            for slug in prompt_slugs:
                if slug not in df_to_process.columns or pd.isnull(df_to_process.loc[index, slug]):
                    tasks_to_submit.append({'index': index, 'slug': slug})
        
        if not tasks_to_submit:
            print("All cells for all rows appear to be annotated. Skipping prompt annotation.")
            return df_to_process
            
        print(f"Found {len(tasks_to_submit)} individual prompt annotation tasks to run.")

        with ThreadPoolExecutor(max_workers=config.ANNOTATION_MAX_WORKERS) as executor:
            # Create a future for each text that needs processing for each prompt tool
            futures = {
                executor.submit(self._apply_prompt_tool, self.prompt_tools[task['slug']], df_to_process.loc[task['index'], 'text'], task['slug']): (task['index'], task['slug'])
                for task in tasks_to_submit
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="GP Model Annotations"):
                index, slug = futures[future]
                try:
                    score = future.result()
                    df_to_process.loc[index, slug] = score
                except Exception as e:
                    print(f"Error processing prompt tool '{slug}' for row {index}: {e}")
                    df_to_process.loc[index, slug] = None # Mark as failed

                # Checkpoint saving logic
                if (df_to_process.loc[index].notnull().sum()) % config.ANNOTATION_CHECKPOINT_INTERVAL == 0:
                    df_to_process.to_csv(self.checkpoint_file, index=False)
        
        # Final save after the loop completes
        df_to_process.to_csv(self.checkpoint_file, index=False)
        return df_to_process

    def _apply_code_tool(self, tool_function, text: str) -> float:
        """Wrapper to safely execute a code tool."""
        try:
            return tool_function(text)
        except Exception as e:
            print(f"-> WARNING: Code tool '{tool_function.__name__}' failed on a text with error: {e}")
            return None

    def _apply_prompt_tool(self, prompt_template: str, text: str, slug: str) -> float:
        """Wrapper to execute a prompt tool using the general-purpose LLM."""
        full_prompt = prompt_template.replace("[TEXT_TO_EVALUATE]", text)
        # Using thread_id from os.getpid() can help trace logs
        response_text = self.llm.call_general_model(full_prompt, thread_id=os.getpid())
        
        match = re.search(r'[-+]?\d*\.?\d+', response_text)
        if match:
            return float(match.group())
        else:
            print(f"-> WARNING: Prompt tool '{slug}' could not parse a number from LLM response: '{response_text}'")
            return None

