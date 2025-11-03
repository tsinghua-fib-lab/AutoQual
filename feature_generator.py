# -*- coding: utf-8 -*-

import pandas as pd
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import prompts
from llm_provider import LLMProvider

def _wait_for_user_confirmation(stage_name: str, generated_files: list[str]):
    """If in manual mode, pauses execution and waits for user input."""
    if config.EXECUTION_MODE != 'manual':
        return

    print("\n" + "="*50)
    print(f"  STAGE COMPLETE: {stage_name}")
    print("="*50)
    print("The following files have been generated/updated:")
    for f in generated_files:
        print(f"  - {f}")
    print("\nYou can now review and modify these files in the 'output' directory.")
    input("Press Enter to continue to the next stage...")
    print("="*50)


class FeatureGenerator:
    """
    Responsible for generating the initial candidate feature pool using a 3-stage process.
    Supports both 'auto' and 'manual' modes with stage-based interaction.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.overwrite_files = False

    def generate_initial_features(self, scene_description: str, df: pd.DataFrame) -> str:
        """
        Executes the 3-stage workflow to generate the initial feature pool.
        """
        # In manual mode, ask user if they want to overwrite existing files
        if config.EXECUTION_MODE == 'manual':
            choice = input("Do you want to overwrite and regenerate existing files? (y/n, default is n): ").lower().strip()
            self.overwrite_files = (choice == 'y')

        # --- STAGE 1: Role Generation ---
        print("\n--- STAGE 1/3: GENERATING ROLES ---")
        role_filename = self._generate_roles(scene_description)
        _wait_for_user_confirmation("Role Generation", [role_filename])
        # Load roles from file (might have been edited by the user)
        roles = self._load_roles_from_file(role_filename)
        
        # --- STAGE 2: Parallel Feature Generation ---
        print("\n--- STAGE 2/3: GENERATING FEATURES IN PARALLEL ---")
        feature_files = self._generate_all_features_parallelly(scene_description, roles, df)
        _wait_for_user_confirmation("Feature Generation", feature_files)
        
        # --- STAGE 3: Feature Integration ---
        print("\n--- STAGE 3/3: INTEGRATING FEATURES ---")
        final_features_filename = self._integrate_features(feature_files)
        _wait_for_user_confirmation("Feature Integration", [final_features_filename])

        # Load the final result from the file
        final_filepath = os.path.join(config.OUTPUT_DIR, final_features_filename)
        with open(final_filepath, 'r', encoding='utf-8') as f:
            final_features = f.read()

        print("\n--- ✅ Initial Feature Pool Generation Complete ---")
        return final_features

    def _load_roles_from_file(self, role_filename: str) -> list[str]:
        """Reads the roles file after user may have edited it."""
        filepath = os.path.join(config.OUTPUT_DIR, role_filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            roles_text = f.read()
        roles = [role.strip() for role in roles_text.strip().split('\n') if role.strip()]
        print(f"Loaded {len(roles)} roles from '{role_filename}' to proceed.")
        return roles

    def _generate_roles(self, scene_description: str) -> str:
        """Generates roles and returns the output filename."""
        filename = "01_generated_roles.txt"
        prompt = prompts.GENERATE_ROLES_PROMPT.format(
            scene_description=scene_description,
            role_count=config.ROLE_COUNT
        )
        self.llm.get_completion(prompt, filename, self.overwrite_files)
        return filename

    def _generate_all_features_parallelly(self, scene_description: str, roles: list[str], df: pd.DataFrame) -> list[str]:
        """
        Submits all feature generation tasks to a thread pool for parallel execution.
        Returns a list of generated filenames.
        """
        df_sorted = df.sort_values(by='score', ascending=False)
        pos_samples = df_sorted.head(config.SAMPLE_COUNT)['text'].tolist()
        neg_samples = df_sorted.tail(config.SAMPLE_COUNT)['text'].tolist()
        
        # Calculate max workers: role tasks + 3 data tasks
        max_workers = config.ROLE_COUNT + 3
        
        generated_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary to map futures to their task name for better error reporting
            futures = {}
            
            # Role-based tasks
            for i, role in enumerate(roles):
                fut = executor.submit(self._get_features_for_role, scene_description, role, i)
                futures[fut] = f"Role {i+1}"
            
            # Data-based tasks
            fut_pos = executor.submit(self._get_features_from_positive, scene_description, pos_samples)
            futures[fut_pos] = "Positive Samples"
            fut_neg = executor.submit(self._get_features_from_negative, scene_description, neg_samples)
            futures[fut_neg] = "Negative Samples"
            fut_con = executor.submit(self._get_features_from_contrastive, scene_description, pos_samples, neg_samples)
            futures[fut_con] = "Contrastive Analysis"
            
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    # The result of each task is the filename
                    filename = future.result()
                    generated_files.append(filename)
                except Exception as e:
                    print(f"Task '{task_name}' failed during execution: {e}")
        
        return sorted(generated_files)

    def _integrate_features(self, feature_files: list[str]) -> str:
        """
        Reads content from feature files, integrates them, and returns the final filename.
        """
        feature_docs = []
        for file in feature_files:
            filepath = os.path.join(config.OUTPUT_DIR, file)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    feature_docs.append(f.read())
            except FileNotFoundError:
                print(f"Warning: Could not find feature file '{file}'. Skipping.")

        combined_features = "\n\n".join(filter(None, feature_docs))
        prompt = prompts.INTEGRATE_FEATURES_PROMPT.format(feature_list=combined_features)
        
        filename = "06_integrated_features.txt"
        self.llm.get_completion(prompt, filename, self.overwrite_files)
        return filename
    
    # --- Single-Responsibility Feature Generation Methods ---
    # These methods now return the filename of the generated file.
    
    def _get_features_for_role(self, scene_description: str, role: str, index: int) -> str:
        filename = f"02_role_{index+1}_features.txt"
        prompt = prompts.GENERATE_FEATURES_FROM_ROLE_PROMPT.format(
            scene_description=scene_description,
            role_description=role,
            feature_count_per_role=config.FEATURE_COUNT_PER_ROLE
        )
        self.llm.get_completion(prompt, filename, self.overwrite_files)
        return filename

    def _get_features_from_positive(self, scene_description: str, samples: list[str]) -> str:
        filename = "03_data_positive_features.txt"
        samples_str = "\n".join([f"- {s}" for s in samples])
        prompt = prompts.ANALYZE_POSITIVE_SAMPLES_PROMPT.format(
            scene_description=scene_description,
            samples=samples_str,
            feature_count_positive=config.FEATURE_COUNT_POSITIVE
        )
        self.llm.get_completion(prompt, filename, self.overwrite_files)
        return filename

    def _get_features_from_negative(self, scene_description: str, samples: list[str]) -> str:
        filename = "04_data_negative_features.txt"
        samples_str = "\n".join([f"- {s}" for s in samples])
        prompt = prompts.ANALYZE_NEGATIVE_SAMPLES_PROMPT.format(
            scene_description=scene_description,
            samples=samples_str,
            feature_count_negative=config.FEATURE_COUNT_NEGATIVE
        )
        self.llm.get_completion(prompt, filename, self.overwrite_files)
        return filename

    def _get_features_from_contrastive(self, scene_description: str, positive_samples: list[str], negative_samples: list[str]) -> str:
        filename = "05_data_contrastive_features.txt"
        pos_samples_str = "\n".join([f"- {s}" for s in positive_samples])
        neg_samples_str = "\n".join([f"- {s}" for s in negative_samples])
        prompt = prompts.ANALYZE_CONTRASTIVE_SAMPLES_PROMPT.format(
            scene_description=scene_description,
            positive_samples=pos_samples_str,
            negative_samples=neg_samples_str,
            feature_count_contrastive=config.FEATURE_COUNT_CONTRASTIVE
        )
        self.llm.get_completion(prompt, filename, self.overwrite_files)
        return filename

