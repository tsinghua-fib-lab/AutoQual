# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pandas as pd
import config
import prompts
from llm_provider import LLMProvider
from feature_generator import FeatureGenerator
from tool_generator import ToolGenerator
from annotator import Annotator
from feature_selector import FeatureSelector
from tool_generator import _slugify
from feature_generator import _wait_for_user_confirmation

def _reflect_and_generate_features(llm_provider: LLMProvider, scene_description: str,
                                   best_features: list, feature_scores: dict, epoch: int,
                                   overwrite: bool) -> str:
    """
    Uses HP model to reflect on current features and generate new ones.
    
    Args:
        llm_provider: LLM provider instance
        scene_description: Task scenario description
        best_features: List of currently selected best features
        feature_scores: Dictionary mapping features to their scores
        epoch: Current epoch number
        overwrite: Whether to overwrite existing files
    
    Returns:
        String containing new feature descriptions
    """
    output_filename = f"08_epoch{epoch}_reflected_features.txt"
    output_filepath = os.path.join(config.OUTPUT_DIR, output_filename)
    
    # Check if file exists and we don't need to overwrite
    if not overwrite and os.path.exists(output_filepath):
        print(f"File '{output_filename}' already exists. Reading from cache.")
        with open(output_filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Sort features by score (descending)
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Format features with scores for the prompt
    features_with_scores = "\n".join([
        f"{i+1}. {feature}: {score:.4f}" 
        for i, (feature, score) in enumerate(sorted_features)
    ])
    
    # Generate prompt
    prompt = prompts.REFLECT_AND_GENERATE_FEATURES_PROMPT.format(
        scene_description=scene_description,
        features_with_scores=features_with_scores,
        new_feature_count=config.NEW_FEATURES_PER_EPOCH
    )
    
    # Call HP model
    new_features_str = llm_provider.get_completion(prompt, output_filename, overwrite=overwrite)
    
    return new_features_str


def _generate_tools_for_new_features(llm_provider: LLMProvider, new_features: list, 
                                     overwrite: bool, epoch: int) -> list:
    """
    Generates annotation tools for new features.
    In iterative epochs, all new features use PROMPT tools by default.
    
    Args:
        llm_provider: LLM provider instance
        new_features: List of new feature descriptions
        overwrite: Whether to overwrite existing tools
        epoch: Current epoch number
    
    Returns:
        List of generated tool file paths
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # In iterative epochs, directly use PROMPT for all new features
    print(f"Generating PROMPT tools for {len(new_features)} new features...")
    generated_tools = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for feature in new_features:
            # All new features in iteration use PROMPT tools
            future = executor.submit(_create_single_tool_for_new_feature, 
                                    llm_provider, feature, "PROMPT", epoch, overwrite)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result_path = future.result()
                if result_path:
                    generated_tools.append(result_path)
            except Exception as e:
                print(f"Tool generation failed: {e}")
    
    print(f"Generated {len(generated_tools)} PROMPT tools for epoch {epoch}")
    return generated_tools


def _create_single_tool_for_new_feature(llm_provider: LLMProvider, feature: str, 
                                       tool_type: str, epoch: int, overwrite: bool):
    """Creates a single CODE or PROMPT tool for a new feature."""
    import re
    
    feature_slug = _slugify(feature)
    
    if tool_type.upper() == 'CODE':
        function_name = f"annotate_{feature_slug}"
        filename = f"{feature_slug}.py"
        filepath = os.path.join(config.CODE_TOOLS_DIR, filename)
        
        if not overwrite and os.path.exists(filepath):
            print(f"Code tool '{filename}' already exists. Skipping generation.")
            return filepath
        
        prompt = prompts.GENERATE_CODE_TOOL_PROMPT.format(
            function_name=function_name,
            feature_name=feature_slug,
            feature_description=feature
        )
        code_content = llm_provider.get_completion(
            prompt, f"temp_code_epoch{epoch}_{feature_slug}.txt", overwrite=True
        )
        
        # Clean up the response
        code_match = re.search(r'```python\n(.*?)```', code_content, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code_content)
        print(f"Generated code tool: {filepath}")
        return filepath
        
    elif tool_type.upper() == 'PROMPT':
        filename = f"{feature_slug}.txt"
        filepath = os.path.join(config.PROMPT_TOOLS_DIR, filename)
        
        if not overwrite and os.path.exists(filepath):
            print(f"Prompt tool '{filename}' already exists. Skipping generation.")
            return filepath
        
        prompt = prompts.GENERATE_PROMPT_TOOL_PROMPT.format(
            feature_description=feature,
        )
        
        prompt_template_content = llm_provider.get_completion(
            prompt, f"temp_prompt_epoch{epoch}_{feature_slug}.txt", overwrite=True
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt_template_content)
        print(f"Generated prompt tool: {filepath}")
        return filepath
    
    return None


def _annotate_new_features(llm_provider: LLMProvider, new_features: list, epoch: int, 
                          overwrite: bool) -> pd.DataFrame:
    """
    Annotates new features on the existing dataset.
    
    Args:
        llm_provider: LLM provider instance
        new_features: List of new feature descriptions
        epoch: Current epoch number
        overwrite: Whether to overwrite existing annotations
    
    Returns:
        Updated DataFrame with new feature columns
    """
    # Check if epoch annotated data already exists
    epoch_annotated_file = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch}_annotated_data.csv")
    
    if not overwrite and os.path.exists(epoch_annotated_file):
        print(f"Epoch {epoch} annotated data already exists. Reading from cache.")
        return pd.read_csv(epoch_annotated_file)
    
    # Load current annotated data
    df = pd.read_csv(config.ANNOTATED_DATA_FILE)
    
    # Create a temporary annotator to annotate just the new features
    annotator = Annotator(llm_provider, df)
    
    # Filter to only new feature tools
    new_feature_slugs = [_slugify(f) for f in new_features]
    
    # Filter code and prompt tools to only new features
    new_code_tools = {slug: func for slug, func in annotator.code_tools.items() 
                     if slug in new_feature_slugs}
    new_prompt_tools = {slug: template for slug, template in annotator.prompt_tools.items() 
                       if slug in new_feature_slugs}
    
    print(f"Annotating with {len(new_code_tools)} code tools and {len(new_prompt_tools)} prompt tools")
    
    # Apply code tools
    if new_code_tools:
        print("Applying code-based tools...")
        for slug, func in new_code_tools.items():
            if slug not in df.columns:
                df[slug] = df['text'].apply(lambda text: annotator._apply_code_tool(func, text))
    
    # Apply prompt tools
    if new_prompt_tools:
        print("Applying prompt-based tools...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        for slug, prompt_template in new_prompt_tools.items():
            if slug not in df.columns:
                df[slug] = None
                
                with ThreadPoolExecutor(max_workers=config.ANNOTATION_MAX_WORKERS) as executor:
                    futures = {
                        executor.submit(annotator._apply_prompt_tool, prompt_template, 
                                      df.loc[idx, 'text'], slug): idx
                        for idx in df.index
                    }
                    
                    for future in tqdm(as_completed(futures), total=len(futures), 
                                     desc=f"Annotating {slug}"):
                        idx = futures[future]
                        try:
                            score = future.result()
                            df.loc[idx, slug] = score
                        except Exception as e:
                            print(f"Error annotating row {idx} for {slug}: {e}")
                            df.loc[idx, slug] = None
    
    # Save epoch-specific annotated data
    df.to_csv(epoch_annotated_file, index=False)
    print(f"Epoch {epoch} annotated data saved to '{epoch_annotated_file}'")
    
    return df


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='AutoQual: Automatic Feature Discovery Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  # Use default settings from config file
  python main.py
  
  # Specify scene
  python main.py --scene Amazon_grocery
        """
    )
    
    parser.add_argument(
        '--scene', '-s',
        type=str,
        default=None,
        help=f'Scene name (default: {config.SCENE_NAME})'
    )
    
    return parser.parse_args()


def update_config_from_args(args):
    """
    Update config based on command line arguments.
    """
    import glob
    
    # Update SCENE_NAME
    if args.scene is not None:
        config.SCENE_NAME = args.scene
        print(f"[Command Line] SCENE_NAME set to: {config.SCENE_NAME}")
    
    # Recalculate derived paths
    config.SCENE_DATA_DIR = os.path.join(config.DATA_DIR_BASE, config.SCENE_NAME)
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR_BASE, config.SCENE_NAME)
    
    # Recalculate scene description file
    config.SCENE_DESCRIPTION_FILE = os.path.join(config.SCENE_DATA_DIR, "scene_description.txt")
    
    # Re-search for data file
    csv_files = glob.glob(os.path.join(config.SCENE_DATA_DIR, "*.csv"))
    if not csv_files:
        config.DATA_FILE = f"Error: No CSV file found in scene directory '{config.SCENE_DATA_DIR}'"
    elif len(csv_files) > 1:
        print(f"Warning: Found multiple CSV files in {config.SCENE_DATA_DIR}. Using the first one: {csv_files[0]}")
        config.DATA_FILE = csv_files[0]
    else:
        config.DATA_FILE = csv_files[0]
    
    # Recalculate tool paths
    config.TOOLS_DIR = os.path.join(config.OUTPUT_DIR, "tools")
    config.CODE_TOOLS_DIR = os.path.join(config.TOOLS_DIR, "code")
    config.PROMPT_TOOLS_DIR = os.path.join(config.TOOLS_DIR, "prompts")
    
    # Recalculate output file paths
    config.TRAIN_DATA_FILE = os.path.join(config.OUTPUT_DIR, "train_data.csv")
    config.TEST_DATA_FILE = os.path.join(config.OUTPUT_DIR, "test_data.csv")
    config.CODE_ANNOTATED_FILE = os.path.join(config.OUTPUT_DIR, "intermediate_code_annotated.csv")
    config.ANNOTATED_DATA_FILE = os.path.join(config.OUTPUT_DIR, "final_annotated_data.csv")
    config.BEST_FEATURES_FILE = os.path.join(config.OUTPUT_DIR, "best_features.txt")


def main():
    """
    Main function, entry point of the project.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Update config based on command line arguments
    update_config_from_args(args)
    
    # Check if input files exist
    if not os.path.exists(config.SCENE_DESCRIPTION_FILE):
        print(f"Error: Scene description file not found at: {config.SCENE_DESCRIPTION_FILE}")
        return
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Data file not found at: {config.DATA_FILE}")
        return

    # Load inputs
    try:
        with open(config.SCENE_DESCRIPTION_FILE, 'r', encoding='utf-8') as f:
            scene_description = f.read()
        
        df = pd.read_csv(config.DATA_FILE)
        
        # Validate CSV file format
        if 'text' not in df.columns or 'score' not in df.columns:
            print("Error: CSV file must contain 'text' and 'score' columns.")
            return
            
    except Exception as e:
        print(f"Error loading input files: {e}")
        return

    print("="*50)
    print(" AutoQual: Automatic Feature Discovery Agent")
    print("="*50)
    print(f"Current mode: {config.EXECUTION_MODE}")
    print(f"Scene description file: {config.SCENE_DESCRIPTION_FILE}")
    print(f"Data file: {config.DATA_FILE}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("="*50)

    try:
        # Initialize core modules
        llm_provider = LLMProvider()
        feature_generator = FeatureGenerator(llm_provider)

        # Execute Part 1: Generate initial feature pool
        final_features_str = feature_generator.generate_initial_features(scene_description, df)

        if not final_features_str or final_features_str.startswith("Error"):
            print("\nCritical error in feature generation. Aborting.")
            return

        # Execute Part 2: Generate annotation tools
        tool_generator = ToolGenerator(llm_provider)
        tool_generator.generate_all_tools(final_features_str, feature_generator.overwrite_files)
        
        # Execute Part 3: Feature annotation
        print("\n--- STAGE 6/6: ANNOTATING FEATURES WITH ALL TOOLS ---")
        annotator = Annotator(llm_provider, df)
        annotated_df = annotator.annotate_features()

        if annotated_df.empty:
            print("\nAnnotation resulted in an empty dataframe. Aborting feature selection.")
            return

        # --- Part 4: Feature Selection with Iterative Optimization ---
        print("\n" + "="*50)
        print(f"  Starting Iterative Feature Optimization")
        print(f"  Total Epochs: 1 initial + {config.ITERATION_EPOCHS} iterations")
        print("="*50)
        
        # Ask about overwrite for iteration in manual mode
        overwrite_iteration = feature_generator.overwrite_files
        
        # Initial feature selection (Epoch 0)
        print("\n--- EPOCH 0: INITIAL FEATURE SELECTION ---")
        epoch_0_results_file = os.path.join(config.OUTPUT_DIR, "epoch_0_best_features.txt")
        
        if not overwrite_iteration and os.path.exists(epoch_0_results_file):
            print(f"Epoch 0 results already exist. Reading from cache.")
            # Load best features from file
            best_features = []
            feature_scores = {}
            with open(epoch_0_results_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        parts = line.strip().rsplit(':', 1)
                        feature = parts[0].strip()
                        score = float(parts[1].strip())
                        best_features.append(feature)
                        feature_scores[feature] = score
        else:
            selector = FeatureSelector(config.ANNOTATED_DATA_FILE)
            best_features, feature_scores = selector.select_features(return_scores=True)
            
            # Save epoch 0 results
            with open(epoch_0_results_file, 'w', encoding='utf-8') as f:
                for feature in best_features:
                    score = feature_scores.get(feature, 0.0)
                    f.write(f"{feature}: {score:.4f}\n")
            print(f"Epoch 0 results saved to '{epoch_0_results_file}'")
        
        _wait_for_user_confirmation(
            f"Epoch 0 - Initial Feature Selection",
            [epoch_0_results_file]
        )
        
        # Iterative optimization loop
        for epoch in range(1, config.ITERATION_EPOCHS + 1):
            print("\n" + "="*50)
            print(f"  EPOCH {epoch}/{config.ITERATION_EPOCHS}: REFLECTION & OPTIMIZATION")
            print("="*50)
            
            # Step 1: HP Model Reflection - Generate new features
            print(f"\n--- Epoch {epoch} - Step 1/4: HP Model Reflecting ---")
            new_features_str = _reflect_and_generate_features(
                llm_provider, scene_description, best_features, feature_scores, 
                epoch, overwrite_iteration
            )
            
            if not new_features_str or new_features_str.startswith("Error"):
                print(f"Warning: Failed to generate new features in epoch {epoch}. Stopping iteration.")
                break
            
            # Parse new features
            new_features = [f.strip() for f in new_features_str.strip().split('\n') if f.strip()]
            print(f"Generated {len(new_features)} new features for epoch {epoch}")
            
            reflection_file = os.path.join(config.OUTPUT_DIR, f"08_epoch{epoch}_reflected_features.txt")
            _wait_for_user_confirmation(
                f"Epoch {epoch} - Step 1: Feature Reflection",
                [reflection_file]
            )
            
            # Re-load features in case user modified them
            with open(reflection_file, 'r', encoding='utf-8') as f:
                new_features_str = f.read()
            new_features = [f.strip() for f in new_features_str.strip().split('\n') if f.strip()]
            
            # Step 2: Generate tools for new features
            print(f"\n--- Epoch {epoch} - Step 2/4: Generating Tools ---")
            generated_tools = _generate_tools_for_new_features(
                llm_provider, new_features, overwrite_iteration, epoch
            )
            
            _wait_for_user_confirmation(
                f"Epoch {epoch} - Step 2: Tool Generation",
                generated_tools
            )
            
            # Step 3: Annotate new features on the existing dataset
            print(f"\n--- Epoch {epoch} - Step 3/4: Annotating New Features ---")
            annotated_df = _annotate_new_features(
                llm_provider, new_features, epoch, overwrite_iteration
            )
            
            if annotated_df.empty:
                print(f"Warning: Annotation failed in epoch {epoch}. Stopping iteration.")
                break
            
            # Update the annotated data file
            annotated_df.to_csv(config.ANNOTATED_DATA_FILE, index=False)
            print(f"Updated annotated data saved to '{config.ANNOTATED_DATA_FILE}'")
            
            epoch_annotated_file = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch}_annotated_data.csv")
            _wait_for_user_confirmation(
                f"Epoch {epoch} - Step 3: Feature Annotation",
                [epoch_annotated_file]
            )
            
            # Step 4: Re-run feature selection with expanded feature pool
            print(f"\n--- Epoch {epoch} - Step 4/4: Re-selecting Best Features ---")
            epoch_results_file = os.path.join(config.OUTPUT_DIR, f"epoch_{epoch}_best_features.txt")
            
            if not overwrite_iteration and os.path.exists(epoch_results_file):
                print(f"Epoch {epoch} results already exist. Reading from cache.")
                best_features = []
                feature_scores = {}
                with open(epoch_results_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if ':' in line:
                            parts = line.strip().rsplit(':', 1)
                            feature = parts[0].strip()
                            score = float(parts[1].strip())
                            best_features.append(feature)
                            feature_scores[feature] = score
            else:
                selector = FeatureSelector(config.ANNOTATED_DATA_FILE)
                best_features, feature_scores = selector.select_features(return_scores=True)
                
                # Save epoch results
                with open(epoch_results_file, 'w', encoding='utf-8') as f:
                    for feature in best_features:
                        score = feature_scores.get(feature, 0.0)
                        f.write(f"{feature}: {score:.4f}\n")
                print(f"Epoch {epoch} results saved to '{epoch_results_file}'")
            
            _wait_for_user_confirmation(
                f"Epoch {epoch} - Step 4: Feature Selection",
                [epoch_results_file]
            )

        print("\n" + "="*50)
        print("      AutoQual process completed successfully!")
        print("="*50)
        print("Final candidate features are in 'output/06_integrated_features.txt'")
        print("Generated annotation tools are in the 'tools/' directory.")
        print(f"Final annotated data is in '{config.ANNOTATED_DATA_FILE}'")
        print(f"Best feature set is in '{config.BEST_FEATURES_FILE}'")
        print(f"Epoch-wise results are in 'epoch_*_best_features.txt' files")
        print("="*50)
        
        # Print token usage summary
        llm_provider.print_token_usage()

    except ValueError as ve:
        print(f"\nConfiguration error: {ve}")
    except Exception as e:
        print(f"\nUnexpected error during execution: {e}")


if __name__ == "__main__":
    main()

