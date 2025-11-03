# -*- coding: utf-8 -*-

"""
This file stores all prompt templates for interacting with the Large Language Model.
"""

# Prompt for generating roles
GENERATE_ROLES_PROMPT = """
We are undertaking a text quality assessment task. The core of this task is to evaluate the quality of text based on the following scenario.

Scenario Description:
---
{scene_description}
---

Your task is to propose {role_count} distinct virtual evaluator roles, each with a different perspective and unique evaluation criteria, based on the text quality requirements of the above scenario. These roles should be representative and cover multiple dimensions of text quality assessment.

Output a list of exactly {role_count} roles. Each role must be on a new line.
Only output {role_count} lines. For each line, provide only the role's name followed by a comma and a concise description of its core evaluation criteria (around 20 words). Do not include any extra explanations, introductory text, or formatting like "Role 1:".

Example Format:
[Role Name 1], [Core evaluation criteria description]
[Role Name 2], [Core evaluation criteria description]
"""

# Prompt for generating features from a single role's perspective
GENERATE_FEATURES_FROM_ROLE_PROMPT = """
We are designing computable and interpretable features for a text quality assessment task. The task scenario is as follows:
---
{scene_description}
---

Now, please fully embody the following role and think from its perspective. The role and its evaluation angle are: {role_description}.
Your task is to propose a set of candidate features for measuring text quality based on your role and evaluation criteria. These features should be concrete, measurable, and interpretable. The feature description must be clear enough for someone to understand how to evaluate the text based on it.

Output exactly {feature_count_per_role} of what you consider the most important features. Each feature must be on a new line, and should be about 30 words.
Only output {feature_count_per_role} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.

Example Format:
[Feature 1], [Feature description]
[Feature 2], [Feature description]
"""

# Prompt for analyzing common features in high-quality texts
ANALYZE_POSITIVE_SAMPLES_PROMPT = """
We are designing features for a text quality assessment task. The task scenario is as follows:
---
{scene_description}
---

We have selected a batch of texts identified as high-score under a certain evaluation system. Here are some of those samples:
---
{samples}
---

Your task is to carefully analyze these high-score text samples and summarize the common features they possess that could explain their high scores. These features should be concrete, measurable, and interpretable. The feature description must be clear enough for someone to understand how to evaluate the text based on it.

Output exactly {feature_count_positive} most important features. Each feature must be on a new line, and should be about 30 words.
Only output {feature_count_positive} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.

Example Format:
[Feature 1], [Feature description]
[Feature 2], [Feature description]
"""

# Prompt for analyzing common features in low-quality texts
ANALYZE_NEGATIVE_SAMPLES_PROMPT = """
We are designing features for a text quality assessment task. The task scenario is as follows:
---
{scene_description}
---

We have selected a batch of texts identified as low-score under a certain evaluation system. Here are some of those samples:
---
{samples}
---

Your task is to carefully analyze these low-score text samples and summarize the common features they possess that could explain their low scores. These features should be concrete, measurable, and interpretable. The feature description must be clear enough for someone to understand how to evaluate the text based on it.

Output exactly {feature_count_negative} most important features. Each feature must be on a new line, and should be about 30 words.
Only output {feature_count_negative} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.

Example Format:
[Feature 1], [Feature description]
[Feature 2], [Feature description]
"""

# Prompt for contrastive analysis between high and low-quality texts
ANALYZE_CONTRASTIVE_SAMPLES_PROMPT = """
We are designing features for a text quality assessment task. The task scenario is as follows:
---
{scene_description}
---

We have selected batches of texts identified as high-score and low-score under a certain evaluation system.

High-Score Samples:
---
{positive_samples}
---

Low-Score Samples:
---
{negative_samples}
---

Your task is to conduct a contrastive analysis of the two sample sets and summarize the most significant distinguishing features. These should be features that high-score texts possess and low-score texts lack. The features must be concrete, measurable, and interpretable. The feature description must be clear enough for someone to understand how to evaluate the text based on it.

Output exactly {feature_count_contrastive} of the most distinctive features. Each feature must be on a new line, and should be about 30 words.
Only output {feature_count_contrastive} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.

Example Format:
[Feature 1], [Feature description]
[Feature 2], [Feature description]
"""

# Prompt for integrating, deduplicating, and refining the feature list
INTEGRATE_FEATURES_PROMPT = """
We have generated a batch of candidate features for text quality assessment through various methods (multi-role perspectives, data sample analysis, etc.). They now need to be consolidated.

Original Feature List:
---
{feature_list}
---

As a feature engineering expert, your task is to process the original feature list above to produce a final, refined pool of candidate features.

Processing Requirements:
1. Merge and Deduplicate: Identify and merge features that are semantically identical or highly similar.
2. Optimize Descriptions: Ensure each feature's description is clear, precise, unambiguous, and actionable for the subsequent development of annotation tools.
3. Format Output: Organize the output into a clean list.

Output each feature on a new line. For each feature, provide only a detailed text description of what it measures.
The final list should contain as many unique features as can be derived from the original list after processing.
Just output a plain list of features. Do not use any special symbols or formatting, including list numbers.
Start a new line ONLY when moving to the next feature. If you find n features, just output n lines, with each line containing a distinct feature.
"""

# --- Part 2: Tool Generation Prompts ---

DECIDE_TOOL_TYPE_PROMPT = """
Your task is to determine the best tool type to annotate a text feature. The options are "CODE" or "PROMPT".

- "CODE" is for features that are simple, explicit, and can be accurately auto-annotated with Python code without requiring intelligent processing.
- "PROMPT" is for features that cannot be accurately annotated with simple rule-based code, and may be abstract, nuanced, subjective, or require deep semantic understanding.

Feature Description:
---
{feature_description}
---

Based on the description, is this feature better suited for "CODE" or "PROMPT"?
Respond with a single word: either "CODE" or "PROMPT". Do not provide any other text or explanation.
"""

GENERATE_CODE_TOOL_PROMPT = """
Your task is to write a Python function that serves as an annotation tool for a specific text feature.

The function should:
1.  Be named `{function_name}`.
2.  Accept a single string argument named `text`.
3.  Return a single numerical value (float or int).
4.  Be self-contained. You can use common libraries like `re`, `nltk`, `textblob`, but do not assume any external files are available.
5.  If a library is used, include the necessary import statement inside the function to ensure it's encapsulated.

Here is the feature the function needs to measure:
---
Feature: {feature_name}
Description: {feature_description}
---

Generate the complete Python code for this function. Do not include any text or explanation outside the function's code block. Start the response directly with the function definition.

Example:
```python
def annotate(text: str) -> float:
    # import necessary libraries here
    # ... function logic ...
    return score
```
"""

GENERATE_PROMPT_TOOL_PROMPT = """
Your task is to create an precise and effective prompt template for a Large Language Model to use as a feature annotation tool.

This template will be used to evaluate different pieces of text. It must contain the placeholder [TEXT_TO_EVALUATE] where the actual text will be inserted later.

The prompt you create should instruct the LLM to:
1.  The LLM should evaluate the text based on the feature described below.
2.  Provide a numerical score on a scale of 1 to 10 (where 1 is low quality/absence of the feature, and 10 is high quality/strong presence of the feature).
3.  Respond with ONLY the numerical score, without any additional text or explanation.
4.  Clearly explain the criteria used to determine the score.

Here is the feature that the annotation prompt needs to measure:
---
{feature_description}
---

Now, generate the annotation prompt template text. Your ONLY task is to generate the raw text for a prompt template. Do not output anything else. Do not use markdown, do not add titles, do not add any explanations. 

Your output must begin directly with the text of the prompt. Your output should end with "The text to evaluate is: [TEXT_TO_EVALUATE]." The [TEXT_TO_EVALUATE] placeholder should only be used once at the end of the prompt.
"""

# --- Iterative Feature Reflection Prompt ---

REFLECT_AND_GENERATE_FEATURES_PROMPT = """
You are an expert in text quality assessment feature engineering. We are working on a text quality assessment task with the following scenario:

Scenario Description:
---
{scene_description}
---

We have completed a round of feature selection using a beam search algorithm. The current best features selected and their performance (Spearman correlation coefficients, sorted in descending order) are:
---
{features_with_scores}
---

Your task is to analyze the selected features and their performance in the context of the above scenario, then propose {new_feature_count} NEW, innovative features that could potentially improve the assessment quality for this specific task.

When designing new features, you should:
1. Consider the specific requirements and context of the scenario described above
2. Identify potential gaps or aspects not well-covered by the current feature set
3. Consider features that might capture different dimensions or perspectives relevant to this scenario
4. Think about features that could complement or enhance the existing ones
5. Ensure the features are concrete, measurable, and interpretable
6. Avoid proposing features that are too similar to existing ones

Output exactly {new_feature_count} new features. Each feature must be on a new line, and should be about 30 words.
Only output {new_feature_count} lines, with each line containing a distinct feature. For each feature, provide only its name and a clear text description of what it measures. Do not use any special symbols or formatting, including list numbers.

Example Format:
[Feature 1], [Feature description]
[Feature 2], [Feature description]
"""

