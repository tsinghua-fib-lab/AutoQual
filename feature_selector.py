# -*- coding: utf-8 -*-

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from tqdm import tqdm
import config

class FeatureSelector:
    """
    Selects the best feature set using a beam search algorithm with a choice of evaluation methods:
    1. 'mutual_information': Uses MI for the entire search.
    2. 'linear_regression': Uses Spearman's Rho from a Linear Regression model.
    3. 'xgboost': Uses Spearman's Rho from an XGBoost model.
    """
    def __init__(self, annotated_data_path: str):
        self.annotated_data_path = annotated_data_path
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names = self._prepare_data()
        self._set_evaluation_method()

    def _prepare_data(self) -> tuple:
        """
        Loads, splits, and normalizes the data.
        - Splits data into train/test sets and saves them if they don't exist.
        - Normalizes features using MinMaxScaler fitted only on the training data.
        - Automatically regenerates split if new features are detected.
        """
        full_df = pd.read_csv(self.annotated_data_path)
        feature_names = [col for col in full_df.columns if col not in ['text', 'score']]
        
        need_new_split = False
        
        if os.path.exists(config.TRAIN_DATA_FILE) and os.path.exists(config.TEST_DATA_FILE):
            print("Train/test split found, checking compatibility...")
            train_df = pd.read_csv(config.TRAIN_DATA_FILE)
            test_df = pd.read_csv(config.TEST_DATA_FILE)
            
            # Check if cached split has all required features
            train_features = set(train_df.columns)
            required_features = set(feature_names + ['text', 'score'])
            
            if not required_features.issubset(train_features):
                missing_features = required_features - train_features
                print(f"Cached split is missing {len(missing_features)} new features. Regenerating split...")
                need_new_split = True
            else:
                print("Cached split is compatible. Loading from cache.")
        else:
            print("No train/test split found. Creating and saving a new one.")
            need_new_split = True
        
        if need_new_split:
            train_df, test_df = train_test_split(
                full_df, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE
            )
            train_df.to_csv(config.TRAIN_DATA_FILE, index=False)
            test_df.to_csv(config.TEST_DATA_FILE, index=False)
            print(f"New train/test split saved with {len(feature_names)} features.")
        
        # Clean data
        train_df = train_df.dropna(subset=feature_names + ['score'])
        test_df = test_df.dropna(subset=feature_names + ['score'])

        X_train_raw = train_df[feature_names]
        y_train = train_df['score']
        X_test_raw = test_df[feature_names]
        y_test = test_df['score']

        # Normalize features
        feature_scaler = MinMaxScaler()
        X_train = pd.DataFrame(feature_scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index)
        X_test = pd.DataFrame(feature_scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)

        # Normalize score
        score_scaler = MinMaxScaler()
        y_train = pd.Series(score_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), name='score', index=y_train.index)
        y_test = pd.Series(score_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name='score', index=y_test.index)

        print(f"Data prepared: {len(X_train)} train samples, {len(X_test)} test samples. Features and scores are normalized.")
        return X_train, X_test, y_train, y_test, feature_names

    def _set_evaluation_method(self):
        """Sets the evaluation function based on the config file."""
        method = config.EVALUATION_METHOD
        if method == "mutual_information":
            self.evaluate_set = self._evaluate_mi
        elif method == "linear_regression":
            self.evaluate_set = self._evaluate_linear_regression
        elif method == "xgboost":
            self.evaluate_set = self._evaluate_xgboost
        else:
            raise ValueError(f"Unknown evaluation method: {method}")
        print(f"Using evaluation method: '{method}'")

    def select_features(self, return_scores: bool = False):
        """
        Executes the beam search to find the best feature set.
        
        Args:
            return_scores: If True, returns a tuple of (feature_list, feature_scores_dict)
                          If False, returns only feature_list for backward compatibility
        
        Returns:
            list or tuple: Best feature set, optionally with scores dict
        """
        print("\n--- Starting Feature Selection using Beam Search ---")
        
        # --- Stage 0: Initialize Beam ---
        # We now evaluate the initial candidates using the same metric as the search,
        # ensuring all scores are comparable.
        print("Evaluating single features to initialize the beam...")
        initial_candidates = []
        for feature in tqdm(self.feature_names, desc="Initial Evaluation"):
            score, _ = self.evaluate_set([feature])
            initial_candidates.append(([feature], score))
        
        initial_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # A beam is a list of tuples: (feature_list, score)
        beams = initial_candidates[:config.BEAM_WIDTH]
        print(f"Initial beam initialized with top {config.BEAM_WIDTH} features based on {config.EVALUATION_METHOD}.")
        for features, score in beams:
            print(f"  - Feature: {features[0]}, Score: {score:.4f}")

        # --- Iterative Search ---
        # Start from iteration 2 since we now have 1 feature in each beam
        for i in tqdm(range(2, config.MAX_FEATURES + 1), desc="Beam Search Iterations"):
            candidates = []
            for feature_set, _ in beams:
                remaining_features = [f for f in self.feature_names if f not in feature_set]
                for new_feature in remaining_features:
                    current_features = feature_set + [new_feature]
                    
                    # Evaluate this new set of features
                    score, _ = self.evaluate_set(current_features)
                    candidates.append((current_features, score))

            # Prune candidates: select top N to form the new beam
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:config.BEAM_WIDTH]
            
            # Print best beam of the current iteration
            if not beams:
                print("No more candidates found. Halting search.")
                break
            best_beam_features, best_beam_score = beams[0]
            print(f"Iteration {i}/{config.MAX_FEATURES} -> Best Score: {best_beam_score:.4f} with {len(best_beam_features)} features.")

        # --- Final Selection ---
        if not beams:
            print("Error: Beam search ended with no selected features.")
            if return_scores:
                return [], {}
            return []
            
        best_feature_set, final_score = beams[0]
        # Recalculate MAE using the final chosen model type for consistency
        _, final_mae = self.evaluate_set(best_feature_set)
        
        # Calculate individual feature scores for reflection
        feature_scores = {}
        if return_scores:
            print("\n--- Calculating Individual Feature Scores ---")
            for feature in best_feature_set:
                score, _ = self.evaluate_set([feature])
                feature_scores[feature] = score
        
        print("\n--- ✅ Feature Selection Complete ---")
        print(f"Best Feature Set Found ({len(best_feature_set)} features):")
        for feature in best_feature_set:
            print(f"  - {feature}")
        print(f"\nFinal Model Performance on Test Set (using '{config.EVALUATION_METHOD}'):")
        print(f"  - Final Score (Rho or MI): {final_score:.4f}")
        print(f"  - Mean Absolute Error (MAE): {final_mae:.4f}")
        
        # Save the best feature list
        with open(config.BEST_FEATURES_FILE, 'w', encoding='utf-8') as f:
            for feature in best_feature_set:
                f.write(f"{feature}\n")
        print(f"\nBest feature set saved to '{config.BEST_FEATURES_FILE}'")

        if return_scores:
            return best_feature_set, feature_scores
        return best_feature_set

    # --- Evaluation Method Implementations ---

    def _evaluate_mi(self, features: list):
        """Evaluates a feature set using mutual information."""
        if len(features) == 1:
            # For a single feature, MI is well-defined
            mi = mutual_info_regression(self.X_train[features], self.y_train, random_state=config.RANDOM_STATE)[0]
            return mi, 0 # MAE is not applicable here
        else:
            # For multiple features, we use a model as a proxy, as MI isn't directly comparable.
            # Defaulting to a simple model for multi-feature MI proxy.
            return self._evaluate_linear_regression(features)

    def _evaluate_linear_regression(self, features: list):
        """Evaluates using Spearman's Rho from a Linear Regression model."""
        model = LinearRegression()
        model.fit(self.X_train[features], self.y_train)
        predictions = model.predict(self.X_test[features])
        rho, _ = spearmanr(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        return rho, mae

    def _evaluate_xgboost(self, features: list):
        """Evaluates using Spearman's Rho from an XGBoost model."""
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        model.fit(self.X_train[features], self.y_train)
        predictions = model.predict(self.X_test[features])
        rho, _ = spearmanr(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        return rho, mae

