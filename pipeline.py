"""
pipeline.py
-----------
Full Production Pipeline: GAN-driven Oversampling, Recursive Feature Elimination (RFECV),
and Random Forest Optimization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

from models import build_generator, build_discriminator, assemble_gan

# Configuration parameters
LATENT_DIM = 100
TARGET_COUNT_PER_CLASS = 100
EPOCHS = 3000
BATCH_SIZE = 32

def train_gan(gan, generator, discriminator, data, epochs=3000, batch_size=32):
    """Executes GAN training loop over a specific minority subset matrix."""
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Prepare real vs synthetic samples
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        generated_data = generator.predict(noise, verbose=0)

        # Train Discriminator
        discriminator.train_on_batch(real_data, valid)
        discriminator.train_on_batch(generated_data, fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        gan.train_on_batch(noise, valid)

def balance_classes_with_gan(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Identifies class imbalances and populates minority classes using class-specific GANs."""
    print("[*] Assessing class distributions...")
    unique_classes = df[target_col].unique()
    
    for cls in unique_classes:
        class_subset = df[df[target_col] == cls]
        current_count = len(class_subset)
        
        if current_count < TARGET_COUNT_PER_CLASS:
            samples_needed = TARGET_COUNT_PER_CLASS - current_count
            print(f"[!] Class '{cls}' imbalanced ({current_count}/{TARGET_COUNT_PER_CLASS}). Training GAN...")
            
            # Extract independent variables
            numerical_data = class_subset.drop(target_col, axis=1)
            features_shape = numerical_data.shape[1]
            
            # Re-initialize fresh GAN components for this class
            generator = build_generator(LATENT_DIM, features_shape)
            discriminator = build_discriminator(features_shape)
            gan = assemble_gan(generator, discriminator, LATENT_DIM)
            
            # Train the network on this specific target feature distribution
            train_gan(gan, generator, discriminator, numerical_data.values, epochs=EPOCHS, batch_size=BATCH_SIZE)
            
            # Generate synthetic buffer
            noise = np.random.normal(0, 1, (samples_needed, LATENT_DIM))
            synthetic_data = generator.predict(noise, verbose=0)
            
            # Structure into analytical DataFrames
            synthetic_df = pd.DataFrame(synthetic_data, columns=numerical_data.columns)
            synthetic_df[target_col] = cls
            
            # Merge back seamlessly
            df = pd.concat([df, synthetic_df], ignore_index=True)
            
    print("[+] All classes successfully balanced via GAN augmentation.")
    return df

def run_pipeline(train_path: str, test_path: str, target_col: str):
    """Executes full loading, balancing, feature elimination, and training pipeline."""
    # 1. Ingest Data
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Verify your dataset paths in your local workspace data directory.")
        
    df = pd.read_csv(train_path).drop(['UID', 'Spotted_knapweed'], axis=1, errors='ignore')
    test_df = pd.read_csv(test_path).drop(['UID', 'Spotted_knapweed'], axis=1, errors='ignore')

    # 2. Augment minority classes
    df_balanced = balance_classes_with_gan(df, target_col)
    
    # 3. Create Splitting Invariants
    X_train = df_balanced.drop(target_col, axis=1)
    y_train = df_balanced[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]

    # 4. Feature Optimization (RFECV Pipeline)
    print("[*] Running Recursive Feature Elimination with Cross-Validation (RFECV)...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFECV(estimator=rf_selector, cv=StratifiedKFold(n_splits=10), step=1, scoring='accuracy')
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.support_].tolist()
    print(f"[+] Optimal features selected: {len(selected_features)} out of {X_train.shape[1]}")

    # Apply feature masking
    X_train_opt = X_train[selected_features]
    X_test_opt = X_test[selected_features]

    # 5. Cross-Validation Engine Check
    rf_final = RandomForestClassifier(n_estimators=500, random_state=42)
    cv_scores = cross_val_score(rf_final, X_train_opt, y_train, cv=10)
    print(f"[+] Robust Cross-Validation Training Accuracy Mean: {np.mean(cv_scores):.4f}")

    # 6. Evaluate Model on Test Sets
    rf_final.fit(X_train_opt, y_train)
    predictions = rf_final.predict(X_test_opt)
    
    cm = confusion_matrix(y_test, predictions)
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print(f"\n[🚀 FINAL EVALUATION] Test Set Overall Accuracy: {overall_accuracy:.4f}")

    # 7. Generate Pipeline Performance Artifacts
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='crimson')
    plt.title('Stratified K-Fold Cross-Validation Matrix Score Evaluation', weight='bold')
    plt.xlabel('Validation Fold Sequence')
    plt.ylabel('Evaluated Accuracy Scale')
    plt.grid(True, linestyle='--')
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.tight_layout()
    
    # Safely save evaluation results to a data artifact folder
    os.makedirs("data", exist_ok=True)
    plt.savefig(os.path.join("data", "cv_performance_metrics.png"))
    print("[*] Performance validation trend metrics saved to 'data/cv_performance_metrics.png'")
    plt.show()

if __name__ == "__main__":
    # Point these paths directly to wherever you store your files locally
    TRAIN_DATA = os.path.join("data", "Data_base.csv")
    TEST_DATA = os.path.join("data", "testdata.csv")
    TARGET_VARIABLE = "Code_sk"
    
    # Fallback to a mock simulation if running pipeline locally without files present 
    if not os.path.exists(TRAIN_DATA):
        os.makedirs("data", exist_ok=True)
        print("[-] Target data path not found. Initializing a mock local simulation...")
        mock_df = pd.DataFrame(np.random.rand(100, 82))
        # Label column
        mock_df.iloc[:, 0] = np.random.choice([1, 2, 3], 100)
        mock_df.rename(columns={0: TARGET_VARIABLE}, inplace=True)
        mock_df.to_csv(TRAIN_DATA, index=False)
        
        mock_test = pd.DataFrame(np.random.rand(20, 82))
        mock_test.iloc[:, 0] = np.random.choice([1, 2, 3], 20)
        mock_test.rename(columns={0: TARGET_VARIABLE}, inplace=True)
        mock_test.to_csv(TEST_DATA, index=False)
        
    # Run the fully automated suite
    run_pipeline(TRAIN_DATA, TEST_DATA, TARGET_VARIABLE)