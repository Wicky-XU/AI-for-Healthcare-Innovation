"""
COVID-19 Lung CT Image Classification Project - Main Entry Point
================================================================

Main Program - Command Line Interface
Provides CLI and complete workflow management, supports all module functions

Usage:
    python src/main.py --mode train --epochs 10
    python src/main.py --mode predict --epochs 10
    python src/main.py --mode compare

"""

import argparse
from . import config
from . import data
from . import model
from . import utils


def train_workflow(epochs=10):
    """Complete training workflow"""
    print(f"\n{'=' * 70}")
    print(f"  COVID-19 Lung CT Classification - {epochs} Epochs Training")
    print(f"{'=' * 70}\n")

    # 1. Check data
    print("Step 1/5: Checking data...")
    data.check_and_setup()

    # 2. Split data
    print("\nStep 2/5: Splitting data...")
    train_dir, val_dir = data.split_data()
    utils.plot_class_distribution(train_dir, val_dir)

    # 3. Create data generators
    print("\nStep 3/5: Creating data generators...")
    version = 'enhanced' if epochs == 50 else 'simple'
    train_gen, val_gen = data.create_generators(train_dir, val_dir, augment=(epochs == 50))

    # 4. Build model
    print(f"\nStep 4/5: Building model...")
    covid_model = model.build_model(version=version)

    # 5. Train model
    print(f"\nStep 5/5: Training model...")
    history = model.train_model(covid_model, train_gen, val_gen, epochs, version)

    # 6. Visualize results
    print("\nGenerating training curves...")
    utils.plot_training_history(epochs)

    print(f"\n{'=' * 70}")
    print("Training workflow completed!")
    print(f"{'=' * 70}\n")


def predict_workflow(epochs=10):
    """Prediction workflow"""
    print(f"\n{'=' * 70}")
    print(f"  COVID-19 Lung CT Prediction - Using {epochs} Epochs Model")
    print(f"{'=' * 70}\n")

    # 1. Load test data
    print("Loading test images...")
    test_images, filenames = data.load_test_images()

    if len(test_images) == 0:
        print("No test images found, exiting prediction workflow")
        return

    # 2. Predict
    predictions = utils.predict_images(test_images, filenames, epochs)

    # 3. Visualize
    print("\nGenerating prediction visualization...")
    utils.plot_predictions(test_images, filenames, predictions)

    print(f"\n{'=' * 70}")
    print("Prediction workflow completed!")
    print(f"{'=' * 70}\n")


def compare_workflow():
    """Compare different model versions"""
    print(f"\n{'=' * 70}")
    print("  Model Comparison Analysis")
    print(f"{'=' * 70}\n")

    utils.compare_models()

    print("\nComparison completed!\n")


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='COVID-19 CT Image Classification')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict', 'compare'],
                        help='Running mode: train, predict, or compare')
    parser.add_argument('--epochs', type=int, default=10,
                        choices=[10, 50],
                        help='Training epochs: 10 or 50')

    args = parser.parse_args()

    if args.mode == 'train':
        train_workflow(args.epochs)
    elif args.mode == 'predict':
        predict_workflow(args.epochs)
    elif args.mode == 'compare':
        compare_workflow()


if __name__ == '__main__':
    main()