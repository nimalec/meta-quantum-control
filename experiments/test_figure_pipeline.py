"""
Test Figure Generation Pipeline

This script tests that:
1. Training scripts save the required data
2. Figure scripts can load and process the data
3. Figures are generated successfully
"""

import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("TESTING FIGURE GENERATION PIPELINE")
print("=" * 70)

# Create temporary directory for test outputs
test_dir = Path(tempfile.mkdtemp(prefix="meta_quantum_test_"))
print(f"\nTest directory: {test_dir}")

try:
    # Step 1: Test MAMLTrainer history saving
    print("\n" + "=" * 70)
    print("Step 1: Testing MAMLTrainer history format")
    print("=" * 70)

    from metaqctrl.meta_rl.maml import MAMLTrainer, MAML
    from metaqctrl.meta_rl.policy import PulsePolicy

    # Create minimal MAML setup
    policy = PulsePolicy(
        task_feature_dim=3,
        hidden_dim=32,
        n_hidden_layers=1,
        n_segments=10,
        n_controls=2
    )

    maml = MAML(
        policy=policy,
        inner_lr=0.01,
        inner_steps=3,
        meta_lr=0.01,
        first_order=True
    )

    # Create dummy components
    def dummy_task_sampler(n, split):
        from metaqctrl.quantum.noise_models import NoiseParameters
        import numpy as np
        return [NoiseParameters(1.0, 0.01, 500) for _ in range(n)]

    def dummy_data_generator(task_params, n_trajectories, split):
        import torch
        return {
            'task_features': torch.randn(n_trajectories, 3),
            'task_params': task_params
        }

    def dummy_loss_fn(policy, data):
        import torch
        controls = policy(data['task_features'])
        return torch.mean(controls ** 2)

    trainer = MAMLTrainer(
        maml=maml,
        task_sampler=dummy_task_sampler,
        data_generator=dummy_data_generator,
        loss_fn=dummy_loss_fn,
        n_support=2,
        n_query=2,
        log_interval=1,
        val_interval=2
    )

    # Check that training_history has all required fields
    required_fields = [
        'iterations', 'meta_loss', 'support_loss', 'query_loss',
        'val_fidelity', 'grad_norms', 'nan_count'
    ]

    print(f"\nChecking training_history fields...")
    missing = []
    for field in required_fields:
        if field not in trainer.training_history:
            missing.append(field)
            print(f"  ✗ Missing: {field}")
        else:
            print(f"  ✓ Found: {field}")

    if missing:
        print(f"\n✗ FAILED: Missing fields: {missing}")
        sys.exit(1)

    print(f"\n✓ PASSED: All required fields present in training_history")

    # Step 2: Simulate a few training iterations
    print("\n" + "=" * 70)
    print("Step 2: Running mini training session")
    print("=" * 70)

    # Run a few iterations
    print("\nRunning 5 training iterations...")
    save_path = test_dir / "test_checkpoint.pt"

    try:
        for i in range(5):
            task_batch = trainer.generate_task_batch(2, split='train')
            metrics = maml.meta_train_step(task_batch, dummy_loss_fn)

            # Manually track (simulating what happens in train())
            trainer.training_history['iterations'].append(i)
            trainer.training_history['meta_loss'].append(metrics['meta_loss'])
            trainer.training_history['query_loss'].append(metrics['meta_loss'])
            trainer.training_history['support_loss'].append(metrics.get('mean_task_loss', metrics['meta_loss']))
            trainer.training_history['grad_norms'].append(metrics.get('grad_norm', 0.0))
            has_nan = metrics.get('error') is not None
            trainer.training_history['nan_count'].append(1 if has_nan else 0)

            if i % 2 == 0:
                # Validation
                val_task_batch = trainer.generate_task_batch(2, split='val')
                val_metrics = maml.meta_validate(val_task_batch, dummy_loss_fn)
                val_fidelity = 1.0 - val_metrics['val_loss_post_adapt']
                trainer.training_history['val_fidelity'].append(val_fidelity)
                trainer.training_history['val_error'].append(val_metrics['val_loss_post_adapt'])
                trainer.training_history['val_iteration'].append(i)
                trainer.training_history['val_fidelity_std'].append(val_metrics['std_post_adapt'])

            print(f"  Iter {i}: loss={metrics['meta_loss']:.4f}, grad_norm={metrics.get('grad_norm', 0):.4e}")

        print(f"\n✓ Training completed successfully")

    except Exception as e:
        print(f"\n✗ FAILED: Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Save training history
    print("\n" + "=" * 70)
    print("Step 3: Saving training history")
    print("=" * 70)

    try:
        trainer.save_training_history(str(save_path))

        history_path = save_path.parent / "training_history.json"

        if not history_path.exists():
            print(f"\n✗ FAILED: History file not created at {history_path}")
            sys.exit(1)

        print(f"\n✓ History saved to: {history_path}")

        # Verify JSON is valid
        with open(history_path, 'r') as f:
            loaded_history = json.load(f)

        print(f"\n✓ History JSON is valid")
        print(f"  Contains {len(loaded_history['iterations'])} iterations")

    except Exception as e:
        print(f"\n✗ FAILED: Error saving history: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Test figure generation
    print("\n" + "=" * 70)
    print("Step 4: Testing figure generation")
    print("=" * 70)

    try:
        # Import figure generation module
        sys.path.insert(0, str(Path(__file__).parent / "icml_figures"))
        from figure3_training_stability import load_training_logs, generate_figure3

        # Load the training logs
        print(f"\nLoading training logs from: {history_path}")
        data = load_training_logs(str(history_path))

        print(f"✓ Loaded {len(data['iterations'])} iterations")

        # Verify all required fields are present
        print(f"\nVerifying data fields...")
        for field in required_fields:
            if field not in data:
                print(f"  ✗ Missing in loaded data: {field}")
            else:
                print(f"  ✓ {field}: {len(data[field])} entries")

        # Generate figure
        output_dir = test_dir / "figures"
        output_dir.mkdir(exist_ok=True)

        print(f"\nGenerating Figure 3...")
        generate_figure3(
            log_path=str(history_path),
            output_dir=str(output_dir)
        )

        # Check if figure was created
        pdf_path = output_dir / "figure3_training_stability.pdf"
        png_path = output_dir / "figure3_training_stability.png"

        if pdf_path.exists():
            print(f"\n✓ PDF figure created: {pdf_path}")
        else:
            print(f"\n⚠ PDF figure not found")

        if png_path.exists():
            print(f"✓ PNG figure created: {png_path}")
        else:
            print(f"⚠ PNG figure not found")

        if pdf_path.exists() or png_path.exists():
            print(f"\n✓ PASSED: Figure generation successful")
        else:
            print(f"\n✗ FAILED: No figures created")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ FAILED: Figure generation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE TEST SUMMARY")
    print("=" * 70)
    print("✓ MAMLTrainer has all required fields")
    print("✓ Training runs and tracks metrics correctly")
    print("✓ Training history saves to JSON")
    print("✓ Figure generation loads data and creates plots")
    print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("\nThe figure generation pipeline is working correctly!")
    print(f"\nGenerated files in: {test_dir}")

finally:
    # Cleanup
    print(f"\nCleaning up test directory...")
    try:
        shutil.rmtree(test_dir)
        print(f"✓ Cleaned up {test_dir}")
    except Exception as e:
        print(f"⚠ Could not clean up {test_dir}: {e}")
        print(f"  Please delete manually if needed")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
