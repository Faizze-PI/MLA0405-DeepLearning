"""
Deep Learning Lab - Final Summary Report Generator
Runs all 32 experiments and generates a comprehensive accuracy report.
Usage: python final_summary.py
"""

import subprocess
import sys
import os
import time

EXPERIMENTS = [
    ('Exp01', 'Exp01_Confusion_Matrix.py', 'Confusion Matrix Demo', 'Part A'),
    ('Exp02', 'Exp02_Two_Class_Analysis.py', 'Two-Class Pima Diabetes', 'Part A'),
    ('Exp03', 'Exp03_Multi_Class_Analysis.py', 'Multi-Class Iris', 'Part A'),
    ('Exp04', 'Exp04_Overfitting_Polynomial.py', 'Overfitting Polynomial', 'Part B'),
    ('Exp05', 'Exp05_Linear_Regression.py', 'Linear Regression', 'Part B'),
    ('Exp06', 'Exp06_KNN_Wine.py', 'KNN Wine', 'Part B'),
    ('Exp07', 'Exp07_Logistic_Regression_Sigmoid.py', 'LR Sigmoid', 'Part B'),
    ('Exp08', 'Exp08_KNN_Iris.py', 'KNN Iris', 'Part B'),
    ('Exp09', 'Exp09_Naive_Bayes_Iris.py', 'Naive Bayes Iris', 'Part B'),
    ('Exp10', 'Exp10_Logistic_Regression_Iris.py', 'LR Iris', 'Part B'),
    ('Exp11', 'Exp11_Decision_Tree_Iris.py', 'Decision Tree Iris', 'Part B'),
    ('Exp12', 'Exp12_Random_Forest_Iris.py', 'Random Forest Iris', 'Part B'),
    ('Exp13', 'Exp13_SVM_Iris.py', 'SVM Iris', 'Part B'),
    ('Exp14', 'Exp14_Gradient_Descent.py', 'Gradient Descent', 'Part B'),
    ('Exp15', 'Exp15_Image_Segmentation_KMeans.py', 'K-Means Segmentation', 'Part C'),
    ('Exp16', 'Exp16_Image_Segmentation_Thresholding.py', 'Thresholding', 'Part C'),
    ('Exp17', 'Exp17_Linear_Separability.py', 'Linear Separability', 'Part D'),
    ('Exp18', 'Exp18_NN_TwoClass_Linear.py', 'NN Linear 2-class', 'Part E'),
    ('Exp19', 'Exp19_NN_Circular_Linear.py', 'NN Linear circles', 'Part E'),
    ('Exp20', 'Exp20_NN_MultiClass_Linear.py', 'NN Linear 3-class', 'Part E'),
    ('Exp21', 'Exp21_NN_Circular_ReLU.py', 'NN ReLU circles', 'Part E'),
    ('Exp22', 'Exp22_NN_TwoClass_ReLU.py', 'NN ReLU 2-class', 'Part E'),
    ('Exp23', 'Exp23_NN_Spiral_Sigmoid.py', 'NN Sigmoid spiral', 'Part E'),
    ('Exp24', 'Exp24_NN_MultiClass_Sigmoid.py', 'NN Sigmoid 3-class', 'Part E'),
    ('Exp25', 'Exp25_NN_Circular_Tanh.py', 'NN Tanh circles', 'Part E'),
    ('Exp26', 'Exp26_NN_MultiClass_Tanh.py', 'NN Tanh 3-class', 'Part E'),
    ('Exp27', 'Exp27_NN_TwoClass_ReLU_LR001.py', 'NN ReLU LR=0.001', 'Part E'),
    ('Exp28', 'Exp28_NN_MultiClass_ReLU_LR001.py', 'NN Multi LR=0.001', 'Part E'),
    ('Exp29', 'Exp29_NN_MultiClass_Tanh_3Neurons.py', 'NN 3 neurons', 'Part E'),
    ('Exp30', 'Exp30_NN_TwoClass_Tanh_3Neurons.py', 'NN 2-class 3 neurons', 'Part E'),
    ('Exp31', 'Exp31_NN_MultiClass_ReLU_LR03.py', 'NN LR=0.03', 'Part E'),
    ('Exp32', 'Exp32_NN_Circular_Tanh_TwoFactors.py', 'NN circles factor=0.7', 'Part E'),
]

INTENTIONAL_LOW = {'Exp19', 'Exp23', 'Exp25', 'Exp32'}

def extract_metric(output):
    """Extract test accuracy from experiment output."""
    last_acc = None
    for line in output.split('\n'):
        if 'Test Accuracy:' in line:
            try:
                val = float(line.split('Test Accuracy:')[-1].strip().split()[0])
                last_acc = val
            except ValueError:
                pass
        if 'Best K:' in line and 'accuracy:' in line:
            try:
                last_acc = float(line.split('accuracy:')[-1].strip())
            except ValueError:
                pass
        if 'R-squared Score:' in line:
            try:
                last_acc = float(line.split('R-squared Score:')[-1].strip())
            except ValueError:
                pass
        if 'Best Test MSE:' in line:
            try:
                last_acc = -float(line.split('Best Test MSE:')[-1].strip())
            except ValueError:
                pass
        if 'Best Final Cost:' in line:
            try:
                last_acc = -float(line.split('Best Final Cost:')[-1].strip())
            except ValueError:
                pass
        # Match "Accuracy: 0.XXXX" (standalone line, not "Train Accuracy" or "Training Accuracy")
        if line.strip().startswith('Accuracy:') and 'Train' not in line and 'Training' not in line:
            try:
                last_acc = float(line.strip().split('Accuracy:')[-1].strip().split()[0])
            except ValueError:
                pass
        # Match "Random Forest (100 trees) Test Accuracy: 0.9000"
        if 'Test Accuracy:' in line and '(' in line:
            try:
                val = float(line.split('Test Accuracy:')[-1].strip().split()[0])
                last_acc = val
            except ValueError:
                pass
    return last_acc

def run_all():
    """Run all experiments and collect results."""
    results = []
    total = len(EXPERIMENTS)

    print('=' * 72)
    print('DEEP LEARNING LAB - FINAL SUMMARY REPORT')
    print('=' * 72)
    print()

    for idx, (name, filename, description, part) in enumerate(EXPERIMENTS, 1):
        print('[{}/{}] Running {} - {}...'.format(idx, total, name, description), end=' ')

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, filename],
                capture_output=True, text=True, timeout=120
            )
            elapsed = time.time() - start
            rc = proc.returncode
            metric = extract_metric(proc.stdout)
            status = 'OK' if rc == 0 else 'FAIL'
        except subprocess.TimeoutExpired:
            elapsed = 120
            rc = -1
            metric = None
            status = 'TIMEOUT'
        except Exception as e:
            elapsed = 0
            rc = -2
            metric = None
            status = 'ERROR'

        results.append((name, description, part, metric, status, elapsed))
        print('{} ({:.1f}s)'.format(status, elapsed))

    # Print summary table
    print()
    print('=' * 72)
    print('ACCURACY SUMMARY')
    print('=' * 72)
    print()
    print('{:<8} {:<30} {:<8} {:<12} {:<8}'.format(
        'Exp', 'Description', 'Part', 'Metric', 'Status'))
    print('-' * 72)

    pass_count = 0
    fail_count = 0
    intentional_count = 0
    na_count = 0

    for name, desc, part, metric, status, elapsed in results:
        if metric is not None:
            pct = metric * 100
            if name in INTENTIONAL_LOW:
                label = '{:.2f}%*'.format(pct)
                intentional_count += 1
            elif pct >= 80:
                label = '{:.2f}%'.format(pct)
                pass_count += 1
            else:
                label = '{:.2f}%'.format(pct)
                fail_count += 1
        else:
            label = 'N/A'
            na_count += 1

        print('{:<8} {:<30} {:<8} {:<12} {:<8}'.format(
            name, desc, part, label, status))

    print('-' * 72)
    print()
    print('SUMMARY:')
    print('  Total experiments: {}'.format(total))
    print('  Passed (>=80%):    {}'.format(pass_count))
    print('  Intentionally low: {}'.format(intentional_count))
    print('  Below 80%:         {}'.format(fail_count))
    print('  No metric:         {}'.format(na_count))
    print()
    print('  * Intentionally low experiments demonstrate pedagogical failures')
    print('    (linear activation on circles, sigmoid on spiral, etc.)')
    print()
    print('=' * 72)

if __name__ == '__main__':
    run_all()
