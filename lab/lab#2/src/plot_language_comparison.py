import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Remove Chinese font settings
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# Create output directory
os.makedirs('fig', exist_ok=True)

# Set chart style
sns.set(style="whitegrid")
plt.style.use('ggplot')

def plot_matrix_vector_comparison():
    # Read data
    df = pd.read_csv('results/language_comparison/matrix_vector_comparison.csv')
    
    # Create figure 1: Absolute time comparison (log scale)
    plt.figure(figsize=(12, 8))
    
    # Convert data structure for Seaborn plotting
    data = []
    for idx, row in df.iterrows():
        size = row['size']
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['cpp_col']), 'Algorithm': 'C++ Column Access'})
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['cpp_row']), 'Algorithm': 'C++ Row Access'})
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['cpp_unroll10']), 'Algorithm': 'C++ Unroll10'})
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['py_col']), 'Algorithm': 'Python Column Access'})
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['py_row']), 'Algorithm': 'Python Row Access'})
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['py_unroll10']), 'Algorithm': 'Python Unroll10'})
        data.append({'Matrix Size': size, 'Execution Time(ms)': float(row['py_numpy']), 'Algorithm': 'NumPy Matrix Mult'})
    
    df_plot = pd.DataFrame(data)
    
    # Draw bar chart - grouped by matrix size
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Matrix Size', y='Execution Time(ms)', hue='Algorithm', data=df_plot)
    plt.title('C++ vs Python Matrix-Vector Multiplication Performance', fontsize=16)
    plt.ylabel('Execution Time (ms) - Log Scale', fontsize=14)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.yscale('log')
    plt.legend(title='Algorithm', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig/language_matrix_vector_comparison.png', dpi=300)
    
    # Create figure 2: Python to C++ performance ratio
    plt.figure(figsize=(10, 8))
    
    # Calculate Python vs C++ performance ratio
    ratios = []
    for idx, row in df.iterrows():
        size = row['size']
        ratios.append({'Matrix Size': size, 'Performance Ratio': float(row['py_col']) / float(row['cpp_col']), 'Comparison': 'Python/C++ Column Access'})
        ratios.append({'Matrix Size': size, 'Performance Ratio': float(row['py_row']) / float(row['cpp_row']), 'Comparison': 'Python/C++ Row Access'})
        ratios.append({'Matrix Size': size, 'Performance Ratio': float(row['py_unroll10']) / float(row['cpp_unroll10']), 'Comparison': 'Python/C++ Unroll10'})
        ratios.append({'Matrix Size': size, 'Performance Ratio': float(row['py_numpy']) / float(row['cpp_col']), 'Comparison': 'NumPy/C++ Column Access'})
        ratios.append({'Matrix Size': size, 'Performance Ratio': float(row['py_numpy']) / float(row['cpp_row']), 'Comparison': 'NumPy/C++ Row Access'})
    
    df_ratios = pd.DataFrame(ratios)
    
    sns.barplot(x='Matrix Size', y='Performance Ratio', hue='Comparison', data=df_ratios)
    plt.title('Python to C++ Performance Ratio (Matrix-Vector Multiplication)', fontsize=16)
    plt.ylabel('Execution Time Ratio (Python/C++)', fontsize=14)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.legend(title='Comparison', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig/language_matrix_vector_ratio.png', dpi=300)

def plot_sum_array_comparison():
    # Read data
    df = pd.read_csv('results/language_comparison/sum_array_comparison.csv')
    
    # Convert array size to more readable format
    df['size_readable'] = df['size'].apply(lambda x: f'{x//1024}K' if x < 1024*1024 else f'{x//(1024*1024)}M')
    
    # Create figure 1: Absolute time comparison (log scale)
    plt.figure(figsize=(12, 8))
    
    # Convert data structure for Seaborn plotting
    data = []
    for idx, row in df.iterrows():
        size = row['size_readable']
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['cpp_naive']), 'Algorithm': 'C++ Naive'})
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['cpp_dual']), 'Algorithm': 'C++ Dual Path'})
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['cpp_recursive']), 'Algorithm': 'C++ Recursive'})
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['py_naive']), 'Algorithm': 'Python Naive'})
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['py_dual']), 'Algorithm': 'Python Dual Path'})
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['py_recursive']), 'Algorithm': 'Python Recursive'})
        data.append({'Array Size': size, 'Execution Time(ms)': float(row['py_numpy']), 'Algorithm': 'NumPy Sum'})
    
    df_plot = pd.DataFrame(data)
    
    # Draw bar chart - grouped by array size
    plt.figure(figsize=(15, 10))
    sns.barplot(x='Array Size', y='Execution Time(ms)', hue='Algorithm', data=df_plot)
    plt.title('C++ vs Python Array Sum Performance', fontsize=16)
    plt.ylabel('Execution Time (ms) - Log Scale', fontsize=14)
    plt.xlabel('Array Size', fontsize=14)
    plt.yscale('log')
    plt.legend(title='Algorithm', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig/language_sum_array_comparison.png', dpi=300)
    
    # Create figure 2: Python to C++ performance ratio
    plt.figure(figsize=(10, 8))
    
    # Calculate Python vs C++ performance ratio
    ratios = []
    for idx, row in df.iterrows():
        size = row['size_readable']
        ratios.append({'Array Size': size, 'Performance Ratio': float(row['py_naive']) / float(row['cpp_naive']), 'Comparison': 'Python/C++ Naive'})
        ratios.append({'Array Size': size, 'Performance Ratio': float(row['py_dual']) / float(row['cpp_dual']), 'Comparison': 'Python/C++ Dual Path'})
        ratios.append({'Array Size': size, 'Performance Ratio': float(row['py_recursive']) / float(row['cpp_recursive']), 'Comparison': 'Python/C++ Recursive'})
        ratios.append({'Array Size': size, 'Performance Ratio': float(row['py_numpy']) / float(row['cpp_naive']), 'Comparison': 'NumPy/C++ Naive'})
    
    df_ratios = pd.DataFrame(ratios)
    
    sns.barplot(x='Array Size', y='Performance Ratio', hue='Comparison', data=df_ratios)
    plt.title('Python to C++ Performance Ratio (Array Sum)', fontsize=16)
    plt.ylabel('Execution Time Ratio (Python/C++)', fontsize=14)
    plt.xlabel('Array Size', fontsize=14)
    plt.axhline(y=1, color='r', linestyle='--')
    plt.legend(title='Comparison', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('fig/language_sum_array_ratio.png', dpi=300)

def plot_language_radar_chart():
    # Radar chart data
    languages = ['C++', 'Python', 'Python+NumPy']
    
    # Scores for each metric (out of 10)
    # Metrics: [Compile/Interpret characteristics, Memory management efficiency, JIT optimization, Vectorization ability, Cache locality optimization, Branch prediction efficiency]
    values = np.array([
        [9, 9, 6, 7, 9, 8.5],  # C++
        [4, 5, 7, 3, 4, 4],    # Python
        [6, 7, 9, 9, 8, 7]      # Python+NumPy
    ])
    
    # Feature labels
    attributes = ['Static Compilation Advantage', 'Memory Management Efficiency', 'JIT Optimization', 
                  'Vectorization Support', 'Cache Friendliness', 'Branch Prediction Friendliness']
    
    # Draw radar chart
    angles = np.linspace(0, 2*np.pi, len(attributes), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add labels for each angle
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], attributes, fontsize=14)
    
    # Set y-axis label range
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ['2', '4', '6', '8', '10'], fontsize=12)
    plt.ylim(0, 10)
    
    # Draw data for each language
    for i, language in enumerate(languages):
        values_language = values[i].tolist()
        values_language += values_language[:1]  # Close the polygon
        ax.plot(angles, values_language, linewidth=2, linestyle='solid', label=language)
        ax.fill(angles, values_language, alpha=0.25)
    
    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    plt.title('Comparison of Programming Languages on Performance-Related Attributes', fontsize=16, y=1.1)
    
    plt.tight_layout()
    plt.savefig('fig/language_radar_comparison.png', dpi=300)

def main():
    # Draw all comparison charts
    plot_matrix_vector_comparison()
    plot_sum_array_comparison()
    plot_language_radar_chart()
    print("C++ and Python performance comparison charts have been generated.")

if __name__ == "__main__":
    main() 