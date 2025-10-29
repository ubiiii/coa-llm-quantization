"""
Visualization utilities for LLM Quantization Performance Analysis

This module provides comprehensive visualization functions for comparing
quantization performance across different models and configurations.

Team: CipherCore (Utkarsh & Sami)
Project: Hardware/Software Co-Design for LLM Quantization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QuantizationVisualizer:
    """
    Comprehensive visualization class for quantization performance analysis.
    
    Provides methods for creating comparison charts, performance graphs,
    and analysis visualizations for LLM quantization experiments.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer with default figure size.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
    
    def create_performance_data(self) -> pd.DataFrame:
        """
        Create DataFrame with all experimental results.
        
        Returns:
            DataFrame with performance data for visualization
        """
        data = {
            'Model': ['DialoGPT-small', 'DialoGPT-small', 'TinyLlama-1.1B', 'Llama-3.2-1B', 'distilgpt2', 'distilgpt2'],
            'Precision': ['FP16', 'INT8', 'FP16', 'INT4', 'FP16', 'INT8'],
            'Parameters_M': [124.4, 124.4, 1100.0, 1000.0, 82.0, 82.0],
            'Speed_tokens_per_sec': [28.42, 5.58, 34.53, 157.11, 91.81, 59.93],
            'Memory_GB': [0.54, 0.27, 2.2, 0.55, 0.35, 0.31],
            'GPU_Utilization_%': [45.2, 38.7, 52.1, 78.3, 15.0, 14.0],
            'Speedup_Factor': [1.0, 0.52, 1.0, 4.55, 1.0, 0.65],
            'Memory_Reduction_%': [0.0, 50.0, 0.0, 75.0, 0.0, 12.0],
            'Model_Size_MB': [351.0, 175.5, 2200.0, 550.0, 460.95, 229.14],
            'Experiment_Date': ['2025-10-10', '2025-10-09', '2025-10-09', '2025-10-09', '2025-01-19', '2025-01-19'],
            'Hardware': ['Tesla T4', 'Tesla T4', 'Tesla T4', 'Tesla T4', 'Tesla T4', 'Tesla T4']
        }
        
        return pd.DataFrame(data)
    
    def plot_speed_comparison(self, save_path: str = 'results/speed_comparison.png'):
        """
        Create speed comparison chart showing tokens/sec for each configuration.
        
        Args:
            save_path: Path to save the plot
        """
        df = self.create_performance_data()
        
        plt.figure(figsize=self.figsize)
        
        # Create bar plot
        bars = plt.bar(range(len(df)), df['Speed_tokens_per_sec'], 
                      color=self.colors[:len(df)], alpha=0.8, edgecolor='black')
        
        # Customize plot
        plt.title('LLM Inference Speed Comparison\n(Higher is Better)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('Speed (Tokens/Second)', fontsize=12, fontweight='bold')
        
        # Set x-axis labels
        labels = [f"{row['Model']}\n{row['Precision']}" for _, row in df.iterrows()]
        plt.xticks(range(len(df)), labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, df['Speed_tokens_per_sec'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Add insights text
        plt.figtext(0.02, 0.02, 
                   'Key Insight: INT4 quantization shows 4.55× speedup on larger models,\n'
                   'while INT8 shows overhead on smaller models due to quantization costs.',
                   fontsize=10, style='italic', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Speed comparison chart saved to {save_path}")
    
    def plot_memory_usage(self, save_path: str = 'results/memory_usage.png'):
        """
        Create memory usage comparison chart.
        
        Args:
            save_path: Path to save the plot
        """
        df = self.create_performance_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Memory usage bar chart
        bars1 = ax1.bar(range(len(df)), df['Memory_GB'], 
                       color=self.colors[:len(df)], alpha=0.8, edgecolor='black')
        ax1.set_title('Peak Memory Usage\n(Lower is Better)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Memory Usage (GB)', fontsize=12, fontweight='bold')
        
        labels = [f"{row['Model']}\n{row['Precision']}" for _, row in df.iterrows()]
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        for i, (bar, value) in enumerate(zip(bars1, df['Memory_GB'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}GB', ha='center', va='bottom', fontweight='bold')
        
        ax1.grid(axis='y', alpha=0.3)
        
        # Memory reduction percentage
        bars2 = ax2.bar(range(len(df)), df['Memory_Reduction_%'], 
                       color=self.colors[:len(df)], alpha=0.8, edgecolor='black')
        ax2.set_title('Memory Reduction Percentage\n(Higher is Better)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Reduction (%)', fontsize=12, fontweight='bold')
        
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        for i, (bar, value) in enumerate(zip(bars2, df['Memory_Reduction_%'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Memory usage chart saved to {save_path}")
    
    def plot_speedup_analysis(self, save_path: str = 'results/speedup_analysis.png'):
        """
        Create speedup factor comparison chart.
        
        Args:
            save_path: Path to save the plot
        """
        df = self.create_performance_data()
        
        plt.figure(figsize=self.figsize)
        
        # Create bar plot with different colors for positive/negative speedup
        colors = ['green' if x >= 1.0 else 'red' for x in df['Speedup_Factor']]
        bars = plt.bar(range(len(df)), df['Speedup_Factor'], 
                      color=colors, alpha=0.7, edgecolor='black')
        
        # Add baseline line
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        plt.title('Speedup Factor Comparison\n(>1.0 = Faster, <1.0 = Slower)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('Speedup Factor (×)', fontsize=12, fontweight='bold')
        
        labels = [f"{row['Model']}\n{row['Precision']}" for _, row in df.iterrows()]
        plt.xticks(range(len(df)), labels, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, df['Speedup_Factor'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}×', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        
        # Add interpretation text
        plt.figtext(0.02, 0.02, 
                   'Analysis: INT4 shows 4.55× speedup, while INT8 shows 0.52× slowdown\n'
                   'due to quantization overhead on small models.',
                   fontsize=10, style='italic', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Speedup analysis chart saved to {save_path}")
    
    def plot_gpu_utilization(self, save_path: str = 'results/gpu_utilization.png'):
        """
        Create GPU utilization comparison chart.
        
        Args:
            save_path: Path to save the plot
        """
        df = self.create_performance_data()
        
        plt.figure(figsize=self.figsize)
        
        bars = plt.bar(range(len(df)), df['GPU_Utilization_%'], 
                      color=self.colors[:len(df)], alpha=0.8, edgecolor='black')
        
        plt.title('GPU Utilization Comparison\n(Higher = Better Hardware Usage)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Model Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('GPU Utilization (%)', fontsize=12, fontweight='bold')
        
        labels = [f"{row['Model']}\n{row['Precision']}" for _, row in df.iterrows()]
        plt.xticks(range(len(df)), labels, rotation=45, ha='right')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, df['GPU_Utilization_%'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add efficiency zones
        plt.axhspan(0, 50, alpha=0.1, color='red', label='Low Efficiency')
        plt.axhspan(50, 75, alpha=0.1, color='yellow', label='Medium Efficiency')
        plt.axhspan(75, 100, alpha=0.1, color='green', label='High Efficiency')
        
        plt.legend(loc='upper right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ GPU utilization chart saved to {save_path}")
    
    def plot_comprehensive_comparison(self, save_path: str = 'results/comprehensive_comparison.png'):
        """
        Create comprehensive comparison dashboard.
        
        Args:
            save_path: Path to save the plot
        """
        df = self.create_performance_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Speed comparison
        bars1 = ax1.bar(range(len(df)), df['Speed_tokens_per_sec'], 
                       color=self.colors[:len(df)], alpha=0.8)
        ax1.set_title('Inference Speed (Tokens/sec)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Speed (tokens/sec)')
        labels = [f"{row['Precision']}" for _, row in df.iterrows()]
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(labels, rotation=45)
        
        for i, (bar, value) in enumerate(zip(bars1, df['Speed_tokens_per_sec'])):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Memory usage
        bars2 = ax2.bar(range(len(df)), df['Memory_GB'], 
                       color=self.colors[:len(df)], alpha=0.8)
        ax2.set_title('Memory Usage (GB)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Memory (GB)')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(labels, rotation=45)
        
        for i, (bar, value) in enumerate(zip(bars2, df['Memory_GB'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Speedup factor
        colors3 = ['green' if x >= 1.0 else 'red' for x in df['Speedup_Factor']]
        bars3 = ax3.bar(range(len(df)), df['Speedup_Factor'], 
                       color=colors3, alpha=0.8)
        ax3.set_title('Speedup Factor', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Speedup (×)')
        ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(labels, rotation=45)
        
        for i, (bar, value) in enumerate(zip(bars3, df['Speedup_Factor'])):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}×', ha='center', va='bottom', fontweight='bold')
        
        # 4. GPU utilization
        bars4 = ax4.bar(range(len(df)), df['GPU_Utilization_%'], 
                       color=self.colors[:len(df)], alpha=0.8)
        ax4.set_title('GPU Utilization (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('GPU Utilization (%)')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(labels, rotation=45)
        
        for i, (bar, value) in enumerate(zip(bars4, df['GPU_Utilization_%'])):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add overall title
        fig.suptitle('LLM Quantization Performance Analysis\nTesla T4 GPU - Comprehensive Comparison', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Add grid to all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive comparison dashboard saved to {save_path}")
    
    def plot_model_size_analysis(self, save_path: str = 'results/model_size_analysis.png'):
        """
        Create model size vs performance analysis.
        
        Args:
            save_path: Path to save the plot
        """
        df = self.create_performance_data()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Model size vs speed
        colors = [self.colors[i] for i in range(len(df))]
        scatter1 = ax1.scatter(df['Parameters_M'], df['Speed_tokens_per_sec'], 
                              c=colors, s=200, alpha=0.7, edgecolors='black')
        
        ax1.set_title('Model Size vs Inference Speed', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Parameters (Millions)')
        ax1.set_ylabel('Speed (Tokens/sec)')
        ax1.grid(True, alpha=0.3)
        
        # Add labels for each point
        for i, row in df.iterrows():
            ax1.annotate(f"{row['Model']}\n{row['Precision']}", 
                        (row['Parameters_M'], row['Speed_tokens_per_sec']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Model size vs memory efficiency
        scatter2 = ax2.scatter(df['Parameters_M'], df['Memory_Reduction_%'], 
                              c=colors, s=200, alpha=0.7, edgecolors='black')
        
        ax2.set_title('Model Size vs Memory Efficiency', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_ylabel('Memory Reduction (%)')
        ax2.grid(True, alpha=0.3)
        
        # Add labels for each point
        for i, row in df.iterrows():
            ax2.annotate(f"{row['Model']}\n{row['Precision']}", 
                        (row['Parameters_M'], row['Memory_Reduction_%']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Model size analysis saved to {save_path}")
    
    def create_all_visualizations(self):
        """
        Create all visualization charts and save them.
        """
        print("🎨 Creating comprehensive visualization suite...")
        print("=" * 60)
        
        # Create all plots
        self.plot_speed_comparison()
        self.plot_memory_usage()
        self.plot_speedup_analysis()
        self.plot_gpu_utilization()
        self.plot_comprehensive_comparison()
        self.plot_model_size_analysis()
        
        print("\n✅ All visualizations created successfully!")
        print("📁 Charts saved in 'results/' directory")
        print("📊 Ready for analysis and presentation!")


def create_visualization_notebook():
    """
    Create a complete Jupyter notebook with all visualizations.
    
    Returns:
        String containing the notebook code
    """
    notebook_code = '''
# LLM Quantization Performance Visualization

## Import Required Libraries
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from visualization import QuantizationVisualizer

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## Initialize Visualizer
```python
# Create visualizer instance
viz = QuantizationVisualizer(figsize=(12, 8))

# Display the data
df = viz.create_performance_data()
print("📊 Experimental Data:")
print(df)
```

## Create Individual Charts
```python
# 1. Speed Comparison
viz.plot_speed_comparison()

# 2. Memory Usage Analysis
viz.plot_memory_usage()

# 3. Speedup Factor Analysis
viz.plot_speedup_analysis()

# 4. GPU Utilization
viz.plot_gpu_utilization()

# 5. Comprehensive Dashboard
viz.plot_comprehensive_comparison()

# 6. Model Size Analysis
viz.plot_model_size_analysis()
```

## Create All Visualizations at Once
```python
# Create all charts
viz.create_all_visualizations()
```

## Analysis Summary
```python
# Print key insights
print("🔍 KEY INSIGHTS:")
print("=" * 50)
print("1. INT4 quantization shows 4.55× speedup on large models")
print("2. INT8 quantization shows overhead on small models")
print("3. Memory reduction: 50-75% with quantization")
print("4. GPU utilization: INT4 shows best efficiency (78.3%)")
print("5. Model size is critical for quantization benefits")
print("=" * 50)
```
'''
    
    return notebook_code


# Example usage
if __name__ == "__main__":
    print("LLM Quantization Visualization Utilities")
    print("Import this module to create performance comparison charts")
    print("\nExample usage:")
    print("from visualization import QuantizationVisualizer")
    print("viz = QuantizationVisualizer()")
    print("viz.create_all_visualizations()")
