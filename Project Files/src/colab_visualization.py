# LLM Quantization Performance Visualization
# Run this entire cell in Google Colab to create all charts

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

print("🎨 Creating LLM Quantization Performance Visualizations")
print("=" * 60)

# Create performance data
data = {
    'Model': ['DialoGPT-small', 'DialoGPT-small', 'TinyLlama-1.1B', 'Llama-3.2-1B'],
    'Precision': ['FP16', 'INT8', 'FP16', 'INT4'],
    'Parameters_M': [124.4, 124.4, 1100.0, 1000.0],
    'Speed_tokens_per_sec': [28.42, 5.58, 34.53, 157.11],
    'Memory_GB': [0.54, 0.27, 2.2, 0.55],
    'GPU_Utilization_%': [45.2, 38.7, 52.1, 78.3],
    'Speedup_Factor': [1.0, 0.52, 1.0, 4.55],
    'Memory_Reduction_%': [0.0, 50.0, 0.0, 75.0]
}

df = pd.DataFrame(data)
print("📊 Experimental Data:")
print(df)

# Create comprehensive comparison dashboard
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 1. Speed comparison
bars1 = ax1.bar(range(len(df)), df['Speed_tokens_per_sec'], color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Inference Speed Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Speed (tokens/sec)')
labels = [f"{row['Model']}\n{row['Precision']}" for _, row in df.iterrows()]
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(labels, rotation=45, ha='right')

for i, (bar, value) in enumerate(zip(bars1, df['Speed_tokens_per_sec'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 2. Memory usage
bars2 = ax2.bar(range(len(df)), df['Memory_GB'], color=colors, alpha=0.8, edgecolor='black')
ax2.set_title('Memory Usage Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Memory (GB)')
ax2.set_xticks(range(len(df)))
ax2.set_xticklabels(labels, rotation=45, ha='right')

for i, (bar, value) in enumerate(zip(bars2, df['Memory_GB'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{value:.2f}GB', ha='center', va='bottom', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Speedup factor
colors3 = ['green' if x >= 1.0 else 'red' for x in df['Speedup_Factor']]
bars3 = ax3.bar(range(len(df)), df['Speedup_Factor'], color=colors3, alpha=0.7, edgecolor='black')
ax3.set_title('Speedup Factor Analysis\n(>1.0 = Faster, <1.0 = Slower)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Speedup Factor (×)')
ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax3.set_xticks(range(len(df)))
ax3.set_xticklabels(labels, rotation=45, ha='right')

for i, (bar, value) in enumerate(zip(bars3, df['Speedup_Factor'])):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{value:.2f}×', ha='center', va='bottom', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. GPU utilization
bars4 = ax4.bar(range(len(df)), df['GPU_Utilization_%'], color=colors, alpha=0.8, edgecolor='black')
ax4.set_title('GPU Utilization Comparison\n(Higher = Better Hardware Usage)', fontsize=14, fontweight='bold')
ax4.set_ylabel('GPU Utilization (%)')
ax4.set_xticks(range(len(df)))
ax4.set_xticklabels(labels, rotation=45, ha='right')

for i, (bar, value) in enumerate(zip(bars4, df['GPU_Utilization_%'])):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# Add overall title
fig.suptitle('LLM Quantization Performance Analysis\nTesla T4 GPU - Hardware/Software Co-Design Results', 
            fontsize=18, fontweight='bold', y=0.95)

plt.tight_layout()
plt.show()

# Create additional charts
print("\n📈 Creating additional analysis charts...")

# Memory reduction analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Memory reduction percentage
bars1 = ax1.bar(range(len(df)), df['Memory_Reduction_%'], color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Memory Reduction Percentage\n(Higher is Better)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Memory Reduction (%)')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels(labels, rotation=45, ha='right')

for i, (bar, value) in enumerate(zip(bars1, df['Memory_Reduction_%'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Model size vs performance scatter plot
scatter = ax2.scatter(df['Parameters_M'], df['Speed_tokens_per_sec'], c=colors, s=200, alpha=0.7, edgecolors='black')
ax2.set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
ax2.set_xlabel('Parameters (Millions)')
ax2.set_ylabel('Speed (Tokens/sec)')
ax2.grid(True, alpha=0.3)

# Add labels for each point
for i, row in df.iterrows():
    ax2.annotate(f"{row['Model']}\n{row['Precision']}", 
                (row['Parameters_M'], row['Speed_tokens_per_sec']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.show()

# Print key insights
print("\n🔍 KEY INSIGHTS FROM VISUALIZATION:")
print("=" * 60)
print("1. 🚀 INT4 quantization shows 4.55× speedup on large models (Llama-3.2-1B)")
print("2. ⚠️  INT8 quantization shows 0.52× slowdown on small models (DialoGPT-small)")
print("3. 💾 Memory reduction: 50-75% with quantization")
print("4. 🖥️  GPU utilization: INT4 shows best efficiency (78.3%)")
print("5. 📏 Model size is critical for quantization benefits")
print("6. 🔧 Hardware/Software co-design: Tesla T4 limitations affect INT8 performance")
print("=" * 60)

print("\n✅ All visualizations created successfully!")
print("📊 Ready for analysis and presentation!")
