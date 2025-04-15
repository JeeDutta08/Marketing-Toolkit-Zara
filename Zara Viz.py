# =================================================================
# IMPORT ESSENTIAL LIBRARIES
# We import data manipulation, visualization, and machine learning tools
# pandas: Data handling and analysis
# matplotlib: Base plotting functionality
# seaborn: Advanced statistical visualizations
# sklearn: Machine learning algorithms and preprocessing
# =================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =================================================================
# 1. VISUAL STYLE CONFIGURATION
# Sets consistent styling for all plots to maintain brand consistency
# and improve readability. Color palette choices enhance visual
# distinction between different data categories.
# =================================================================
sns.set_style("whitegrid")
BG_COLOR = '#F8F9FA'  # Light background for modern aesthetic
plt.rcParams['figure.facecolor'] = BG_COLOR

# Custom color palette for different analysis components
PALETTE = {
    'clusters': ['#2E86AB', '#A23B72', '#F18F01'],  # Cluster distinction colors
    'sections': ['#264653', '#2A9D8F'],              # Department comparison
    'promotion': ['#3A606E', '#E76F51'],             # Promotion status colors
    'heatmap': 'mako_r'                              # Price-section analysis
}

# =================================================================
# 2. DATA PREPARATION & CLEANING (Data file is present here in GitHub name- cleaned_zara_dataset.xlsx)
# Load and prepare the dataset for analysis. Key steps:
# - Filter relevant columns
# - Handle missing values
# - Convert categorical variables to numerical
# Note: Update file path according to your local storage location
# =================================================================
df = pd.read_excel("cleaned_zara_dataset.xlsx")

# Select key features and clean data
df = df[['Product Category', 'price', 'Sales Volume', 'Promotion', 'section']].dropna()

# Convert Yes/No promotions to binary (1/0) for mathematical operations
df['Promotion'] = df['Promotion'].map({'Yes': 1, 'No': 0})

# =================================================================
# 3. CUSTOMER SEGMENTATION ANALYSIS (K-MEANS CLUSTERING)
# Identify distinct customer groups based on price sensitivity and
# purchasing behavior. Uses scaled data for proper cluster formation.
# =================================================================

# Prepare features for clustering
X = df[['price', 'Sales Volume']]

# Standardize features to equalize scale (crucial for distance-based algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create visualization canvas
plt.figure(figsize=(14, 6), facecolor=BG_COLOR)

# Elbow Method to Determine Optimal Clusters
# Calculates within-cluster variance for different cluster counts
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Store variance metric

# Plot Elbow Curve
plt.subplot(121)
plt.plot(range(1, 11), wcss, marker='o', color='#2A9D8F', markersize=8, linewidth=2.5)
plt.title('Optimal Cluster Determination', fontsize=14, pad=20)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Within-Cluster Variance', fontsize=12)
plt.axvline(x=3, linestyle='--', color='#E76F51', linewidth=2)
plt.text(3.1, max(wcss)*0.8, 'Optimal Clusters: 3', fontsize=12, color='#E76F51', fontweight='bold')
plt.gca().set_facecolor(BG_COLOR)

# Final Cluster Visualization
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.subplot(122)
sns.scatterplot(
    x=X['price'], 
    y=X['Sales Volume'], 
    hue=clusters, 
    palette=PALETTE['clusters'],
    s=100,
    edgecolor='w',
    linewidth=0.5
)
# Plot cluster centers (reverse scaled to original values)
plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:,0],
    scaler.inverse_transform(kmeans.cluster_centers_)[:,1],
    s=400, 
    c='#E76F51', 
    marker='X',
    edgecolor='k',
    linewidth=1
)
plt.title('Customer Segmentation Analysis', fontsize=14, pad=20)
plt.xlabel('Price (USD)', fontsize=12)
plt.ylabel('Sales Volume', fontsize=12)
plt.legend(title='Cluster', title_fontsize=12)
plt.gca().set_facecolor(BG_COLOR)
plt.tight_layout()
plt.savefig('cluster_analysis.png', dpi=300, facecolor=BG_COLOR)
plt.show()

# =================================================================
# 4. PRODUCT PREFERENCE ANALYSIS (CHOICE MODEL)
# Compare average sales performance across product categories
# and departments to identify popular combinations
# =================================================================
plt.figure(figsize=(16, 8), facecolor=BG_COLOR)

# Aggregate sales data by product category and department
choice_data = df.groupby(['section', 'Product Category'])['Sales Volume'].mean().reset_index()

# Create comparative bar plot
sns.barplot(
    x='Product Category', 
    y='Sales Volume', 
    hue='section', 
    data=choice_data,
    palette=PALETTE['sections'],
    edgecolor='w',
    linewidth=1.5
)
plt.title('Product Preference Analysis by Department', fontsize=14, pad=20)
plt.xlabel('Product Category', fontsize=12)
plt.ylabel('Average Sales Volume', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Department', title_fontsize=12, fontsize=11, frameon=True)
plt.gca().set_facecolor(BG_COLOR)
plt.tight_layout()
plt.savefig('choice_model.png', dpi=300, facecolor=BG_COLOR)
plt.show()

# =================================================================
# 5. PRICE-SENSITIVITY ANALYSIS (CONJOINT ANALYSIS)
# Heatmap showing how different prices perform in various departments
# Helps identify optimal pricing strategies per section
# =================================================================
plt.figure(figsize=(14, 8), facecolor=BG_COLOR)

# Create price-section utility matrix
heatmap_data = df.pivot_table(
    index='price', 
    columns='section', 
    values='Sales Volume', 
    aggfunc='mean'
)

# Generate heatmap with annotated values
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".0f", 
    cmap=PALETTE['heatmap'], 
    linewidths=.5, 
    cbar_kws={'label': 'Sales Volume'}
)
plt.title('Price-Section Utility Matrix', fontsize=14, pad=20)
plt.xlabel('Department Section', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.tight_layout()
plt.savefig('conjoint_analysis.png', dpi=300, facecolor=BG_COLOR)
plt.show()

# =================================================================
# 6. PROMOTIONAL IMPACT ANALYSIS (MARKET RESPONSE MODEL)
# Visualize relationship between pricing, promotions, and sales
# Helps understand price elasticity and promotional effectiveness
# =================================================================
plt.figure(figsize=(14, 7), facecolor=BG_COLOR)

# Create regression plot with promotion distinction
lm = sns.lmplot(
    x='price', 
    y='Sales Volume', 
    hue='Promotion', 
    data=df,
    palette=PALETTE['promotion'],
    height=6, 
    aspect=1.8,
    scatter_kws={
        's': 80,
        'edgecolor': 'w',
        'linewidths': 0.5  # Corrected parameter name
    },
    line_kws={'linewidth': 2.5}
)

plt.title('Pricing & Promotion Elasticity', fontsize=14, pad=20)
plt.xlabel('Price (USD)', fontsize=12)
plt.ylabel('Sales Volume', fontsize=12)
plt.annotate(
    'Promotional Lift Effect', 
    xy=(150, 2500), 
    xytext=(200, 3000), 
    arrowprops=dict(arrowstyle='->', color='#2A9D8F', linewidth=1.5),
    fontsize=12,
    color='#2A9D8F'
)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('market_response.png', dpi=300, facecolor=BG_COLOR)
plt.show()
