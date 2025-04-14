import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ======================
# 1. STYLE CONFIGURATION
# ======================
sns.set_style("whitegrid")
BG_COLOR = '#F8F9FA'
plt.rcParams['figure.facecolor'] = BG_COLOR

PALETTE = {
    'clusters': ['#2E86AB', '#A23B72', '#F18F01'],
    'sections': ['#264653', '#2A9D8F'],
    'promotion': ['#3A606E', '#E76F51'],
    'heatmap': 'mako_r'
}

# ======================
# 2. DATA PREPARATION 
#Please download the dataset(cleaned_zara_dataset) and load it locally from your computer, you can adjust the path accordingly
# ======================
df = pd.read_excel("cleaned_zara_dataset.xlsx")
df = df[['Product Category', 'price', 'Sales Volume', 'Promotion', 'section']].dropna()
df['Promotion'] = df['Promotion'].map({'Yes': 1, 'No': 0})

# ======================
# 3. CLUSTER ANALYSIS
# ======================
X = df[['price', 'Sales Volume']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(14, 6), facecolor=BG_COLOR)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.subplot(121)
plt.plot(range(1, 11), wcss, marker='o', color='#2A9D8F', markersize=8, linewidth=2.5)
plt.title('Optimal Cluster Determination', fontsize=14, pad=20)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Within-Cluster Variance', fontsize=12)
plt.axvline(x=3, linestyle='--', color='#E76F51', linewidth=2)
plt.text(3.1, max(wcss)*0.8, 'Optimal Clusters: 3', fontsize=12, color='#E76F51', fontweight='bold')
plt.gca().set_facecolor(BG_COLOR)

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

# ======================
# 4. CHOICE MODEL
# ======================
plt.figure(figsize=(16, 8), facecolor=BG_COLOR)
choice_data = df.groupby(['section', 'Product Category'])['Sales Volume'].mean().reset_index()

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

# ======================
# 5. CONJOINT ANALYSIS
# ======================
plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
heatmap_data = df.pivot_table(
    index='price', 
    columns='section', 
    values='Sales Volume', 
    aggfunc='mean'
)

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

# ======================
# 6. MARKET RESPONSE MODEL 
# ======================
plt.figure(figsize=(14, 7), facecolor=BG_COLOR)
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
        'linewidths': 0.5  # Changed from 'linewidth' to 'linewidths'
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
