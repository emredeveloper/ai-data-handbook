


import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import requests

# Data and Embeddings
from sklearn.datasets import fetch_20newsgroups

# Dimensionality Reduction (UMAP)
try:
    import umap as _umap
    UMAP = _umap.UMAP if hasattr(_umap, "UMAP") else None
    if UMAP is None:
        from umap.umap_ import UMAP  # umap-learn canonical import
except Exception:
    try:
        from umap.umap_ import UMAP
    except Exception as e:
        raise ImportError(
            "UMAP bulunamadı. Lütfen 'pip install -U umap-learn' ile kurun (öncesinde 'pip uninstall -y umap')."
        ) from e

# Anomaly Detection Algorithms
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

HEADLESS = os.getenv("HEADLESS", "0") == "1"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class OpenAICompatClient:
    """Basit OpenAI uyumlu HTTP istemcisi (requests tabanlı)."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.session = requests.Session()

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def create_embeddings(self, model: str, input: list[str]) -> list[dict]:
        url = f"{self.base_url}/embeddings"
        resp = self.session.post(url, headers=self._headers(), json={"model": model, "input": input}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Beklenen: { data: [ { embedding: [...] }, ... ] }
        return data.get("data", [])

    def create_chat_completion(self, model: str, messages: list[dict], temperature: float = 0.3, max_tokens: int = 256) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self.session.post(url, headers=self._headers(), json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Beklenen: { choices: [ { message: { content: "..." } } ] }
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# --- Clients ---
# Embeddings: LM Studio (OpenAI uyumlu yerel sunucu)
emb_api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
embeddings_client = OpenAICompatClient(
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    api_key=emb_api_key
)

# Chat/Completions: Harici (varsayılan Nebius; değiştirilebilir)
chat_api_key = os.getenv("NEBIUS_API_KEY", os.getenv("OPENAI_API_KEY", "API_KEY_HERE"))
chat_client = OpenAICompatClient(
    base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
    api_key=chat_api_key
)

# Set a consistent style for plots
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
plt.rcParams.update({'font.size': 12})
if HEADLESS:
    try:
        plt.switch_backend('Agg')
    except Exception:
        pass


def finalize_plot(fig, filename: str | None = None):
    if HEADLESS:
        if not filename:
            filename = "plot.png"
        safe_name = filename.replace(" ", "_").lower()
        path = os.path.join(OUTPUT_DIR, safe_name)
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved figure to {path}")
    else:
        plt.show()

# ## 3. Data Loading and Preprocessing
# 
# ### 3.1. Load 20 Newsgroups Dataset
# We'll load the training subset of the '20 Newsgroups' dataset. This provides us with a rich source of labeled text data to work with.


# Data and Embeddings
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset="train")
print("20 Newsgroups Categories:")
print(newsgroups_train.target_names)

# Display the first document in the training set
print(newsgroups_train.data[0])

# ### 3.2. Clean and Structure the Data
# This function will clean the raw text by removing headers, emails, and names, and then structure it into a convenient Pandas DataFrame.


def clean_and_structure_data(dataset):
    """Cleans the raw text data and structures it into a Pandas DataFrame."""
    data = [re.sub(r"[\w\.-]+@[\w\.-]+", "", d) for d in dataset.data]
    data = [re.sub(r"\([^()]*\)", "", d) for d in data]
    data = [d.replace("From: ", "") for d in data]
    data = [d.replace("\nSubject: ", "") for d in data]
    data = [d.replace("Subject: ", "") for d in data]
    data = [d.replace("Lines:", "") for d in data]
    data = [d[:2000] for d in data]

    df = pd.DataFrame(data, columns=["Text"])
    df["Label"] = dataset.target
    df["Class Name"] = df["Label"].map(dataset.target_names.__getitem__)
    
    return df

df_full = clean_and_structure_data(newsgroups_train)
df_full.head()

# ### 3.3. Filter and Sample the Dataset
# To create a focused and manageable dataset, we select only the four science-related categories and then take a random sample of 150 documents from each. This gives us a balanced corpus of 600 documents.


def filter_and_sample_data(df, sample_size=150):
    """Filters for specific categories and samples the data."""
    df_sci = df[df["Class Name"].str.contains("sci")]
    df_sampled = (
        df_sci.groupby("Class Name", group_keys=False)
        .apply(lambda x: x.sample(sample_size, random_state=42))
        .reset_index(drop=True)
    )
    return df_sampled

df_train = filter_and_sample_data(df_full, sample_size=150)

print(df_train.info())
print("\nClass distribution in the sampled dataset:")
print(df_train["Class Name"].value_counts())

# ## 4. Generating High-Dimensional Embeddings
# 
# ### 4.1. Theory: BGE Embedding Models
# We will use the `BAAI/bge-en-icl` model for generating our text embeddings. BGE (Beijing Academy of Artificial Intelligence) models are renowned for their strong performance, especially in tasks related to information retrieval and semantic similarity. They are designed to map text to a high-dimensional vector space where the distance between vectors corresponds to semantic closeness.
# 
# ### 4.2. Code: Function for Batch Embedding via API
# This function takes our DataFrame and the initialized OpenAI client, then iterates through the text corpus in batches. For each batch, it calls the `client.embeddings.create` method and extracts the resulting vectors, which are then added back to our DataFrame.


def embed_corpus_with_api(df, client, model_name, batch_size=50):
    """Generates embeddings for the 'Text' column using the specified API client."""
    print(f"Generating embeddings using '{model_name}'...")
    all_embeddings = []
    text_corpus = df['Text'].tolist()
    
    for i in tqdm(range(0, len(text_corpus), batch_size)):
        batch = text_corpus[i:i+batch_size]
        try:
            response_items = client.create_embeddings(
                model=model_name,
                input=batch
            )
            embeddings = [np.array(item.get("embedding")) for item in response_items]
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"An error occurred during batch {i//batch_size}: {e}")
            all_embeddings.extend([None] * len(batch))
            
    df['Embeddings'] = all_embeddings
    df.dropna(subset=['Embeddings'], inplace=True)
    print("Embeddings generated and added to the DataFrame.")
    return df

df_train = embed_corpus_with_api(df_train, embeddings_client, model_name="text-embedding-embeddinggemma-300m")
print(f"Shape of the first embedding vector: {df_train['Embeddings'].iloc[0].shape}")

df_train.head()

# ## 5. Method 1: Baseline Anomaly Detection (Euclidean Distance)
# 
# ### 5.1. Theory: A Quick Recap of Centroid and Radius-Based Outliers
# Our first approach is a simple and intuitive baseline. For each category, we define a "center" point, or **centroid**, by calculating the average of all embedding vectors in that category. Anomalies are then identified as any points whose **Euclidean distance** from this centroid exceeds a predefined radius. This method assumes that each category forms a roughly spherical cluster in the embedding space.


def get_embedding_centroids(df):
    """Calculates the centroid of the embeddings for each class."""
    emb_centroids = {}
    grouped = df.groupby("Class Name")
    for class_name, group in grouped:
        emb_centroids[class_name] = np.mean(np.vstack(group['Embeddings']), axis=0)
    return emb_centroids

def calculate_euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two vectors."""
    return np.sqrt(np.sum(np.square(p1 - p2)))

def detect_outliers_euclidean(df, emb_centroids, radius):
    """Flags outliers based on Euclidean distance from the class centroid."""
    outlier_indices = []
    for idx, row in df.iterrows():
        class_name = row["Class Name"]
        dist = calculate_euclidean_distance(row["Embeddings"], emb_centroids[class_name])
        if dist > radius:
            outlier_indices.append(idx)
            
    df['Outlier_Euclidean'] = False
    df.loc[outlier_indices, 'Outlier_Euclidean'] = True
    return df

RADIUS = 0.55
baseline_centroids = get_embedding_centroids(df_train)
df_train = detect_outliers_euclidean(df_train, baseline_centroids, RADIUS)
num_outliers_baseline = df_train["Outlier_Euclidean"].sum()
print(f"Found {num_outliers_baseline} outliers with Euclidean method at radius {RADIUS}")

# ### 5.2. Visualizing Baseline Results
# Before diving into other methods, let's visualize the baseline results. We first project the high-dimensional data into 2D with UMAP and then highlight the outliers found by the Euclidean distance method. This plot will serve as a visual reference for comparison later.


def project_with_umap(df, n_neighbors=15, min_dist=0.1, random_state=42):
    """Projects high-dimensional embeddings into 2D using UMAP."""
    print("Projecting embeddings to 2D using UMAP...")
    embeddings = np.vstack(df['Embeddings'])
    reducer = UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=random_state,
        metric='cosine'
    )
    umap_results = reducer.fit_transform(embeddings)
    
    df_umap = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
    df_umap = pd.concat([df_umap, df.reset_index(drop=True)], axis=1)
    print("UMAP projection complete.")
    return df_umap

df_umap = project_with_umap(df_train)

def plot_single_method_outliers(df_plot, outlier_col, method_name):
    """Generic function to plot UMAP projection highlighting outliers for one method."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    inliers = df_plot[df_plot[outlier_col] == False]
    sns.scatterplot(data=inliers, x='UMAP1', y='UMAP2', hue='Class Name', palette='viridis', ax=ax, alpha=0.6)
    
    outliers = df_plot[df_plot[outlier_col] == True]
    sns.scatterplot(data=outliers, x='UMAP1', y='UMAP2', color='red', marker='X', s=150, label='Outlier', ax=ax)
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title(f'UMAP Projection with Outliers Detected by {method_name}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    finalize_plot(fig, f"umap_{method_name.replace(' ', '_').lower()}.png")

plot_single_method_outliers(df_umap, 'Outlier_Euclidean', 'Euclidean Distance')

# ## 6. Method 2: Advanced Anomaly Detection Algorithms
# Now we apply our three advanced algorithms and visualize their results individually.


# ### 6.1. Algorithm A: Statistical Outliers with Mahalanobis Distance
# 
# #### 6.1.1. Theory: Accounting for Cluster Shape and Variance
# The Mahalanobis distance improves upon Euclidean distance by considering the covariance of the data. This makes it excellent for identifying outliers in non-spherical clusters. We set a threshold using the chi-squared distribution, a statistically-grounded way to determine an "unlikely" distance.


def detect_outliers_mahalanobis(df, confidence=0.99):
    """Flags outliers based on Mahalanobis distance for each class."""
    outlier_indices = []
    grouped = df.groupby('Class Name')
    
    for class_name, group in grouped:
        embeddings = np.vstack(group['Embeddings'])
        centroid = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings.T)
        inv_cov = np.linalg.inv(cov + np.identity(cov.shape[0]) * 1e-6)
        
        distances = [mahalanobis(emb, centroid, inv_cov) for emb in embeddings]
        threshold = chi2.ppf(confidence, df=embeddings.shape[1])
        print(f"Mahalanobis threshold for '{class_name}': {threshold:.2f}")
        
        group_outlier_indices = group.index[np.array(distances) > threshold].tolist()
        outlier_indices.extend(group_outlier_indices)
        
    df['Outlier_Mahalanobis'] = False
    df.loc[outlier_indices, 'Outlier_Mahalanobis'] = True
    return df

df_umap = detect_outliers_mahalanobis(df_umap, confidence=0.99)
num_outliers = df_umap["Outlier_Mahalanobis"].sum()
print(f"Found {num_outliers} outliers with Mahalanobis Distance method.")

plot_single_method_outliers(df_umap, 'Outlier_Mahalanobis', 'Mahalanobis Distance')

# ### 6.2. Algorithm B: Density-Based Outliers with Local Outlier Factor (LOF)
# 
# #### 6.2.1. Theory: Identifying Points in Low-Density Neighborhoods
# LOF measures the local density of each data point relative to its neighbors. An outlier is a point that has a substantially lower density than its neighbors. This makes it powerful for finding anomalies in isolated, sparse regions of the embedding space.


def detect_outliers_lof(df, n_neighbors=20, contamination='auto'):
    """Flags outliers using the Local Outlier Factor algorithm."""
    embeddings = np.vstack(df['Embeddings'])
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(embeddings)
    
    df['Outlier_LOF'] = (predictions == -1)
    return df

df_umap = detect_outliers_lof(df_umap)
num_outliers = df_umap["Outlier_LOF"].sum()
print(f"Found {num_outliers} outliers with Local Outlier Factor method.")

plot_single_method_outliers(df_umap, 'Outlier_LOF', 'Local Outlier Factor')

# ### 6.3. Algorithm C: Isolation-Based Outliers with Isolation Forest
# 
# #### 6.3.1. Theory: The "Few and Different" Principle
# Isolation Forest works on the principle that anomalies are easier to "isolate" from the data than normal points. It builds random trees to partition the data, and the average path length to isolate a point determines its anomaly score. We set `contamination` to define our expected proportion of outliers (e.g., 5%).


def detect_outliers_isoforest(df, contamination=0.05, random_state=42):
    """Flags outliers using the Isolation Forest algorithm."""
    embeddings = np.vstack(df['Embeddings'])
    
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = iso_forest.fit_predict(embeddings)
    
    df['Outlier_ISO'] = (predictions == -1)
    return df

df_umap = detect_outliers_isoforest(df_umap, contamination=0.05)
num_outliers = df_umap["Outlier_ISO"].sum()
print(f"Found {num_outliers} outliers with Isolation Forest method.")

plot_single_method_outliers(df_umap, 'Outlier_ISO', 'Isolation Forest')

# ## 7. Comparative Analysis and Final Visualization
# 
# ### 7.1. Final Outlier Summary
# Now we aggregate the results. We are most interested in the points that multiple advanced algorithms agree on. These are our **high-confidence outliers**.


outlier_cols = ['Outlier_Euclidean', 'Outlier_Mahalanobis', 'Outlier_LOF', 'Outlier_ISO']
summary = df_umap[outlier_cols].sum().to_dict()

advanced_cols = ['Outlier_Mahalanobis', 'Outlier_LOF', 'Outlier_ISO']
df_umap['Outlier_HighConfidence'] = df_umap[advanced_cols].all(axis=1)
summary['High-Confidence'] = df_umap['Outlier_HighConfidence'].sum()

print("--- Final Outlier Detection Summary ---")
print(f"Euclidean Distance:      {summary['Outlier_Euclidean']}")
print(f"Mahalanobis Distance:    {summary['Outlier_Mahalanobis']}")
print(f"Local Outlier Factor:    {summary['Outlier_LOF']}")
print(f"Isolation Forest:        {summary['Outlier_ISO']}")
print("---------------------------------")
print(f"High-Confidence Outliers (Mahalanobis AND LOF AND ISO): {summary['High-Confidence']}")

# ### 7.2. Final Comparative Visualization
# This final plot synthesizes all our findings. It shows the data points colored by their ground-truth category. Outliers from each of the three advanced methods are marked with different symbols. The most important points—our 18 high-confidence outliers—are highlighted with large red 'X's. This gives a clear, multi-layered view of the anomalies in our dataset.


def plot_outliers_final_comparison(df_plot):
    """Plots the UMAP projection, highlighting outliers from different methods with a focus on high-confidence ones."""
    fig, ax = plt.subplots(figsize=(12, 9))

    sns.scatterplot(
        data=df_plot,
        x="UMAP1", 
        y="UMAP2", 
        hue="Class Name", 
        palette="viridis", 
        alpha=0.4, 
        ax=ax
    )

    iso_only = df_plot[(df_plot['Outlier_ISO'] == True) & (df_plot['Outlier_HighConfidence'] == False)]
    sns.scatterplot(data=iso_only, x='UMAP1', y='UMAP2', color='orange', marker='s', s=70, label='Isolation Forest Only', ax=ax, alpha=0.7)
    
    lof_only = df_plot[(df_plot['Outlier_LOF'] == True) & (df_plot['Outlier_HighConfidence'] == False)]
    sns.scatterplot(data=lof_only, x='UMAP1', y='UMAP2', facecolors='none', edgecolors='cyan', marker='o', s=100, label='LOF Only', ax=ax, linewidth=1.5)
    
    maha_only = df_plot[(df_plot['Outlier_Mahalanobis'] == True) & (df_plot['Outlier_HighConfidence'] == False)]
    sns.scatterplot(data=maha_only, x='UMAP1', y='UMAP2', color='magenta', marker='D', s=70, label='Mahalanobis Only', ax=ax, alpha=0.7)
    
    high_conf_outliers = df_plot[df_plot['Outlier_HighConfidence'] == True]
    sns.scatterplot(data=high_conf_outliers, x='UMAP1', y='UMAP2', color='red', marker='X', s=250, label='High-Confidence Outlier (All 3)', ax=ax)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title("UMAP Projection of Newsgroup Embeddings with Outlier Comparison")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    finalize_plot(fig, "umap_outlier_comparison.png")

plot_outliers_final_comparison(df_umap)

# ## 8. Explainable AI: Using Llama-3.1 for Interpretation
# 
# ### 8.1. Theory: Leveraging Generative Models for Semantic Explanation
# We now move to the final, interpretive step. Using the `meta-llama/Meta-Llama-3.1-8B-Instruct` model via our custom API, we will generate a natural language explanation for one of our high-confidence outliers. This adds a critical layer of explainability to our quantitative findings.


def explain_outlier_with_llm(client, outlier_doc, inlier_docs, category, model_name):
    """Uses a configured LLM pipeline to generate an explanation for an outlier."""
    
    inlier_text = "\n\n".join([f"--- Normal Document {i+1} ---\n{doc}" for i, doc in enumerate(inlier_docs)])
    
    system_prompt = "You are an expert data analyst. Your task is to explain why a document is a semantic outlier within its given category. Analyze the content and themes, and provide a concise, one-paragraph explanation highlighting the key differences."
    
    user_message = f"""
    Category: '{category}'

    --- Outlier Document ---
    {outlier_doc}

    {inlier_text}

    **Explanation of the Outlier:**
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        content = client.create_chat_completion(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=256
        )
        return content.strip()
    except Exception as e:
        return f"An error occurred while generating the explanation: {e}"

# Select a high-confidence outlier
high_confidence_outliers = df_umap[df_umap['Outlier_HighConfidence'] == True]

# Güvenli seçim akışı
outlier_to_explain = None

if not high_confidence_outliers.empty:
    subset = high_confidence_outliers[high_confidence_outliers['Class Name'] == 'sci.electronics']
    if subset.empty:
        subset = high_confidence_outliers
    outlier_to_explain = subset.iloc[0]
else:
    # 1) ISO
    iso_only = df_umap[df_umap['Outlier_ISO'] == True]
    subset = iso_only[iso_only['Class Name'] == 'sci.electronics'] if not iso_only.empty else iso_only
    if subset.empty:
        # 2) LOF
        lof_only = df_umap[df_umap['Outlier_LOF'] == True]
        subset = lof_only[lof_only['Class Name'] == 'sci.electronics'] if not lof_only.empty else lof_only
    if subset.empty:
        # 3) Mahalanobis
        maha_only = df_umap[df_umap['Outlier_Mahalanobis'] == True]
        subset = maha_only[maha_only['Class Name'] == 'sci.electronics'] if not maha_only.empty else maha_only
    if subset.empty:
        # 4) En uzak Euclidean (merkezden en uzak nokta)
        centroids = get_embedding_centroids(df_train)
        df_umap['euclid_dist'] = df_umap.apply(lambda r: calculate_euclidean_distance(r['Embeddings'], centroids[r['Class Name']]), axis=1)
        subset = df_umap.sort_values('euclid_dist', ascending=False)
    outlier_to_explain = subset.iloc[0]

# Select some normal inliers from the same category
inliers = df_umap[
    (df_umap['Class Name'] == 'sci.electronics') & 
    (df_umap['Outlier_HighConfidence'] == False)
]
sample_n = min(3, max(1, len(inliers)))
inlier_examples = inliers.sample(sample_n, random_state=42)['Text'].tolist() if sample_n > 0 else df_umap[df_umap['Class Name'] == outlier_to_explain['Class Name']].sample(3, replace=True, random_state=42)['Text'].tolist()

print("--- GENERATING EXPLANATION FOR OUTLIER ---\n")
print(f"Category: {outlier_to_explain['Class Name']}\n")
print(f"**Outlier Document Text:**\n{outlier_to_explain['Text']}")
print("\n----------------------------------------\n")

# Generate the explanation using the API
explanation = explain_outlier_with_llm(
    chat_client, 
    outlier_to_explain['Text'], 
    inlier_examples, 
    outlier_to_explain['Class Name'],
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
)

print(f"**LLM Explanation:**\n{explanation}")

# ## 9. Bonus: Unsupervised Clustering and Comparison with Anomaly Detection
# 
# ### 9.1. Theory: Unsupervised Clustering vs. Anomaly Detection
# We now perform **unsupervised clustering** using K-Means to see if a machine can rediscover the original categories without being given the labels. Since we know our sampled dataset has four distinct science categories, we will set `K=4`.


def apply_kmeans(df, n_clusters=4, random_state=42):
    """Applies K-Means clustering to the embeddings and adds cluster labels to the DataFrame."""
    embeddings = np.vstack(df['Embeddings'])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['KMeans_Cluster'] = kmeans.fit_predict(embeddings)
    
    print("K-Means clustering complete. Cluster labels added to the DataFrame.")
    return df

# ### 9.2. Visualizing Ground Truth vs. K-Means Clusters
# Let's create a side-by-side comparison. The left plot shows the UMAP projection colored by the **original, ground-truth labels**. The right plot shows the exact same points, but colored by the **cluster labels predicted by K-Means**. This comparison is a powerful way to validate the quality of our embeddings.


def plot_kmeans_vs_ground_truth(df_plot):
    """Creates a side-by-side plot comparing K-Means clusters to ground truth labels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True, sharex=True)
    
    # Plot Ground Truth
    sns.scatterplot(data=df_plot, x='UMAP1', y='UMAP2', hue='Class Name', palette='viridis', ax=ax1, alpha=0.8)
    ax1.set_title('Ground Truth Labels')
    ax1.legend(title='Category')
    
    # Plot K-Means Results
    sns.scatterplot(data=df_plot, x='UMAP1', y='UMAP2', hue='KMeans_Cluster', palette='viridis', ax=ax2, alpha=0.8)
    ax2.set_title('K-Means Predicted Clusters')
    ax2.legend(title='Cluster ID')

    fig.suptitle('UMAP Projection: Ground Truth vs. K-Means Clustering', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    finalize_plot(fig, "umap_kmeans_vs_truth.png")

# Run K-Means and plot comparison
df_umap = apply_kmeans(df_umap, n_clusters=4)
plot_kmeans_vs_ground_truth(df_umap)

# ### 9.3. Combined Visualization: K-Means Clusters and High-Confidence Outliers
# 
# Finally, let's combine our clustering and anomaly detection results. The following plot shows the documents colored by their **K-Means predicted cluster**, with our **high-confidence outliers** overlaid as large red 'X's. This visualization helps us understand the nature of our outliers in the context of the discovered clusters.


def plot_clusters_with_outliers(df_plot):
    """Plots the K-Means clusters and highlights the high-confidence outliers."""
    fig, ax = plt.subplots(figsize=(12, 9))
    
    sns.scatterplot(
        data=df_plot, 
        x='UMAP1', 
        y='UMAP2', 
        hue='KMeans_Cluster', 
        palette='viridis', 
        ax=ax, 
        alpha=0.7, 
        legend='full'
    )
    
    high_conf_outliers = df_plot[df_plot['Outlier_HighConfidence'] == True]
    sns.scatterplot(data=high_conf_outliers, x='UMAP1', y='UMAP2', color='red', marker='X', s=250, label='High-Confidence Outlier', ax=ax)
    
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title('K-Means Clusters with High-Confidence Outliers')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    finalize_plot(fig, "umap_kmeans_with_outliers.png")

plot_clusters_with_outliers(df_umap)

# ## 10. Conclusion
# 
# This notebook successfully demonstrated a powerful, flexible, and interpretable pipeline for text anomaly detection using a custom OpenAI-compatible API endpoint. We have shown that:
# 
# 1.  **High-Quality Embeddings are Key**: The `BAAI/bge-en-icl` model provided excellent semantic representations, enabling both unsupervised clustering and sophisticated anomaly detection to work effectively.
# 
# 2.  **Advanced Algorithms Provide Nuance**: While a simple Euclidean baseline works, combining statistical (Mahalanobis), density-based (LOF), and isolation-based (Isolation Forest) methods gives a more robust and high-confidence view of what constitutes an anomaly.
# 
# 3.  **Visualization is Critical for Understanding**: UMAP proved to be an effective tool for visualizing the high-dimensional space, clearly showing the cluster structures and the position of outliers.
# 
# 4.  **Explainability is Achievable**: By leveraging a powerful LLM like `meta-llama/Meta-Llama-3.1-8B-Instruct`, we can bridge the gap between a quantitative anomaly score and a qualitative, human-understandable reason, making the entire system more valuable and actionable.


if __name__ == "__main__":
    # Normal script çalıştırma
    pass


# --- Flask UI ---
try:
    from flask import Flask, render_template_string, redirect, url_for
    import subprocess
except Exception:
    Flask = None


INDEX_HTML = """
<!doctype html>
<title>Outlier Text Pipeline</title>
<h2>Outlier Text Pipeline</h2>
<p>Embedding model: <b>text-embedding-embeddinggemma-300m</b> (LM Studio)</p>
<form action="{{ url_for('run_pipeline') }}" method="post">
  <button type="submit">Çalıştır (HEADLESS)</button>
</form>
{% if msg %}
<pre>{{ msg }}</pre>
{% endif %}
<hr/>
<p>Çıktı görselleri 'outputs' klasörüne kaydedilir.</p>
"""


def launch_pipeline_headless() -> str:
    env = os.environ.copy()
    env["HEADLESS"] = "1"
    python_exe = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv", "Scripts", "python.exe"))
    if not os.path.exists(python_exe):
        python_exe = "python"
    try:
        proc = subprocess.run([python_exe, os.path.abspath(__file__)], cwd=os.path.dirname(__file__), env=env, capture_output=True, text=True, timeout=3600)
        output = proc.stdout + "\n" + proc.stderr
        return output[-4000:]
    except Exception as e:
        return f"Çalıştırma hatası: {e}"


def create_app():
    if Flask is None:
        raise RuntimeError("Flask yüklü değil. 'pip install flask' ile kurun.")
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(INDEX_HTML, msg=None)

    @app.route("/run", methods=["POST"])
    def run_pipeline():
        msg = launch_pipeline_headless()
        return render_template_string(INDEX_HTML, msg=msg)

    return app


if __name__ == "__main__":
    # Eğer doğrudan Flask arayüzü isteniyorsa: `python app.py flask`
    import sys
    if len(sys.argv) > 1 and sys.argv[1].lower() == "flask":
        app = create_app()
        app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
    else:
        # Mevcut script akışını çalıştır
        # Yukarıdaki kodlar modül yüklenirken koştu; yalnızca çıktıyı sonlandırıyoruz.
        pass