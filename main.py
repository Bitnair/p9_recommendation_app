# 'main.py'
"""
handles data loading, preprocessing, and evaluation for both Content-Based and Collaborative Filtering approaches.
"""

import os
import math
import random
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from sklearn.metrics import precision_score, recall_score, f1_score

# ------------------------------
# Data Loading Functions
# ------------------------------

def load_clicks_data(input_dir):
    df_list = []
    clicks_dir = os.path.join(input_dir, "clicks")

    if not os.path.exists(clicks_dir):
        raise FileNotFoundError(f"The directory '{clicks_dir}' does not exist.")

    for dirname, _, filenames in os.walk(clicks_dir):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
            if filename.endswith(".csv"):  # Ensure it's a CSV file
                try:
                    df = pd.read_csv(file_path)
                    df_list.append(df)
                    print(f"Loaded file: {file_path} ({len(df)} rows)")
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")

    if not df_list:
        raise ValueError(f"No valid CSV files found in '{clicks_dir}'.")
    
    return pd.concat(df_list, ignore_index=True)

def load_items_data(input_dir):
    items_df = pd.read_csv(os.path.join(input_dir, "articles_metadata.csv"))
    return items_df

def load_embeddings(input_dir):
    with open(os.path.join(input_dir, "articles_embeddings.pickle"), "rb") as f:
        embedding = pickle.load(f)
    return embedding

def preprocess_interactions(df):
    df["click_count"] = 1
    df = df.groupby(["user_id", "click_article_id"])["click_count"].sum()
    df = df.apply(lambda x: math.log(1 + x, 2)).reset_index()
    return df

def verify_embeddings(cb_recommender, interactions_test_df):
    test_items = set(interactions_test_df['click_article_id'].unique())
    missing_items = test_items - set(cb_recommender.item_ids)
    print(f"Number of test items missing in embeddings: {len(missing_items)}")
    if len(missing_items) > 0:
        print("Some test items are missing in the embeddings. Consider updating the embeddings to include these items.")


# ------------------------------
# Visualization Function
# ------------------------------

def plot_metrics(metrics_dict, title):
    ks = sorted(metrics_dict.keys())
    precisions = [metrics_dict[k]["precision"] for k in ks]
    recalls = [metrics_dict[k]["recall"] for k in ks]
    f1_scores = [metrics_dict[k]["f1_score"] for k in ks]

    x = np.arange(len(ks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, precisions, width, label='Precision')
    rects2 = ax.bar(x, recalls, width, label='Recall')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score')

    ax.set_xlabel('k')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.legend()
    plt.show()


# ------------------------------
# Evaluation Function
# ------------------------------

def evaluate_recommender(
    recommender, 
    recommend_method, 
    interactions_train_df, 
    interactions_test_df, 
    n_users=100, 
    k=10
):
    """
    Evaluate the recommender using test data to compute precision, recall, and F1-score.

    Parameters:
        recommender: The recommender object.
        recommend_method: Method to get recommendations.
        interactions_train_df: Training interactions DataFrame.
        interactions_test_df: Test interactions DataFrame.
        n_users: Number of users to sample for evaluation.
        k: Number of top recommendations to evaluate.

    Returns:
        dict: A dictionary containing precision, recall, and F1-score.
    """
    # Users present in both train and test sets
    train_users = set(interactions_train_df['user_id'].unique())
    test_users = set(interactions_test_df['user_id'].unique())
    common_users = list(train_users & test_users)

    print(f"Total common users for evaluation: {len(common_users)}")

    # Ensure we have enough users to sample
    if n_users > len(common_users):
        print(f"Reducing n_users to {len(common_users)} (users in both train and test).")
        n_users = len(common_users)

    # Sample users for evaluation
    sampled_users = random.sample(common_users, n_users)
    precisions, recalls = [], []

    for idx, user_id in enumerate(sampled_users, 1):
        print(f"\nEvaluating user {idx}/{n_users}: {user_id}")

        # Get actual items for the user from the test data
        user_interactions_test = interactions_test_df[interactions_test_df['user_id'] == user_id]
        actual_items = set(user_interactions_test['click_article_id'].unique())
        print(f"  Actual items: {actual_items}")
        if not actual_items:
            print(f"  No actual items for user {user_id} in test data. Skipping.")
            continue

        # Get recommended items for the user
        recommended_items = recommend_method(user_id, k)
        print(f"  Recommended items: {recommended_items}")
        if not recommended_items:
            print(f"  No recommendations for user {user_id}. Skipping.")
            continue

        hits = len(actual_items & recommended_items)
        print(f"  Hits: {hits}")

        # Compute precision and recall
        precision = hits / k
        recall = hits / len(actual_items)

        precisions.append(precision)
        recalls.append(recall)

    # Calculate average precision, recall, and F1-score
    if precisions and recalls:
        precision_avg = np.mean(precisions)
        recall_avg = np.mean(recalls)
        f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg) if (precision_avg + recall_avg) else 0
    else:
        precision_avg, recall_avg, f1 = 0.0, 0.0, 0.0

    return {"precision": precision_avg, "recall": recall_avg, "f1_score": f1}


# ------------------------------
# Debugging Functions
# ------------------------------
def manual_inspection(cb_recommender, cf_recommender, interactions_test_df, num_users=5):
    # Sample a few users who have interactions in both train and test sets
    common_users = list(set(cb_recommender.train_df.index) & set(interactions_test_df['user_id'].unique()))
    sampled_users = random.sample(common_users, num_users)
    
    for user_id in sampled_users:
        # Get actual items
        actual_items = set(interactions_test_df[interactions_test_df['user_id'] == user_id]['click_article_id'].unique())
        
        # Get recommended items
        recommended_items_cb = cb_recommender.get_recommended_items(user_id, k=10)
        recommended_items_cf = cf_recommender.get_recommended_items(user_id, k=10)
        
        # Calculate hits
        hits_cb = actual_items & recommended_items_cb
        hits_cf = actual_items & recommended_items_cf
        
        # Print results
        print(f"\nUser ID: {user_id}")
        print(f"  Actual Items: {actual_items}")
        print(f"  CB Recommended Items: {recommended_items_cb}")
        print(f"  CF Recommended Items: {recommended_items_cf}")
        print(f"  CB Hits: {hits_cb}")
        print(f"  CF Hits: {hits_cf}")


# ------------------------------
# Recommender Classes
# ------------------------------

class ContentBasedRecommender:
    """
    Recommend top n articles based on content-based filtering.
    """

    def __init__(self, interactions_df, items_df, embedding_matrix):
        self.train_df = interactions_df.set_index("user_id")
        self.items_df = items_df
        self.embedding = normalize(embedding_matrix, axis=1)  # Normalize embeddings
        self.item_ids = items_df["article_id"].tolist()
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def get_item_profile(self, item_id):
        idx = self.item_id_to_idx.get(item_id)
        if idx is not None:
            return self.embedding[idx]
        else:
            print(f"Item ID {item_id} not found in item_ids.")
            return np.zeros(self.embedding.shape[1])

    def get_items_profiles(self, ids):
        return np.array([self.get_item_profile(x) for x in ids])

    def build_user_profile(self, user_id):
        if user_id not in self.train_df.index:
            print(f"User {user_id} not found in training data.")
            return np.zeros(self.embedding.shape[1])
        
        interactions = self.train_df.loc[user_id]
        if isinstance(interactions, pd.Series):
            interactions = interactions.to_frame().T
        item_profiles = self.get_items_profiles(interactions["click_article_id"])
        click_counts = np.array(interactions["click_count"]).reshape(-1, 1)
        if np.sum(click_counts) == 0:
            return np.zeros(self.embedding.shape[1])
        user_profile = np.sum(item_profiles * click_counts, axis=0) / np.sum(click_counts)
        
        # Normalize user profile
        norm = np.linalg.norm(user_profile)
        if norm != 0:
            user_profile = user_profile / norm
        return user_profile

    def get_similar_items(self, user_id, topn=100):  # Increase topn
        user_profile = self.build_user_profile(user_id)
        if np.all(user_profile == 0):
            return []
        cosine_sim = cosine_similarity([user_profile], self.embedding)
        similar_indices = cosine_sim.argsort().flatten()[::-1]
        similar_items = [
            (self.item_ids[i], cosine_sim[0, i]) for i in similar_indices
        ]
        return similar_items[:topn]

    def get_items_interacted(self, user_id):
        if user_id not in self.train_df.index:
            return set()
        interacted_items = self.train_df.loc[user_id]["click_article_id"]
        if isinstance(interacted_items, pd.Series):
            return set(interacted_items)
        else:
            return {interacted_items}

    def recommend_items(self, user_id, topn=10, verbose=False):
        if user_id not in self.train_df.index:
            print(f"User {user_id} not found in training data.")
            return pd.DataFrame()

        items_to_ignore = self.get_items_interacted(user_id)
        print(f"  Items to ignore for user {user_id}: {items_to_ignore}")
        
        similar_items = self.get_similar_items(user_id, topn=100)  # Retrieve more similar items
        print(f"  Retrieved {len(similar_items)} similar items.")
        
        recommendations = [
            item for item in similar_items if item[0] not in items_to_ignore
        ][:topn]
        print(f"  Filtered recommendations (top {topn}): {recommendations}")
        
        if not recommendations:
            print(f"No recommendations for user {user_id}.")
            return pd.DataFrame()
        
        recommendations_df = pd.DataFrame(recommendations, columns=["click_article_id", "similarity"])
        if verbose:
            recommendations_df = recommendations_df.merge(
                self.items_df,
                how="left",
                left_on="click_article_id",
                right_on="article_id"
            )
            recommendations_df = recommendations_df[["click_article_id", "similarity", "category_id", "words_count"]]
            print(f"Detailed Recommendations for user {user_id}:\n", recommendations_df)
        return recommendations_df

    def get_recommended_items(self, user_id, k):
        recommendations_df = self.recommend_items(user_id, topn=k)
        if recommendations_df.empty:
            return set()
        return set(recommendations_df['click_article_id'])


class CollaborativeFilteringRecommender:
    """
    Recommend top n articles based on collaborative filtering.
    """

    def __init__(self, interactions_df, items_df, sample_size=None):
        # Store the training interactions
        self.train_df = interactions_df  # Store training DataFrame
        self.items_df = items_df
        self.df = interactions_df.copy()  # For internal processing
        if sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42)
        self.prepare_data()
        self.create_mappings()
        self.create_sparse_matrix()

    def prepare_data(self):
        self.df = self.df[["user_id", "click_article_id"]]
        self.df["click_counts"] = 1
        self.df = self.df.groupby(["user_id", "click_article_id"])["click_counts"].sum().reset_index()
        self.df["click_counts"] = self.df["click_counts"].apply(lambda x: math.log(1 + x, 2))
        self.df["user_idx"] = self.df["user_id"].astype("category").cat.codes
        self.df["item_idx"] = self.df["click_article_id"].astype("category").cat.codes

    def create_mappings(self):
        self.user_id_map = dict(enumerate(self.df["user_id"].astype("category").cat.categories))
        self.item_id_map = dict(enumerate(self.df["click_article_id"].astype("category").cat.categories))
        self.user_map = {v: k for k, v in self.user_id_map.items()}
        self.item_map = {v: k for k, v in self.item_id_map.items()}

    def create_sparse_matrix(self):
        users = self.df["user_idx"].values
        items = self.df["item_idx"].values
        counts = self.df["click_counts"].values
        self.interaction_matrix = sparse.coo_matrix((counts, (users, items)))
        self.interaction_matrix = self.interaction_matrix.tocsr()

    def fit(self, factors=20, regularization=0.1, iterations=10, alpha=15):
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )
        data_conf = (self.interaction_matrix * alpha).astype("double")
        model.fit(data_conf)
        self.user_factors = model.user_factors
        self.item_factors = model.item_factors

    def recommend_items(self, user_id, num_items=10, verbose=False):
        user_idx = self.user_map.get(user_id)
        if user_idx is None:
            print(f"User ID {user_id} not found in user_map.")
            return set()
        user_vector = self.user_factors[user_idx]
        scores = user_vector.dot(self.item_factors.T)
        liked_items = self.interaction_matrix[user_idx].indices
        scores[liked_items] = -np.inf
        item_indices = np.argpartition(scores, -num_items)[-num_items:]
        best_items = item_indices[np.argsort(scores[item_indices])[::-1]]
        recommendations = [self.item_id_map.get(idx) for idx in best_items]
        recommended_items = set(recommendations)

        if verbose:
            recommendations_df = pd.DataFrame(recommendations, columns=["click_article_id"])
            recommendations_df['score'] = scores[best_items]
            recommendations_df = recommendations_df.merge(
                self.items_df,
                how="left",
                left_on="click_article_id",
                right_on="article_id"
            )
            recommendations_df = recommendations_df[["click_article_id", "score", "category_id", "words_count"]]
            print(f"Detailed Recommendations for user {user_id}:\n", recommendations_df)

        return recommended_items

    def get_recommended_items(self, user_id, k):
        recommended_items = self.recommend_items(user_id, num_items=k)
        return recommended_items

    def get_actual_items(self, user_id):
        """Return the set of actual items the user interacted with."""
        user_idx = self.user_map.get(user_id)
        if user_idx is None:
            print(f"User ID {user_id} not found in user_map.")
            return set()
        item_indices = self.interaction_matrix[user_idx].indices
        actual_items = set(self.item_id_map.get(idx) for idx in item_indices)
        return actual_items
    
# ------------------------------
# Main Script
# ------------------------------

def main():
    INPUT_DIR = "/Users/tgeof/Documents/Documents/B - Travaux Perso/1 - Scolarité/Ingenieur IA/OpenClassroom/_Projets/9. Réalisez une application de recommandation de contenu/1_data_projet/data/news-portal-user-interactions"

    # Load Data
    print("Loading clicks data...")
    df = load_clicks_data(INPUT_DIR)
    print("Loading items data...")
    items_df = load_items_data(INPUT_DIR)
    print("Loading embeddings...")
    embedding = load_embeddings(INPUT_DIR)

    # Debugging: Check data integrity
    print(">> Interactions DataFrame columns:\n", df.columns)
    print("\n>> First few rows of Interactions DataFrame:")
    print(df.head())

    print("\n>> Items DataFrame columns:\n", items_df.columns)
    print("\n>> First few rows of Items DataFrame:\n")
    print(items_df.head())

    print("\n>> Embedding shape:\n", embedding.shape)

    # Preprocess interactions
    interactions_full_df = preprocess_interactions(df)
    print("Preprocessed interactions.")

    # Split into training and test sets
    interactions_train_df, interactions_test_df = train_test_split(
        interactions_full_df,
        stratify=interactions_full_df["user_id"],
        test_size=0.2,
        random_state=42
    )
    print("Split data into training and test sets.")
    print(f"Training set size: {interactions_train_df.shape}")
    print(f"Test set size: {interactions_test_df.shape}")

    # Instantiate Content-Based Recommender with training data
    print("Instantiating Content-Based Recommender...")
    cb_recommender = ContentBasedRecommender(interactions_train_df, items_df, embedding)

    # Instantiate Collaborative Filtering Recommender with training data
    print("Instantiating Collaborative Filtering Recommender...")
    cf_recommender = CollaborativeFilteringRecommender(interactions_train_df, items_df, sample_size=None)

    # Verify embeddings
    verify_embeddings(cb_recommender, interactions_test_df)

    # Fit the Collaborative Filtering Recommender
    print("Fitting Collaborative Filtering Recommender...")
    cf_recommender.fit()
    print("Fitted Collaborative Filtering Recommender.")
    print("User factors shape:", cf_recommender.user_factors.shape)
    print("Item factors shape:", cf_recommender.item_factors.shape)

    # Debugging: Get recommendations for a user
    manual_inspection(cb_recommender, cf_recommender, interactions_test_df, num_users=5)

    # Define the list of k values to evaluate
    k_values = [1, 5, 10]

    # Initialize dictionaries to hold metrics
    cb_metrics_dict = {}
    cf_metrics_dict = {}

    # Evaluate for each k
    for k in k_values:
        print(f"\nEvaluating Content-Based Recommender for k={k}...")
        cb_metrics = evaluate_recommender(
            cb_recommender,
            cb_recommender.get_recommended_items,
            interactions_train_df,
            interactions_test_df,
            n_users=500,
            k=k
        )
        print(f"Content-Based Recommender Metrics for k={k}: {cb_metrics}")
        cb_metrics_dict[k] = cb_metrics

        print(f"\nEvaluating Collaborative Filtering Recommender for k={k}...")
        cf_metrics = evaluate_recommender(
            cf_recommender,
            cf_recommender.get_recommended_items,
            interactions_train_df,
            interactions_test_df,
            n_users=500,
            k=k
        )
        print(f"Collaborative Filtering Recommender Metrics for k={k}: {cf_metrics}")
        cf_metrics_dict[k] = cf_metrics

    # Plot metrics
    plot_metrics(cb_metrics_dict, 'Content-Based Recommender Metrics')
    plot_metrics(cf_metrics_dict, 'Collaborative Filtering Recommender Metrics')

    # Debugging
    zero_vectors = np.sum(cb_recommender.embedding == 0, axis=1)
    print(f"Number of zero vectors in embeddings: {np.sum(zero_vectors)}")

    # Debugging: check first 5 mappings
    for item_id in cb_recommender.item_ids[:5]:
        idx = cb_recommender.item_id_to_idx.get(item_id)
        if idx is not None:
            print(f"Item ID: {item_id}, Embedding Index: {idx}, Embedding Vector: {cb_recommender.embedding[idx][:5]}...")
        else:
            print(f"Item ID {item_id} not found in item_id_to_idx mapping.")

if __name__ == "__main__":
    main()