import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

# 警告の抑制
warnings.filterwarnings('ignore')

class SuzuriHighAccuracyBenchmark:
    def __init__(self, log_df, product_df, creator_df, sale_df):
        self.log_df = log_df.copy()
        self.product_df = product_df.copy()
        self.creator_df = creator_df.copy()
        self.sale_df = sale_df.copy()
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        self.results = {}

    # ---------------------------------------------------------------------
    # Phase 1: 前処理 & ノイズ除去
    # ---------------------------------------------------------------------
    def preprocess_and_clean(self):
        print("\n=== Phase 1: Preprocessing & Cleaning ===")
        # UTC統一
        self.log_df['accessed_at'] = pd.to_datetime(self.log_df['accessed_at'], utc=True, errors='coerce')
        self.product_df = self.product_df.drop_duplicates(subset='product_id')
        self.creator_df = self.creator_df.drop_duplicates(subset='creator_id')
        self.log_df = self.log_df.dropna(subset=['accessed_at'])

        # マスタ結合
        print("Merging Dataframes...")
        self.full_df = self.log_df.merge(self.product_df, on='product_id', how='left')
        self.full_df = self.full_df.merge(self.creator_df, on='creator_id', how='left')
        self.full_df.sort_values('accessed_at', inplace=True)

        # ノイズ除去 (簡易版: 速度重視)
        print("Cleaning Noise...")
        # 自己購入
        mask_self = (self.full_df['user_id'] != self.full_df['creator_id'])
        self.clean_df = self.full_df[mask_self].copy()
        
        # IDエンコーディング
        all_users = self.clean_df['user_id'].unique()
        all_items = self.product_df['product_id'].unique() # 全商品
        
        self.user_encoder.fit(all_users)
        self.item_encoder.fit(all_items)
        
        # LogデータのID変換
        self.clean_df = self.clean_df[self.clean_df['user_id'].isin(all_users)]
        self.clean_df = self.clean_df[self.clean_df['product_id'].isin(all_items)]
        
        self.clean_df['user_idx'] = self.user_encoder.transform(self.clean_df['user_id'])
        self.clean_df['item_idx'] = self.item_encoder.transform(self.clean_df['product_id'])
        
        print(f"Clean Records: {len(self.clean_df)}")

    # ---------------------------------------------------------------------
    # Phase 2: 時系列分割 & 特徴量計算 (Preferences)
    # ---------------------------------------------------------------------
    def split_and_engineer(self):
        print("\n=== Phase 2: Feature Engineering (User Preferences) ===")
        
        # 時系列分割 (Last 14 days Test)
        split_date = self.clean_df['accessed_at'].max() - pd.Timedelta(days=14)
        self.train_logs = self.clean_df[self.clean_df['accessed_at'] < split_date].copy()
        self.test_logs = self.clean_df[self.clean_df['accessed_at'] >= split_date].copy()
        
        print(f"Train: {len(self.train_logs)}, Test: {len(self.test_logs)}")

        # --- 1. アイテム特徴量 (Item Profile) ---
        # item_category_name を数値化
        self.cat_encoder = LabelEncoder()
        # 欠損埋め
        self.product_df['item_category_name'] = self.product_df['item_category_name'].fillna('Unknown')
        self.product_df['item_cat_idx'] = self.cat_encoder.fit_transform(self.product_df['item_category_name'])
        
        # item_idx を付与
        self.product_df = self.product_df[self.product_df['product_id'].isin(self.item_encoder.classes_)]
        self.product_df['item_idx'] = self.item_encoder.transform(self.product_df['product_id'])
        
        # アイテム辞書 {item_idx: cat_idx}
        self.item_cat_map = self.product_df.set_index('item_idx')['item_cat_idx'].to_dict()
        self.item_price_map = self.product_df.set_index('item_idx')['price'].fillna(0).to_dict()
        
        # アイテム人気度 (Train期間)
        train_purchases = self.train_logs[self.train_logs['event_action'] == 'purchase']
        self.pop_map = train_purchases['item_idx'].value_counts().to_dict()
        
        # --- 2. ユーザー特徴量 (User Profile) ---
        # ユーザーが「最もよく買うカテゴリ」と「平均購入価格」を計算
        print("Calculating User Profiles...")
        
        # カテゴリ結合
        train_purchases = train_purchases.merge(self.product_df[['product_id', 'item_cat_idx']], on='product_id', how='left')
        
        # ユーザーごとの好みのカテゴリ (Mode)
        user_fav_cat = train_purchases.groupby('user_idx')['item_cat_idx'].agg(lambda x: x.mode()[0] if len(x) > 0 else -1)
        self.user_fav_cat_map = user_fav_cat.to_dict()
        
        # ユーザーごとの平均購入価格
        train_purchases['price'] = train_purchases['product_id'].map(lambda x: self.item_price_map.get(self.item_encoder.transform([x])[0], 0))
        user_avg_price = train_purchases.groupby('user_idx')['price'].mean()
        self.user_avg_price_map = user_avg_price.to_dict()
        
        # カテゴリごとの人気アイテムリスト (候補生成用)
        # {cat_idx: [item_id, item_id, ... (Top 50)]}
        self.cat_top_items = {}
        # Train期間の人気順
        cat_pop = train_purchases.groupby(['item_cat_idx', 'item_idx']).size().reset_index(name='count')
        cat_pop = cat_pop.sort_values(['item_cat_idx', 'count'], ascending=[True, False])
        
        for cat, group in cat_pop.groupby('item_cat_idx'):
            self.cat_top_items[cat] = group['item_idx'].head(50).values

        # 全体の人気アイテム (バックアップ用)
        self.global_top_items = train_purchases['item_idx'].value_counts().head(50).index.values

    # ---------------------------------------------------------------------
    # Phase 3: 学習データ作成 (Matching Features)
    # ---------------------------------------------------------------------
    def create_train_dataset(self):
        print("\n=== Phase 3: Creating Training Dataset ===")
        
        # 正例
        pos = self.train_logs[self.train_logs['event_action'] == 'purchase'].copy()
        pos['target'] = 1
        # 負例 (Viewのみ)
        neg = self.train_logs[self.train_logs['event_action'] == 'view'].sample(n=len(pos)*5, random_state=42).copy()
        neg['target'] = 0
        
        train_df = pd.concat([pos[['user_idx', 'item_idx', 'target']], neg[['user_idx', 'item_idx', 'target']]], ignore_index=True)
        
        # 特徴量付与
        self.add_features(train_df)
        self.lgbm_train_df = train_df
        
    def add_features(self, df):
        # 1. Item Features
        df['popularity'] = df['item_idx'].map(self.pop_map).fillna(0)
        df['price'] = df['item_idx'].map(self.item_price_map).fillna(0)
        df['cat_idx'] = df['item_idx'].map(self.item_cat_map).fillna(-1)
        
        # 2. User Features
        df['user_fav_cat'] = df['user_idx'].map(self.user_fav_cat_map).fillna(-1)
        df['user_avg_price'] = df['user_idx'].map(self.user_avg_price_map).fillna(3000) # デフォルト3000円
        
        # 3. Interaction Features (ここが重要！)
        # カテゴリ一致フラグ
        df['is_cat_match'] = (df['cat_idx'] == df['user_fav_cat']).astype(int)
        
        # 価格差 (比率)
        # 0除算回避
        df['price_ratio'] = df['price'] / (df['user_avg_price'] + 1)
        # 乖離度: 1に近いほど良い -> abs(1 - ratio) が小さいほど良い
        df['price_diff_score'] = np.abs(1 - df['price_ratio'])

        self.features = ['popularity', 'price', 'is_cat_match', 'price_ratio', 'price_diff_score']

    # ---------------------------------------------------------------------
    # Phase 4: LightGBM学習
    # ---------------------------------------------------------------------
    def train_model(self):
        print("\n=== Phase 4: Training LightGBM ===")
        lgb_train = lgb.Dataset(self.lgbm_train_df[self.features], self.lgbm_train_df['target'])
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'learning_rate': 0.1,
            'num_leaves': 31
        }
        self.model = lgb.train(params, lgb_train, num_boost_round=100)
        
        # 重要度表示
        imp = pd.DataFrame({
            'Feature': self.features,
            'Gain': self.model.feature_importance(importance_type='gain')
        }).sort_values('Gain', ascending=False)
        print("Feature Importance:\n", imp)

    # ---------------------------------------------------------------------
    # Phase 5: 候補生成 & 推論 (Evaluation)
    # ---------------------------------------------------------------------
    def evaluate(self):
        print("\n=== Phase 5: Evaluation (Candidate Generation + Re-ranking) ===")
        
        test_purchases = self.test_logs[self.test_logs['event_action'] == 'purchase']
        target_users = test_purchases['user_idx'].unique()
        
        # 50人サンプリング
        if len(target_users) > 50:
            eval_users = np.random.choice(target_users, 50, replace=False)
        else:
            eval_users = target_users
            
        print(f"Evaluating {len(eval_users)} users...")
        
        results = {
            'Random': [],
            'Popularity': [],
            'LightGBM_Rerank': [] # これが本命
        }
        
        all_items = list(self.item_price_map.keys())
        global_top_10 = self.global_top_items[:10] # Popularity Top 10
        
        for i, uid in enumerate(eval_users):
            if i % 10 == 0: print(f"Processing user {i+1}...")
            
            true_items = set(test_purchases[test_purchases['user_idx'] == uid]['item_idx'].values)
            if not true_items: continue
            
            # 1. Random (Baseline)
            rand_recs = np.random.choice(all_items, 10, replace=False)
            results['Random'].append(1 if len(true_items & set(rand_recs)) > 0 else 0)
            
            # 2. Popularity (Baseline)
            results['Popularity'].append(1 if len(true_items & set(global_top_10)) > 0 else 0)
            
            # 3. LightGBM (Candidate Generation -> Rerank)
            # 候補生成: ユーザーの好きなカテゴリの人気商品 + 全体の人気商品
            fav_cat = self.user_fav_cat_map.get(uid, -1)
            
            candidates = set(self.global_top_items) # まず全体のTop50
            if fav_cat != -1 and fav_cat in self.cat_top_items:
                candidates.update(self.cat_top_items[fav_cat]) # カテゴリTop50を追加
            
            candidate_list = list(candidates)
            
            # 推論用データフレーム作成
            infer_df = pd.DataFrame({'user_idx': [uid]*len(candidate_list), 'item_idx': candidate_list})
            self.add_features(infer_df) # 特徴量計算 (Match系含む)
            
            # スコアリング
            scores = self.model.predict(infer_df[self.features])
            top_indices = scores.argsort()[::-1][:10]
            lgb_recs = [candidate_list[i] for i in top_indices]
            
            results['LightGBM_Rerank'].append(1 if len(true_items & set(lgb_recs)) > 0 else 0)

        # 集計
        print("\n=== FINAL RESULTS (Global Recall@10) ===")
        final_res = {}
        for model, scores in results.items():
            final_res[model] = np.mean(scores)
        
        print(pd.DataFrame(list(final_res.items()), columns=['Model', 'Recall@10']))

    def run(self):
        self.preprocess_and_clean()
        self.split_and_engineer()
        self.create_train_dataset()
        self.train_model()
        self.evaluate()

if __name__ == "__main__":
    PRODUCT_FILE = 'products_20260204.csv'
    EVENT_FILE   = 'events_20260204.csv'
    CREATOR_FILE = 'creators_20260304.csv'
    SALE_FILE    = 'time_discounts_2025.csv'

    try:
        print("Loading files...")
        prod = pd.read_csv(PRODUCT_FILE)
        log = pd.read_csv(EVENT_FILE)
        creat = pd.read_csv(CREATOR_FILE)
        sale = pd.read_csv(SALE_FILE)
        
        pipeline = SuzuriHighAccuracyBenchmark(log, prod, creat, sale)
        pipeline.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
