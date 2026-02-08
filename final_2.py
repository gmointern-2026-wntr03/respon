import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

# 警告の抑制
warnings.filterwarnings('ignore')

class SuzuriFastBenchmark:
    def __init__(self, log_df, product_df, creator_df, sale_df):
        self.log_df = log_df.copy()
        self.product_df = product_df.copy()
        self.creator_df = creator_df.copy()
        self.sale_df = sale_df.copy()
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.creator_encoder = LabelEncoder()
        
        self.results = {}

    # ---------------------------------------------------------------------
    # Phase 1: 前処理 & IDエンコーディング
    # ---------------------------------------------------------------------
    def preprocess(self):
        print("\n=== Phase 1: Preprocessing & Encoding ===")
        
        # 1. UTC統一
        self.log_df['accessed_at'] = pd.to_datetime(self.log_df['accessed_at'], utc=True, errors='coerce')
        self.creator_df['created_at'] = pd.to_datetime(self.creator_df['created_at'], utc=True, errors='coerce')
        self.sale_df['start_time'] = pd.to_datetime(self.sale_df['start_time'], utc=True, errors='coerce')
        self.sale_df['end_time'] = pd.to_datetime(self.sale_df['end_time'], utc=True, errors='coerce')
        self.log_df = self.log_df.dropna(subset=['accessed_at'])

        # 2. マスタ結合
        print("Merging Dataframes...")
        unique_products = self.product_df.drop_duplicates(subset='product_id')
        unique_creators = self.creator_df.drop_duplicates(subset='creator_id')
        
        # 3. IDの数値化
        all_users = pd.concat([self.log_df['user_id']]).unique()
        all_items = pd.concat([self.log_df['product_id'], unique_products['product_id']]).unique()
        all_creators = pd.concat([unique_creators['creator_id']]).unique()
        
        self.user_encoder.fit(all_users)
        self.item_encoder.fit(all_items)
        self.creator_encoder.fit(all_creators)
        
        # 変換
        self.log_df['user_idx'] = self.user_encoder.transform(self.log_df['user_id'])
        self.log_df['item_idx'] = self.item_encoder.transform(self.log_df['product_id'])
        
        unique_products['item_idx'] = self.item_encoder.transform(unique_products['product_id'])
        unique_products['creator_idx'] = self.creator_encoder.transform(unique_products['creator_id'])
        unique_creators['creator_idx'] = self.creator_encoder.transform(unique_creators['creator_id'])
        
        # 結合
        self.full_df = self.log_df.merge(unique_products, left_on='product_id', right_on='product_id', how='left', suffixes=('', '_prod'))
        self.full_df = self.full_df.merge(unique_creators, left_on='creator_id', right_on='creator_id', how='left', suffixes=('', '_creator'))
        self.full_df.sort_values('accessed_at', inplace=True)
        
        print(f"Total Unique Items: {len(all_items)}")

    # ---------------------------------------------------------------------
    # Phase 2: ノイズ除去 (厳密版)
    # ---------------------------------------------------------------------
    def clean_noise(self):
        print("\n=== Phase 2: Noise Cleaning ===")
        df = self.full_df.copy()
        
        # A. 自己購入
        is_self = (df['user_id'] == df['creator_id'])
        
        # B. 太客
        purchases = df[df['event_action'] == 'purchase']
        if not purchases.empty:
            c_total = purchases.groupby('creator_id').size()
            c_user = purchases.groupby(['creator_id', 'user_id']).size()
            dominance = (c_user / c_total.reindex(c_user.index.get_level_values(0)).values)
            sus_set = set(dominance[dominance >= 0.8].index.tolist())
            temp_pairs = list(zip(df['creator_id'], df['user_id']))
            is_dominant = [x in sus_set for x in temp_pairs]
        else:
            is_dominant = [False] * len(df)

        # C. 閲覧なし購入
        actions = df.groupby(['user_id', 'product_id'])['event_action'].apply(set)
        no_view_indices = actions[actions.apply(lambda x: 'purchase' in x and 'view' not in x)].index
        no_view_set = set(no_view_indices)
        temp_pairs = list(zip(df['user_id'], df['product_id']))
        is_direct = [x in no_view_set for x in temp_pairs]

        # フィルタリング
        mask = (~np.array(is_self)) & (~np.array(is_dominant)) & (~np.array(is_direct))
        self.clean_df = df[mask].copy()
        
        print(f"Cleaned Records: {len(self.clean_df)}")
        del df
        gc.collect()

    # ---------------------------------------------------------------------
    # Phase 3: 時系列分割 & 特徴量生成
    # ---------------------------------------------------------------------
    def split_and_engineer(self):
        print("\n=== Phase 3: Split & Feature Engineering ===")
        
        # 時系列分割 (Last 14 days as Test)
        split_date = self.clean_df['accessed_at'].max() - pd.Timedelta(days=14)
        self.train_logs = self.clean_df[self.clean_df['accessed_at'] < split_date].copy()
        self.test_logs = self.clean_df[self.clean_df['accessed_at'] >= split_date].copy()
        
        print(f"Train Logs: {len(self.train_logs)}")
        print(f"Test Logs:  {len(self.test_logs)}")

        # --- LightGBM用特徴量 (Trainのみから計算) ---
        purchases = self.train_logs[self.train_logs['event_action'] == 'purchase']
        
        # 人気度
        self.pop_map = purchases['item_idx'].value_counts().to_dict()
        
        # クリエイター特徴量
        def calculate_gini(x):
            if len(x) == 0: return 0
            array = np.sort(np.array(x, dtype=np.float64))
            index = np.arange(1, array.shape[0] + 1)
            n = array.shape[0]
            return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))
            
        gini_map = {}
        if not purchases.empty:
            gini_series = purchases.groupby('creator_idx')['user_idx'].apply(lambda x: calculate_gini(x.value_counts().values))
            gini_map = gini_series.to_dict()
            
        # 商品マスタ情報の整備
        mat_cols = [c for c in self.clean_df.columns if str(c).startswith('material_') and c != 'material_url']
        
        # 重複削除
        self.item_master = self.clean_df[['item_idx', 'price', 'creator_idx', 'created_at'] + mat_cols].drop_duplicates('item_idx').copy()
        
        # 特徴量付与
        self.item_master['material_complexity'] = self.item_master[mat_cols].notnull().sum(axis=1)
        self.item_master['creator_gini_index'] = self.item_master['creator_idx'].map(gini_map).fillna(0)
        self.item_master['popularity_score'] = self.item_master['item_idx'].map(self.pop_map).fillna(0)
        self.item_master['price'] = self.item_master['price'].fillna(0)
        
        # 使用する特徴量
        self.lgbm_cols = ['material_complexity', 'creator_gini_index', 'popularity_score', 'price']

    # ---------------------------------------------------------------------
    # Phase 4: LightGBM学習
    # ---------------------------------------------------------------------
    def train_lightgbm(self):
        print("\n=== Phase 4: Training LightGBM ===")
        
        # 負例サンプリング
        pos = self.train_logs[self.train_logs['event_action'] == 'purchase'].copy()
        pos['target'] = 1
        neg = self.train_logs[self.train_logs['event_action'] == 'view'].sample(n=len(pos)*5, random_state=42).copy()
        neg['target'] = 0
        
        # ★修正: カラム衝突回避のため、IDのみ抽出して結合
        train_base = pd.concat([
            pos[['user_idx', 'item_idx', 'target']], 
            neg[['user_idx', 'item_idx', 'target']]
        ], ignore_index=True)
        
        # 特徴量結合
        train_df = train_base.merge(self.item_master[['item_idx'] + self.lgbm_cols], on='item_idx', how='left')
        
        lgb_train = lgb.Dataset(train_df[self.lgbm_cols], train_df['target'])
        params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'learning_rate': 0.1}
        
        print("Training started...")
        self.lgbm_model = lgb.train(params, lgb_train, num_boost_round=100)
        print("LightGBM Trained.")

    # ---------------------------------------------------------------------
    # Phase 5: 全アイテムに対する評価 (Global Ranking)
    # ---------------------------------------------------------------------
    def evaluate_all_items(self):
        print("\n=== Phase 5: Global Ranking Evaluation (Recall@10 on ALL Items) ===")
        
        test_purchases = self.test_logs[self.test_logs['event_action'] == 'purchase']
        target_users = test_purchases['user_idx'].unique()
        
        # 計算負荷のため50人サンプリング
        if len(target_users) > 50:
            eval_users = np.random.choice(target_users, 50, replace=False)
            print(f"Sampling 50 users from {len(target_users)} active test users.")
        else:
            eval_users = target_users
            print(f"Evaluating all {len(target_users)} active test users.")
            
        all_item_indices = self.item_master['item_idx'].values
        results = {'Random': [], 'Popularity': [], 'LightGBM': []}
        
        # LightGBMのスコア事前計算 (Global Score)
        print("Pre-calculating LightGBM scores for all items...")
        lgb_global_scores = self.lgbm_model.predict(self.item_master[self.lgbm_cols])
        # スコア順に並んだアイテムIDのリストを作成
        lgb_sorted_args = lgb_global_scores.argsort()[::-1]
        lgb_top_items = self.item_master.iloc[lgb_sorted_args[:10]]['item_idx'].values
        
        # PopularityのTop10も固定
        pop_recs = sorted(self.pop_map, key=self.pop_map.get, reverse=True)[:10]

        # --- 評価ループ ---
        for i, uid in enumerate(eval_users):
            if i % 10 == 0: print(f"Processing user {i+1}...")
            
            # 正解
            true_items = test_purchases[test_purchases['user_idx'] == uid]['item_idx'].values
            if len(true_items) == 0: continue
            
            # 1. Random
            rand_recs = np.random.choice(all_item_indices, 10, replace=False)
            results['Random'].append(1 if len(set(true_items) & set(rand_recs)) > 0 else 0)
            
            # 2. Popularity
            results['Popularity'].append(1 if len(set(true_items) & set(pop_recs)) > 0 else 0)
            
            # 3. LightGBM (Global Optimization)
            results['LightGBM'].append(1 if len(set(true_items) & set(lgb_top_items)) > 0 else 0)

        # 結果集計
        print("\n=== FINAL RESULTS (Global Recall@10) ===")
        print(f"Evaluated on {len(eval_users)} users against {len(all_item_indices)} items.")
        
        final_res = {}
        for model, scores in results.items():
            final_res[model] = np.mean(scores)
            
        df_res = pd.DataFrame(list(final_res.items()), columns=['Model', 'Recall@10'])
        print(df_res)

    def run(self):
        self.preprocess()
        self.clean_noise()
        self.split_and_engineer()
        self.train_lightgbm()
        self.evaluate_all_items()

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
        
        pipeline = SuzuriFastBenchmark(log, prod, creat, sale)
        pipeline.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
