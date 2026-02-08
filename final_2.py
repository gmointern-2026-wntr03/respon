import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings
import gc

# 警告の抑制
warnings.filterwarnings('ignore')

class SuzuriStrictBenchmark:
    def __init__(self, log_df, product_df, creator_df, sale_df):
        self.log_df = log_df.copy()
        self.product_df = product_df.copy()
        self.creator_df = creator_df.copy()
        self.sale_df = sale_df.copy()
        
        # データ格納用
        self.train_logs = None
        self.test_logs = None
        self.train_dataset = None
        self.test_dataset = None
        
        # 結果比較用
        self.results = {}

    # ---------------------------------------------------------------------
    # Phase 1: 前処理 & ノイズ除去 (Preprocessing & Cleaning)
    # ---------------------------------------------------------------------
    def preprocess_and_clean(self):
        print("\n=== Phase 1: Preprocessing & Noise Cleaning ===")
        
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
        
        full_df = self.log_df.merge(unique_products, on='product_id', how='left', suffixes=('', '_prod'))
        full_df = full_df.merge(unique_creators, on='creator_id', how='left', suffixes=('', '_creator'))
        full_df.sort_values(['user_id', 'accessed_at'], inplace=True)

        # 3. ノイズ除去 (厳密版)
        initial_len = len(full_df)
        
        # A. 自己購入
        is_self = (full_df['user_id'] == full_df['creator_id'])
        if 'name' in full_df.columns and 'name_creator' in full_df.columns:
            name_match = (full_df['name'].fillna('') == full_df['name_creator'].fillna('')) & (full_df['name'].notnull())
            is_self = is_self | name_match
        
        # B. 太客 (Dominant Buyer)
        purchases = full_df[full_df['event_action'] == 'purchase']
        if not purchases.empty:
            c_total = purchases.groupby('creator_id').size()
            c_user = purchases.groupby(['creator_id', 'user_id']).size()
            dominance = (c_user / c_total.reindex(c_user.index.get_level_values(0)).values)
            sus_set = set(dominance[dominance >= 0.8].index.tolist())
            temp_pairs = list(zip(full_df['creator_id'], full_df['user_id']))
            is_dominant = [x in sus_set for x in temp_pairs]
        else:
            is_dominant = [False] * len(full_df)

        # C. 閲覧なし購入 (Direct Buy)
        actions = full_df.groupby(['user_id', 'product_id'])['event_action'].apply(set)
        no_view_indices = actions[actions.apply(lambda x: 'purchase' in x and 'view' not in x)].index
        no_view_set = set(no_view_indices)
        temp_pairs = list(zip(full_df['user_id'], full_df['product_id']))
        is_direct = [x in no_view_set for x in temp_pairs]

        # フィルタリング適用
        mask = (~np.array(is_self)) & (~np.array(is_dominant)) & (~np.array(is_direct))
        self.clean_df = full_df[mask].copy()
        
        print(f"Removed Noise Records: {initial_len - len(self.clean_df)}")
        print(f"Remaining Clean Records: {len(self.clean_df)}")
        
        del full_df
        gc.collect()

    # ---------------------------------------------------------------------
    # Phase 2: 厳密な時系列分割 (Strict Split)
    # ---------------------------------------------------------------------
    def split_data(self):
        print("\n=== Phase 2: Strict Time-Series Split ===")
        # データの最後の20%の日数をテスト期間とする
        min_date = self.clean_df['accessed_at'].min()
        max_date = self.clean_df['accessed_at'].max()
        total_duration = max_date - min_date
        split_date = max_date - (total_duration * 0.2)
        
        print(f"Data Range: {min_date} to {max_date}")
        print(f"Split Date: {split_date}")
        
        self.train_logs = self.clean_df[self.clean_df['accessed_at'] < split_date].copy()
        self.test_logs = self.clean_df[self.clean_df['accessed_at'] >= split_date].copy()
        
        print(f"Train Logs: {len(self.train_logs)}")
        print(f"Test Logs:  {len(self.test_logs)}")

    # ---------------------------------------------------------------------
    # Phase 3: 特徴量エンジニアリング (Leakage Free)
    # ---------------------------------------------------------------------
    def engineer_features(self):
        print("\n=== Phase 3: Feature Engineering (No Leakage) ===")
        
        # ★重要: 全ての特徴量は「Trainログ」だけを元に計算する
        
        # 1. クリエイター特徴量 (Gini, Tenure)
        purchases = self.train_logs[self.train_logs['event_action'] == 'purchase']
        
        def calculate_gini(array):
            array = np.array(array, dtype=np.float64)
            if np.sum(array) == 0: return 0
            array += 1e-9
            array = np.sort(array)
            index = np.arange(1, array.shape[0] + 1)
            n = array.shape[0]
            return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

        gini_map = {}
        if not purchases.empty:
            gini_series = purchases.groupby('creator_id')['user_id'].apply(lambda x: calculate_gini(x.value_counts().values))
            gini_map = gini_series.to_dict()

        # 2. 商品人気度 (Popularity Baseline用)
        pop_map = purchases['product_id'].value_counts().to_dict()

        # 3. ユーザー購買率 (User Buy Rate)
        u_stats = self.train_logs.groupby('user_id')['event_action'].value_counts().unstack(fill_value=0)
        user_buy_map = {}
        if 'purchase' in u_stats.columns:
            total = u_stats.sum(axis=1).replace(0, 1)
            user_buy_map = (u_stats['purchase'] / total).to_dict()

        # --- 特徴量適用関数 ---
        def apply_features(df):
            # 静的特徴量
            mat_cols = [c for c in df.columns if str(c).startswith('material_') and c != 'material_url']
            df['material_complexity'] = df[mat_cols].notnull().sum(axis=1)
            df['price'] = df['price'].fillna(0)
            
            # 動的特徴量 (Trainの統計情報をMap)
            df['creator_gini_index'] = df['creator_id'].map(gini_map).fillna(0) # 知らない人は0
            df['popularity_score'] = df['product_id'].map(pop_map).fillna(0)    # 知らない商品は0
            df['user_buy_rate'] = df['user_id'].map(user_buy_map).fillna(0)     # 新規ユーザーは0
            df['creator_tenure_days'] = (df['accessed_at'] - df['created_at']).dt.days
            
            # セール情報 (これはカレンダー通りの事実なのでリークではない)
            df['is_sale_target'] = 0
            df['days_to_sale_end'] = 999.0
            if not self.sale_df.empty:
                for _, sale in self.sale_df.iterrows():
                    t_mask = (df['accessed_at'] >= sale['start_time']) & (df['accessed_at'] <= sale['end_time'])
                    if 'item_category_name' in df.columns and 'item' in sale:
                        mask = t_mask & (df['item_category_name'] == sale['item'])
                    else:
                        mask = t_mask
                    if mask.any():
                        df.loc[mask, 'is_sale_target'] = 1
                        df.loc[mask, 'days_to_sale_end'] = (sale['end_time'] - df.loc[mask, 'accessed_at']).dt.total_seconds() / 86400
            
            return df

        print("Applying features to Train & Test...")
        self.train_dataset = apply_features(self.train_logs.copy())
        self.test_dataset = apply_features(self.test_logs.copy())
        
        self.feature_cols = [
            'material_complexity', 'creator_gini_index', 'creator_tenure_days',
            'days_to_sale_end', 'is_sale_target', 'user_buy_rate', 'price',
            'popularity_score' # LightGBMにも人気度を教えてあげる
        ]

    # ---------------------------------------------------------------------
    # Phase 4: データセット構築 (Hard Negative Mining)
    # ---------------------------------------------------------------------
    def create_ranking_datasets(self):
        print("\n=== Phase 4: Dataset Construction (Hard Negatives) ===")
        
        # --- Trainデータ作成 ---
        # 正例: Purchase
        pos = self.train_dataset[self.train_dataset['event_action'] == 'purchase'].copy()
        pos['target'] = 1
        
        # 負例: Viewのみ (購入済みペアを除外)
        purchase_pairs = set(zip(pos['user_id'], pos['product_id']))
        views = self.train_dataset[self.train_dataset['event_action'] == 'view'].copy()
        view_pairs = list(zip(views['user_id'], views['product_id']))
        is_bought = [p in purchase_pairs for p in view_pairs]
        
        neg = views[~np.array(is_bought)].copy()
        neg['target'] = 0
        
        # ダウンサンプリング
        if len(neg) > len(pos) * 5:
            neg = neg.sample(n=len(pos)*5, random_state=42)
            
        self.lgbm_train_df = pd.concat([pos, neg], ignore_index=True)
        print(f"LGBM Train Samples: {len(self.lgbm_train_df)}")

        # --- Testデータ作成 (Ranking Evaluation用) ---
        # Test期間に購入があるユーザーのみを評価対象にする
        test_purchases = self.test_dataset[self.test_dataset['event_action'] == 'purchase']
        valid_users = test_purchases['user_id'].unique()
        
        # ターゲット設定: Purchase=1, View=0
        # (ここではシンプルに「購入したものを上位に出せるか」を見る)
        self.eval_df = self.test_dataset[self.test_dataset['user_id'].isin(valid_users)].copy()
        self.eval_df['target'] = (self.eval_df['event_action'] == 'purchase').astype(int)
        
        print(f"Evaluation Candidates: {len(self.eval_df)}")
        print(f"Target Users: {len(valid_users)}")

    # ---------------------------------------------------------------------
    # Phase 5: モデル学習 & 比較評価 (Comparison)
    # ---------------------------------------------------------------------
    def train_and_compare(self):
        print("\n=== Phase 5: Training & Model Comparison ===")
        
        # 1. Random Baseline
        print("[1/3] Evaluating Random Model...")
        self.eval_df['score_random'] = np.random.rand(len(self.eval_df))
        self.evaluate_model('Random', 'score_random')
        
        # 2. Popularity Baseline
        print("[2/3] Evaluating Popularity Model...")
        # 特徴量エンジニアリングで作った 'popularity_score' (Train期間の人気度) をそのまま使う
        self.eval_df['score_pop'] = self.eval_df['popularity_score']
        self.evaluate_model('Popularity', 'score_pop')
        
        # 3. LightGBM
        print("[3/3] Training & Evaluating LightGBM...")
        X_train = self.lgbm_train_df[self.feature_cols]
        y_train = self.lgbm_train_df['target']
        
        lgb_train = lgb.Dataset(X_train, y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_seed': 42
        }
        
        model = lgb.train(params, lgb_train, num_boost_round=500)
        
        self.eval_df['score_lgbm'] = model.predict(self.eval_df[self.feature_cols])
        self.evaluate_model('LightGBM', 'score_lgbm')
        
        # 特徴量重要度表示
        imp = pd.DataFrame({
            'Feature': self.feature_cols,
            'Gain': model.feature_importance(importance_type='gain')
        }).sort_values('Gain', ascending=False)
        print("\nLightGBM Feature Importance:\n", imp)

    # ---------------------------------------------------------------------
    # 共通評価ロジック
    # ---------------------------------------------------------------------
    def evaluate_model(self, name, col):
        recall_10_list = []
        mrr_list = []
        
        grouped = self.eval_df.groupby('user_id')
        
        for uid, group in grouped:
            if group['target'].sum() == 0: continue
            
            # スコア順にソート
            sorted_group = group.sort_values(col, ascending=False)
            targets = sorted_group['target'].values
            
            # Recall@10
            recall_10_list.append(1 if targets[:10].sum() > 0 else 0)
            
            # MRR
            try:
                rank = np.where(targets == 1)[0][0] + 1
                mrr_list.append(1.0 / rank)
            except IndexError:
                mrr_list.append(0)
        
        r10 = np.mean(recall_10_list)
        mrr = np.mean(mrr_list)
        
        self.results[name] = {'Recall@10': r10, 'MRR': mrr}
        print(f"  > {name}: Recall@10 = {r10:.4f}, MRR = {mrr:.4f}")

    def run(self):
        self.preprocess_and_clean()
        self.split_data()
        self.engineer_features()
        self.create_ranking_datasets()
        self.train_and_compare()
        
        print("\n=== FINAL RESULTS ===")
        print(pd.DataFrame(self.results).T)

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
        
        pipeline = SuzuriStrictBenchmark(log, prod, creat, sale)
        pipeline.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
