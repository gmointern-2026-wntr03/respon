import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
import warnings
import gc

# 警告の抑制
warnings.filterwarnings('ignore')

class SuzuriRankPipeline:
    def __init__(self, log_df, product_df, creator_df, sale_df):
        self.log_df = log_df.copy()
        self.product_df = product_df.copy()
        self.creator_df = creator_df.copy()
        self.sale_df = sale_df.copy()
        
        self.full_df = None
        self.train_df = None
        self.test_df = None  # ランキング評価用
        self.model = None
        self.features = []

    # ---------------------------------------------------------------------
    # Phase 1: 前処理 (UTC統一・マスタ結合)
    # ---------------------------------------------------------------------
    def preprocess(self):
        print("\n=== Phase 1: Preprocessing ===")
        # 日時変換
        self.log_df['accessed_at'] = pd.to_datetime(self.log_df['accessed_at'], utc=True, errors='coerce')
        self.creator_df['created_at'] = pd.to_datetime(self.creator_df['created_at'], utc=True, errors='coerce')
        self.sale_df['start_time'] = pd.to_datetime(self.sale_df['start_time'], utc=True, errors='coerce')
        self.sale_df['end_time'] = pd.to_datetime(self.sale_df['end_time'], utc=True, errors='coerce')
        self.log_df = self.log_df.dropna(subset=['accessed_at'])

        # マスタ結合
        print("Merging Dataframes...")
        unique_products = self.product_df.drop_duplicates(subset='product_id')
        unique_creators = self.creator_df.drop_duplicates(subset='creator_id')
        
        self.full_df = self.log_df.merge(unique_products, on='product_id', how='left', suffixes=('', '_prod'))
        self.full_df = self.full_df.merge(unique_creators, on='creator_id', how='left', suffixes=('', '_creator'))
        self.full_df.sort_values(['user_id', 'accessed_at'], inplace=True)

    # ---------------------------------------------------------------------
    # Phase 2: ノイズ除去
    # ---------------------------------------------------------------------
    def clean_noise(self):
        print("\n=== Phase 2: Noise Cleaning ===")
        df = self.full_df

        # A. 自己購入
        is_self = (df['user_id'] == df['creator_id'])
        if 'name' in df.columns and 'name_creator' in df.columns:
            is_self = is_self | (df['name'].fillna('') == df['name_creator'].fillna(''))
        
        # B. 太客 (Dominant Buyer)
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

        # C. 閲覧なし購入 (Direct Buy)
        # 今回は「ViewとPurchaseの差」を見るため、Direct Buyは厳密に除外
        actions = df.groupby(['user_id', 'product_id'])['event_action'].apply(set)
        no_view_indices = actions[actions.apply(lambda x: 'purchase' in x and 'view' not in x)].index
        no_view_set = set(no_view_indices)
        temp_pairs = list(zip(df['user_id'], df['product_id']))
        is_direct = [x in no_view_set for x in temp_pairs]

        # フィルタリング
        clean_condition = (~is_self) & (~np.array(is_dominant)) & (~np.array(is_direct))
        self.full_df = df[clean_condition].copy()
        
        print(f"Cleaned Records: {len(self.full_df)}")
        del df
        gc.collect()

    # ---------------------------------------------------------------------
    # Phase 3: 特徴量エンジニアリング
    # ---------------------------------------------------------------------
    def engineer_features(self):
        print("\n=== Phase 3: Feature Engineering ===")
        df = self.full_df

        # デザイン複雑度
        mat_cols = [c for c in df.columns if str(c).startswith('material_') and c != 'material_url']
        df['material_complexity'] = df[mat_cols].notnull().sum(axis=1)

        # クリエイター特徴量
        purchases = df[df['event_action'] == 'purchase']
        def calculate_gini(array):
            array = np.array(array, dtype=np.float64)
            if np.sum(array) == 0: return 0
            array += 1e-9
            array = np.sort(array)
            index = np.arange(1, array.shape[0] + 1)
            n = array.shape[0]
            return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

        if not purchases.empty:
            gini_series = purchases.groupby('creator_id')['user_id'].apply(lambda x: calculate_gini(x.value_counts().values))
            df['creator_gini_index'] = df['creator_id'].map(gini_series).fillna(0)
        else:
            df['creator_gini_index'] = 0
        
        df['creator_tenure_days'] = (df['accessed_at'] - df['created_at']).dt.days

        # セール特徴量
        df['days_to_sale_end'] = 999.0
        df['is_sale_target'] = 0
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

        # ユーザー行動特徴量
        u_stats = df.groupby('user_id')['event_action'].value_counts().unstack(fill_value=0)
        if 'purchase' in u_stats.columns:
            total_actions = u_stats.sum(axis=1).replace(0, 1)
            df['user_buy_rate'] = df['user_id'].map(u_stats['purchase'] / total_actions).fillna(0)
        else:
            df['user_buy_rate'] = 0

        self.full_df = df

    # ---------------------------------------------------------------------
    # Phase 4: データセット構築 (Hard Negative: View but No Purchase)
    # ---------------------------------------------------------------------
    def create_dataset(self):
        print("\n=== Phase 4: Dataset Construction (Hard Negatives) ===")
        
        # 1. 正例 (Purchase)
        positives = self.full_df[self.full_df['event_action'] == 'purchase'].copy()
        positives['target'] = 1
        
        # 2. 負例 (Hard Negative: View Only)
        # 購入に至らなかったViewイベントを抽出
        # まず、(user, product) のペアで「購入あり」のものを特定
        purchase_pairs = set(zip(positives['user_id'], positives['product_id']))
        
        views = self.full_df[self.full_df['event_action'] == 'view'].copy()
        
        # applyだと遅いので、merge (anti-join) で「購入済みペア」を除外
        # 一時的にフラグを立てる
        positives_keys = positives[['user_id', 'product_id']].drop_duplicates()
        positives_keys['is_bought'] = True
        
        views = views.merge(positives_keys, on=['user_id', 'product_id'], how='left')
        
        # is_bought が NaN のもの = 見たけど買わなかったもの
        negatives = views[views['is_bought'].isna()].copy()
        negatives['target'] = 0
        negatives.drop('is_bought', axis=1, inplace=True)
        
        # データバランスの調整（負例が多すぎる場合、正例の10倍程度にダウンサンプリング）
        if len(negatives) > len(positives) * 10:
            print(f"Downsampling negatives from {len(negatives)} to {len(positives)*10}...")
            negatives = negatives.sample(n=len(positives) * 10, random_state=42)
            
        print(f"Positive Samples (Buy): {len(positives)}")
        print(f"Negative Samples (View only): {len(negatives)}")
        
        # 結合
        dataset = pd.concat([positives, negatives], ignore_index=True)
        dataset.sort_values('accessed_at', inplace=True)
        
        # 特徴量の選定
        candidates = [
            'material_complexity', 'creator_gini_index', 'creator_tenure_days',
            'days_to_sale_end', 'is_sale_target', 'user_buy_rate', 'price'
        ]
        self.features = [c for c in candidates if c in dataset.columns]
        
        # 3. 時系列分割 (Train / Test)
        # ここで単純なランダム分割ではなく、ユーザーごとの未来予測にするため時間を基準にする
        # 全期間の後半20%をテストにする
        split_point = int(len(dataset) * 0.8)
        self.train_df = dataset.iloc[:split_point]
        self.test_df = dataset.iloc[split_point:]
        
        print(f"Train Size: {len(self.train_df)}")
        print(f"Test Size:  {len(self.test_df)}")

    # ---------------------------------------------------------------------
    # Phase 5: 学習 & ランキング評価 (Ranking Evaluation)
    # ---------------------------------------------------------------------
    def train_and_evaluate(self):
        print("\n=== Phase 5: Training & Ranking Evaluation ===")
        
        # --- LightGBM 学習 ---
        X_train = self.train_df[self.features]
        y_train = self.train_df['target']
        # 検証用データとしてテストデータの20%を使う（ランキング評価とは別）
        X_val = self.test_df[self.features]
        y_val = self.test_df['target']
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc', # 学習時はAUCで最適化
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
        
        self.model = lgb.train(
            params, lgb_train, valid_sets=[lgb_train, lgb_val],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # --- ランキング評価 (Recall@K / MRR) ---
        print("\nCalculating Ranking Metrics (Recall@K, MRR)...")
        
        # テストデータに対してスコア予測
        test_data = self.test_df.copy()
        test_data['pred_score'] = self.model.predict(test_data[self.features])
        
        # ユーザーごとにグルーピングしてランキング評価
        # 「そのユーザーが見た(View)商品と買った(Buy)商品の中で、買った商品を上位に表示できたか」
        
        recall_at_10_list = []
        recall_at_5_list = []
        mrr_list = []
        
        grouped = test_data.groupby('user_id')
        
        # 購入履歴があるユーザーのみ評価対象
        valid_users = 0
        
        for uid, group in grouped:
            # 正解（購入）があるか確認
            if group['target'].sum() == 0:
                continue
                
            valid_users += 1
            
            # スコア順にソート（降順）
            sorted_group = group.sort_values('pred_score', ascending=False)
            
            # 正解ラベルのリスト
            targets = sorted_group['target'].values
            
            # 1. Recall@K (上位K個に1つでも正解があれば1)
            # 正解が1つ以上あるので、TopKに1があればRecall=1とみなす（簡易版）
            recall_at_5_list.append(1 if targets[:5].sum() > 0 else 0)
            recall_at_10_list.append(1 if targets[:10].sum() > 0 else 0)
            
            # 2. MRR (最初の正解が出る順位の逆数)
            try:
                # 最初の '1' のインデックスを探す
                first_hit_rank = np.where(targets == 1)[0][0] + 1
                mrr_list.append(1.0 / first_hit_rank)
            except IndexError:
                mrr_list.append(0)

        # 結果出力
        print("\n" + "="*40)
        print("       RANKING EVALUATION       ")
        print("       (Task: Rerank User's Views) ")
        print("="*40)
        print(f" Evaluated Users : {valid_users}")
        print(f" Recall@5        : {np.mean(recall_at_5_list):.4f}")
        print(f" Recall@10       : {np.mean(recall_at_10_list):.4f}")
        print(f" MRR (Mean Rank) : {np.mean(mrr_list):.4f}")
        print("="*40)
        
        # AUCも一応出力
        overall_auc = roc_auc_score(y_val, test_data['pred_score'])
        print(f" Overall AUC     : {overall_auc:.4f}")

        # 特徴量重要度
        importance = pd.DataFrame({
            'Feature': self.features,
            'Gain': self.model.feature_importance(importance_type='gain')
        }).sort_values('Gain', ascending=False)
        print("\nTop Features:\n", importance)

    # ---------------------------------------------------------------------
    # 実行
    # ---------------------------------------------------------------------
    def run(self):
        self.preprocess()
        self.clean_noise()
        self.engineer_features()
        self.create_dataset()
        self.train_and_evaluate()

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
        
        pipeline = SuzuriRankPipeline(log, prod, creat, sale)
        pipeline.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
