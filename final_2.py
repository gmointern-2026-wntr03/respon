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
        self.train_df = None
        self.test_df = None
        self.feature_cols = []
        
        # 結果格納用
        self.results = {}

    # ---------------------------------------------------------------------
    # Phase 1: 前処理 & 時系列分割 (Strict Split)
    # ---------------------------------------------------------------------
    def preprocess_and_split(self):
        print("\n=== Phase 1: Preprocessing & Strict Split ===")
        
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
        self.full_df.sort_values('accessed_at', inplace=True)

        # ★厳密な時系列分割★
        # 全期間のラスト14日間（または20%）をテストデータにする
        # これにより「未来のデータ」は学習に一切使われない
        split_date = self.full_df['accessed_at'].max() - pd.Timedelta(days=14)
        
        print(f"Split Date: {split_date}")
        
        self.train_logs = self.full_df[self.full_df['accessed_at'] < split_date].copy()
        self.test_logs = self.full_df[self.full_df['accessed_at'] >= split_date].copy()
        
        print(f"Train Logs: {len(self.train_logs)}")
        print(f"Test Logs:  {len(self.test_logs)}")

    # ---------------------------------------------------------------------
    # Phase 2: 特徴量生成 (Leakage Prevention)
    # ---------------------------------------------------------------------
    def engineer_features(self):
        print("\n=== Phase 2: Feature Engineering (Train Data Only) ===")
        
        # ★重要: 特徴量は「Trainログ」だけを使って計算する
        
        # 1. 商品・クリエイター特徴量 (Giniなど)
        purchases = self.train_logs[self.train_logs['event_action'] == 'purchase']
        
        # Gini係数計算
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

        # 2. 人気度特徴量 (Popularity) - ベースライン用にも使う
        popularity_map = purchases['product_id'].value_counts().to_dict()

        # 3. 特徴量をマッピングする関数
        def apply_features(df):
            # マテリアル複雑度 (これは静的なのでOK)
            mat_cols = [c for c in df.columns if str(c).startswith('material_') and c != 'material_url']
            df['material_complexity'] = df[mat_cols].notnull().sum(axis=1)
            
            # Trainで計算したGiniをマップ (新規クリエイターは0)
            df['creator_gini_index'] = df['creator_id'].map(gini_map).fillna(0)
            
            # 活動期間
            df['creator_tenure_days'] = (df['accessed_at'] - df['created_at']).dt.days
            
            # Trainでの人気度 (新規商品は0)
            df['popularity_score'] = df['product_id'].map(popularity_map).fillna(0)
            
            # 価格
            df['price'] = df['price'].fillna(0)
            
            return df

        # TrainとTestそれぞれに適用
        self.train_dataset = apply_features(self.train_logs.copy())
        self.test_dataset = apply_features(self.test_logs.copy())
        
        # 学習に使う特徴量
        self.feature_cols = [
            'material_complexity', 'creator_gini_index', 'creator_tenure_days', 
            'popularity_score', 'price'
        ]
        # ※ user_buy_rate はリークしやすいので今回は除外（厳密性重視）
        
        print(f"Features Used: {self.feature_cols}")

    # ---------------------------------------------------------------------
    # Phase 3: データセット構築 (Re-ranking Task)
    # ---------------------------------------------------------------------
    def create_ranking_dataset(self):
        print("\n=== Phase 3: Dataset Construction for Ranking ===")
        
        # 学習用: Hard Negative (Viewしたけど買わなかったものを負例に)
        train_pos = self.train_dataset[self.train_dataset['event_action'] == 'purchase'].copy()
        train_pos['target'] = 1
        
        # 負例: Viewのみ
        # 高速化のため、Train期間のViewからランダムサンプリングして「買ってない」とみなす簡易手法
        # (厳密にはAnti-joinすべきだが、メモリ節約のため簡略化)
        train_neg = self.train_dataset[self.train_dataset['event_action'] == 'view'].sample(n=len(train_pos)*5, random_state=42).copy()
        train_neg['target'] = 0
        
        self.train_df = pd.concat([train_pos, train_neg], ignore_index=True)
        
        # 評価用: Testデータに含まれる「View」と「Purchase」をユーザーごとにまとめる
        # ターゲット: 実際にPurchaseしたアイテム
        # 候補: そのユーザーがTest期間にViewした全アイテム
        
        # Test期間に購入履歴があるユーザーのみ抽出
        test_purchases = self.test_dataset[self.test_dataset['event_action'] == 'purchase']
        valid_users = test_purchases['user_id'].unique()
        
        self.test_eval_df = self.test_dataset[self.test_dataset['user_id'].isin(valid_users)].copy()
        
        # ターゲットフラグ作成
        # 同一セッション内で Purchase されたものを 1, View だけを 0 にしたいが
        # ここではシンプルに event_action == purchase を 1 とする
        self.test_eval_df['target'] = (self.test_eval_df['event_action'] == 'purchase').astype(int)
        
        print(f"Train Samples: {len(self.train_df)}")
        print(f"Test Eval Users: {len(valid_users)}")

    # ---------------------------------------------------------------------
    # Phase 4: モデル学習 & 比較評価
    # ---------------------------------------------------------------------
    def train_and_evaluate(self):
        print("\n=== Phase 4: Training & Benchmarking ===")
        
        # --- 1. Random Model (Baseline) ---
        print("Evaluating: Random Model...")
        self.test_eval_df['score_random'] = np.random.rand(len(self.test_eval_df))
        self.evaluate_model('Random', 'score_random')

        # --- 2. Popularity Model (Baseline) ---
        print("Evaluating: Popularity Model...")
        # Train期間の人気度をスコアにする
        self.test_eval_df['score_pop'] = self.test_eval_df['popularity_score']
        self.evaluate_model('Popularity', 'score_pop')

        # --- 3. LightGBM (Main Model) ---
        print("Training: LightGBM...")
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df['target']
        
        lgb_train = lgb.Dataset(X_train, y_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
        
        model = lgb.train(params, lgb_train, num_boost_round=500)
        
        print("Evaluating: LightGBM...")
        self.test_eval_df['score_lgbm'] = model.predict(self.test_eval_df[self.feature_cols])
        self.evaluate_model('LightGBM', 'score_lgbm')
        
        # 特徴量重要度
        importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Gain': model.feature_importance(importance_type='gain')
        }).sort_values('Gain', ascending=False)
        print("\nLightGBM Feature Importance:\n", importance)

    # ---------------------------------------------------------------------
    # 共通評価関数 (Recall@10)
    # ---------------------------------------------------------------------
    def evaluate_model(self, model_name, score_col):
        recall_at_10 = []
        mrr_list = []
        
        # ユーザーごとに処理
        grouped = self.test_eval_df.groupby('user_id')
        
        for uid, group in grouped:
            # 購入(正解)がないユーザーはスキップ
            if group['target'].sum() == 0:
                continue
                
            # スコアで降順ソート
            sorted_group = group.sort_values(score_col, ascending=False)
            targets = sorted_group['target'].values
            
            # Recall@10
            # 上位10件の中に1つでも購入品があればOK
            recall_at_10.append(1 if targets[:10].sum() > 0 else 0)
            
            # MRR
            try:
                rank = np.where(targets == 1)[0][0] + 1
                mrr_list.append(1.0 / rank)
            except IndexError:
                mrr_list.append(0)
                
        r10 = np.mean(recall_at_10)
        mrr = np.mean(mrr_list)
        
        self.results[model_name] = {'Recall@10': r10, 'MRR': mrr}
        print(f"  [{model_name}] Recall@10: {r10:.4f}, MRR: {mrr:.4f}")

    # ---------------------------------------------------------------------
    # 実行
    # ---------------------------------------------------------------------
    def run(self):
        self.preprocess_and_split()
        self.engineer_features()
        self.create_ranking_dataset()
        self.train_and_evaluate()
        
        print("\n=== Final Comparison ===")
        res_df = pd.DataFrame(self.results).T
        print(res_df)

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
