import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import sys

# 警告の抑制
warnings.filterwarnings('ignore')

# =========================================================================
# 分析・予測パイプラインクラス definition
# =========================================================================
class SuzuriFullPipeline:
    def __init__(self, log_df, product_df, creator_df, sale_df):
        # データのコピー
        self.log_df = log_df.copy()
        self.product_df = product_df.copy()
        self.creator_df = creator_df.copy()
        self.sale_df = sale_df.copy()
        
        self.full_df = None       # 全結合データ
        self.train_df = None      # 学習用データ（負例込み）
        self.model = None         # 学習済みモデル
        self.features = []        # 学習に使用した特徴量リスト

    # ---------------------------------------------------------------------
    # Phase 1: 前処理 (Preprocessing)
    # ---------------------------------------------------------------------
    def preprocess(self):
        print("\n=== Phase 1: Preprocessing & Merging ===")
        
        # 1. 日時変換 (UTC統一でタイムゾーンエラーを回避)
        # エラーハンドリング: 日時パースに失敗した場合はCoerce(NaT)にする
        self.log_df['accessed_at'] = pd.to_datetime(self.log_df['accessed_at'], utc=True, errors='coerce')
        self.creator_df['created_at'] = pd.to_datetime(self.creator_df['created_at'], utc=True, errors='coerce')
        self.sale_df['start_time'] = pd.to_datetime(self.sale_df['start_time'], utc=True, errors='coerce')
        self.sale_df['end_time'] = pd.to_datetime(self.sale_df['end_time'], utc=True, errors='coerce')

        # NaTがある場合は除外（念のため）
        self.log_df = self.log_df.dropna(subset=['accessed_at'])

        # 2. マスタ結合
        print("Merging Dataframes...")
        self.full_df = self.log_df.merge(self.product_df, on='product_id', how='left', suffixes=('', '_prod'))
        self.full_df = self.full_df.merge(self.creator_df, on='creator_id', how='left', suffixes=('', '_creator'))
        
        # 3. 時系列ソート
        self.full_df.sort_values(['user_id', 'accessed_at'], inplace=True)
        print(f"Total Raw Records: {len(self.full_df)}")

    # ---------------------------------------------------------------------
    # Phase 2: ノイズ除去 (Noise Cleaning)
    # ---------------------------------------------------------------------
    def clean_noise(self):
        print("\n=== Phase 2: Noise Detection & Filtering ===")
        df = self.full_df

        # --- A. 自己購入 (Self Purchase) ---
        is_self = (df['user_id'] == df['creator_id'])
        # 名前カラムが存在する場合のみ名前一致もチェック
        if 'name' in df.columns and 'name_creator' in df.columns:
            # fillna('') で欠損によるエラー防止
            is_self = is_self | (df['name'].fillna('') == df['name_creator'].fillna(''))
        
        df['is_self_purchase'] = is_self

        # --- B. 身内買い・太客 (Dominant Buyer) ---
        purchases = df[df['event_action'] == 'purchase']
        if not purchases.empty:
            c_total = purchases.groupby('creator_id').size()
            c_user = purchases.groupby(['creator_id', 'user_id']).size()
            dominance = (c_user / c_total.reindex(c_user.index.get_level_values(0)).values)
            
            sus_pairs = dominance[dominance >= 0.8].index.tolist()
            sus_set = set(sus_pairs)
            
            # 高速化のため map と tuple を使用
            df['temp_pair'] = list(zip(df['creator_id'], df['user_id']))
            df['is_dominant_buyer'] = df['temp_pair'].isin(sus_set)
            df.drop('temp_pair', axis=1, inplace=True)
        else:
            df['is_dominant_buyer'] = False

        # --- C. 閲覧なし購入 (Direct Buy) ---
        print("Detecting Direct Buys...")
        # ユーザー×商品ごとのアクションセットを取得
        actions = df.groupby(['user_id', 'product_id'])['event_action'].apply(set)
        # purchaseはあるがviewがないもの
        no_view_indices = actions[actions.apply(lambda x: 'purchase' in x and 'view' not in x)].index
        no_view_set = set(no_view_indices)
        
        df['temp_pair'] = list(zip(df['user_id'], df['product_id']))
        df['is_direct_buy'] = df['temp_pair'].isin(no_view_set)
        df.drop('temp_pair', axis=1, inplace=True)

        # レポート
        print(f"Noise Report:")
        print(f" - Self Purchases: {df['is_self_purchase'].sum()}")
        print(f" - Dominant Buyers: {df['is_dominant_buyer'].sum()}")
        print(f" - Direct Buys: {df['is_direct_buy'].sum()}")

        # フィルタリング
        clean_condition = (
            (~df['is_self_purchase']) & 
            (~df['is_dominant_buyer']) & 
            (~df['is_direct_buy'])
        )
        self.full_df = df[clean_condition].copy()
        print(f"Records after cleaning: {len(self.full_df)} (Removed: {len(df) - len(self.full_df)})")

    # ---------------------------------------------------------------------
    # Phase 3: 特徴量エンジニアリング (Feature Engineering)
    # ---------------------------------------------------------------------
    def engineer_features(self):
        print("\n=== Phase 3: Feature Engineering ===")
        df = self.full_df

        # 1. デザイン特徴量 (material_1...12)
        mat_cols = [c for c in df.columns if str(c).startswith('material_') and c != 'material_url']
        df['material_complexity'] = df[mat_cols].notnull().sum(axis=1)

        # 2. クリエイター特徴量 (Gini係数 & 活動期間)
        purchases = df[df['event_action'] == 'purchase']
        
        # Gini係数計算関数
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

        # 3. セール特徴量
        print("Processing Sale Features...")
        df['days_to_sale_end'] = 999.0
        df['is_sale_target'] = 0

        # セールデータが空でない場合のみ処理
        if not self.sale_df.empty:
            for _, sale in self.sale_df.iterrows():
                # 期間
                t_mask = (df['accessed_at'] >= sale['start_time']) & (df['accessed_at'] <= sale['end_time'])
                # アイテム一致 (item_category_name と sale['item'] の比較)
                if 'item_category_name' in df.columns and 'item' in sale:
                    i_mask = df['item_category_name'] == sale['item']
                    mask = t_mask & i_mask
                else:
                    mask = t_mask # カテゴリ情報がない場合は期間のみで判定（要注意）
                
                if mask.any():
                    df.loc[mask, 'is_sale_target'] = 1
                    df.loc[mask, 'days_to_sale_end'] = (sale['end_time'] - df.loc[mask, 'accessed_at']).dt.total_seconds() / 86400

        # 4. ユーザー行動特徴量 (Buy Rate)
        u_stats = df.groupby('user_id')['event_action'].value_counts().unstack(fill_value=0)
        if 'purchase' in u_stats.columns:
            total_actions = u_stats.sum(axis=1)
            # ゼロ除算回避
            total_actions = total_actions.replace(0, 1)
            df['user_buy_rate'] = df['user_id'].map(u_stats['purchase'] / total_actions).fillna(0)
        else:
            df['user_buy_rate'] = 0

        self.full_df = df
        print("Features Created.")

    # ---------------------------------------------------------------------
    # Phase 4: データセット構築 (Negative Sampling)
    # ---------------------------------------------------------------------
    def create_dataset(self, negative_ratio=5):
        print("\n=== Phase 4: Dataset Construction (Negative Sampling) ===")
        
        # 正例
        positives = self.full_df[self.full_df['event_action'] == 'purchase'].copy()
        positives['target'] = 1
        
        # 学習に使用する特徴量
        candidates = [
            'material_complexity', 'creator_gini_index', 'creator_tenure_days',
            'days_to_sale_end', 'is_sale_target', 'user_buy_rate', 'price'
        ]
        self.features = [c for c in candidates if c in positives.columns]
        print(f"Using Features: {self.features}")
        
        base_cols = ['user_id', 'product_id', 'accessed_at', 'target'] + self.features
        pos_df = positives[base_cols]
        print(f"Positive samples: {len(pos_df)}")

        if len(pos_df) == 0:
            print("Error: No purchase data found after cleaning.")
            return

        # 負例生成
        # プロダクト情報を辞書化して高速アクセス
        prod_info = self.product_df.set_index('product_id').to_dict(orient='index')
        all_pids = list(prod_info.keys())
        
        neg_samples = []
        
        # 全量だと時間がかかるため、正例1件につきN件の負例を作る
        print(f"Generating Negative Samples (Ratio 1:{negative_ratio})...")
        for _, row in pos_df.iterrows():
            uid = row['user_id']
            acc_time = row['accessed_at']
            u_buy_rate = row['user_buy_rate']
            
            for _ in range(negative_ratio):
                neg_pid = np.random.choice(all_pids)
                p_data = prod_info.get(neg_pid, {})
                
                # 特徴量の割り当て（簡易版）
                # 本来はマスタから正しく引き当てるべきだが、ここではpriceや固定値を使用
                sample = {
                    'user_id': uid,
                    'product_id': neg_pid,
                    'accessed_at': acc_time,
                    'target': 0,
                    'user_buy_rate': u_buy_rate,
                    'price': p_data.get('price', 0),
                    'is_sale_target': 0, 
                    'days_to_sale_end': 999,
                    'material_complexity': 1, # 平均的な値
                    'creator_gini_index': 0,  # 新規/無名と仮定
                    'creator_tenure_days': 365
                }
                neg_samples.append(sample)

        neg_df = pd.DataFrame(neg_samples)
        self.train_df = pd.concat([pos_df, neg_df], ignore_index=True)
        self.train_df.sort_values('accessed_at', inplace=True)
        print(f"Total Training Samples: {len(self.train_df)}")

    # ---------------------------------------------------------------------
    # Phase 5: モデル学習 (LightGBM)
    # ---------------------------------------------------------------------
    def train_model(self):
        print("\n=== Phase 5: Model Training (LightGBM) ===")
        
        if self.train_df is None or len(self.train_df) == 0:
            print("No training data available.")
            return

        # 時系列分割 (Train 80% : Valid 20%)
        split_idx = int(len(self.train_df) * 0.8)
        
        train_data = self.train_df.iloc[:split_idx]
        valid_data = self.train_df.iloc[split_idx:]
        
        X_train = train_data[self.features]
        y_train = train_data['target']
        X_valid = valid_data[self.features]
        y_valid = valid_data['target']
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_seed': 42
        }
        
        print("Training started...")
        self.model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # 評価
        if len(X_valid) > 0:
            preds = self.model.predict(X_valid)
            auc = roc_auc_score(y_valid, preds)
            print(f"\n>>> Validation AUC: {auc:.4f}")
            
            importance = pd.DataFrame({
                'Feature': self.features,
                'Gain': self.model.feature_importance(importance_type='gain')
            }).sort_values('Gain', ascending=False)
            
            print("\nTop Important Features:")
            print(importance)
        else:
            print("Validation set is empty.")

    # ---------------------------------------------------------------------
    # 実行
    # ---------------------------------------------------------------------
    def run(self):
        self.preprocess()
        self.clean_noise()
        self.engineer_features()
        self.create_dataset()
        self.train_model()


# =========================================================================
# メイン実行ブロック
# =========================================================================
if __name__ == "__main__":
    # ファイル定義
    PRODUCT_FILE = 'products_20260204.csv'
    EVENT_FILE   = 'events_20260204.csv'
    CREATOR_FILE = 'creators_20260304.csv'
    SALE_FILE    = 'time_discounts_2025.csv'

    print("Loading CSV files...")
    try:
        # CSV読み込み
        product_df = pd.read_csv(PRODUCT_FILE)
        log_df     = pd.read_csv(EVENT_FILE)
        creator_df = pd.read_csv(CREATOR_FILE)
        sale_df    = pd.read_csv(SALE_FILE)

        print(f" Loaded Products: {len(product_df)} rows")
        print(f" Loaded Events:   {len(log_df)} rows")
        print(f" Loaded Creators: {len(creator_df)} rows")
        print(f" Loaded Sales:    {len(sale_df)} rows")

        # パイプライン実行
        pipeline = SuzuriFullPipeline(log_df, product_df, creator_df, sale_df)
        pipeline.run()

    except FileNotFoundError as e:
        print(f"\n[Error] ファイルが見つかりません: {e}")
        print("CSVファイルが正しい名前で同じフォルダにあるか確認してください。")
    except Exception as e:
        print(f"\n[Error] 予期せぬエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()