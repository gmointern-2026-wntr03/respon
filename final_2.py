import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
import sys
import gc

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
        self.log_df['accessed_at'] = pd.to_datetime(self.log_df['accessed_at'], utc=True, errors='coerce')
        self.creator_df['created_at'] = pd.to_datetime(self.creator_df['created_at'], utc=True, errors='coerce')
        self.sale_df['start_time'] = pd.to_datetime(self.sale_df['start_time'], utc=True, errors='coerce')
        self.sale_df['end_time'] = pd.to_datetime(self.sale_df['end_time'], utc=True, errors='coerce')

        # NaTがある場合は除外
        self.log_df = self.log_df.dropna(subset=['accessed_at'])

        # 2. マスタ結合
        print("Merging Dataframes...")
        # 重複除去 (重要)
        unique_products = self.product_df.drop_duplicates(subset='product_id')
        unique_creators = self.creator_df.drop_duplicates(subset='creator_id')
        
        # 結合
        self.full_df = self.log_df.merge(unique_products, on='product_id', how='left', suffixes=('', '_prod'))
        self.full_df = self.full_df.merge(unique_creators, on='creator_id', how='left', suffixes=('', '_creator'))
        
        # 時系列ソート
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
        if 'name' in df.columns and 'name_creator' in df.columns:
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
            temp_pairs = list(zip(df['creator_id'], df['user_id']))
            df['is_dominant_buyer'] = [x in sus_set for x in temp_pairs]
        else:
            df['is_dominant_buyer'] = False

        # --- C. 閲覧なし購入 (Direct Buy) ---
        print("Detecting Direct Buys...")
        actions = df.groupby(['user_id', 'product_id'])['event_action'].apply(set)
        no_view_indices = actions[actions.apply(lambda x: 'purchase' in x and 'view' not in x)].index
        no_view_set = set(no_view_indices)
        
        temp_pairs = list(zip(df['user_id'], df['product_id']))
        df['is_direct_buy'] = [x in no_view_set for x in temp_pairs]

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
        
        # メモリ解放
        del df
        gc.collect()

    # ---------------------------------------------------------------------
    # Phase 3: 特徴量エンジニアリング (Feature Engineering)
    # ---------------------------------------------------------------------
    def engineer_features(self):
        print("\n=== Phase 3: Feature Engineering ===")
        df = self.full_df

        # 1. デザイン特徴量
        mat_cols = [c for c in df.columns if str(c).startswith('material_') and c != 'material_url']
        df['material_complexity'] = df[mat_cols].notnull().sum(axis=1)

        # 2. クリエイター特徴量 (Gini & Tenure)
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

        # 3. セール特徴量
        print("Processing Sale Features...")
        df['days_to_sale_end'] = 999.0
        df['is_sale_target'] = 0

        if not self.sale_df.empty:
            for _, sale in self.sale_df.iterrows():
                t_mask = (df['accessed_at'] >= sale['start_time']) & (df['accessed_at'] <= sale['end_time'])
                
                # アイテムカテゴリ一致判定
                if 'item_category_name' in df.columns and 'item' in sale:
                    i_mask = df['item_category_name'] == sale['item']
                    mask = t_mask & i_mask
                else:
                    mask = t_mask 
                
                if mask.any():
                    df.loc[mask, 'is_sale_target'] = 1
                    df.loc[mask, 'days_to_sale_end'] = (sale['end_time'] - df.loc[mask, 'accessed_at']).dt.total_seconds() / 86400

        # 4. ユーザー行動特徴量
        u_stats = df.groupby('user_id')['event_action'].value_counts().unstack(fill_value=0)
        if 'purchase' in u_stats.columns:
            total_actions = u_stats.sum(axis=1).replace(0, 1)
            df['user_buy_rate'] = df['user_id'].map(u_stats['purchase'] / total_actions).fillna(0)
        else:
            df['user_buy_rate'] = 0

        self.full_df = df
        print("Features Created.")

    # ---------------------------------------------------------------------
    # Phase 4: データセット構築 (Negative Sampling) 【修正版】
    # ---------------------------------------------------------------------
    def create_dataset(self, negative_ratio=5):
        print("\n=== Phase 4: Dataset Construction (Negative Sampling) ===")
        
        # 1. 特徴量マスタの作成 (Product ID -> Features の対応表)
        # ログデータ(full_df)から、計算済みの特徴量を抽出してマスタ化する
        feature_cols = [
            'product_id', 
            'price', 
            'material_complexity', 
            'creator_gini_index', 
            'creator_tenure_days'
        ]
        # full_dfに存在するカラムのみ使用
        feature_cols = [c for c in feature_cols if c in self.full_df.columns]
        
        # 商品ごとに1行にする (平均値や最大値をとるなどして集約)
        # ここではシンプルに drop_duplicates
        print("Creating Feature Dictionary...")
        feature_master = self.full_df[feature_cols].drop_duplicates('product_id').set_index('product_id')
        feature_dict = feature_master.to_dict(orient='index')
        
        # 2. 正例データの作成
        positives = self.full_df[self.full_df['event_action'] == 'purchase'].copy()
        positives['target'] = 1
        
        use_features = [
            'material_complexity', 'creator_gini_index', 'creator_tenure_days',
            'days_to_sale_end', 'is_sale_target', 'user_buy_rate', 'price'
        ]
        self.features = [c for c in use_features if c in positives.columns]
        
        base_cols = ['user_id', 'product_id', 'accessed_at', 'target'] + self.features
        pos_df = positives[base_cols]
        print(f"Positive samples: {len(pos_df)}")

        if len(pos_df) == 0:
            print("Error: No purchase data found after cleaning.")
            return

        # 3. 負例データの生成
        # 特徴量が存在する商品IDリスト（ログに登場した商品のみを対象とする）
        all_pids = list(feature_dict.keys())
        neg_samples = []
        
        print(f"Generating Negative Samples (Ratio 1:{negative_ratio})...")
        
        # プログレスバー対応
        try:
            from tqdm import tqdm
            iterator = tqdm(pos_df.iterrows(), total=len(pos_df))
        except ImportError:
            iterator = pos_df.iterrows()

        for _, row in iterator:
            uid = row['user_id']
            acc_time = row['accessed_at']
            u_buy_rate = row['user_buy_rate']
            
            for _ in range(negative_ratio):
                # ランダムに商品を選択
                neg_pid = np.random.choice(all_pids)
                
                # ★修正点: 辞書から「本当の特徴量」を取得
                f_data = feature_dict.get(neg_pid, {})
                
                sample = {
                    'user_id': uid,
                    'product_id': neg_pid,
                    'accessed_at': acc_time,
                    'target': 0,
                    'user_buy_rate': u_buy_rate, # ユーザー特徴量はそのまま
                    
                    # 商品・クリエイター特徴量はマスタから取得
                    'price': f_data.get('price', 0),
                    'material_complexity': f_data.get('material_complexity', 0),
                    'creator_gini_index': f_data.get('creator_gini_index', 0),
                    'creator_tenure_days': f_data.get('creator_tenure_days', 0),
                    
                    # 文脈依存特徴量は「非セール」と仮定
                    'is_sale_target': 0, 
                    'days_to_sale_end': 999
                }
                neg_samples.append(sample)

        # 4. 結合
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
