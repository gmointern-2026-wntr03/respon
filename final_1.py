import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# 警告の抑制
warnings.filterwarnings('ignore')

class SuzuriAdvancedAnalyzer:
    def __init__(self, log_df, product_df, creator_df, sale_df):
        # データのコピー（元データを破壊しないため）
        self.log_df = log_df.copy()
        self.product_df = product_df.copy()
        self.creator_df = creator_df.copy()
        self.sale_df = sale_df.copy()
        
        self.full_df = None       # 全結合データ
        self.model_data = None    # 学習用（クリーン）データ
        self.noise_report = {}    # ノイズ除去数などのレポート

    # ---------------------------------------------------------
    # Helper: ジニ係数計算 (クリエイターのファン偏り分析用)
    # ---------------------------------------------------------
    def _calculate_gini(self, array):
        """配列の不平等度（ジニ係数）を計算。1に近いほど特定の少数に依存している。"""
        array = np.array(array, dtype=np.float64)
        if np.amin(array) < 0: return 0
        if np.sum(array) == 0: return 0
        array += 0.0000001 # ゼロ除算回避
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    # ---------------------------------------------------------
    # 1. 前処理 & データ結合 (Preprocessing)
    # ---------------------------------------------------------
    def preprocess(self):
        print("--- [1/6] Preprocessing & Merging ---")
        
        # 日時変換
        self.log_df['accessed_at'] = pd.to_datetime(self.log_df['accessed_at'])
        self.creator_df['created_at'] = pd.to_datetime(self.creator_df['created_at'])
        self.sale_df['start_time'] = pd.to_datetime(self.sale_df['start_time'])
        self.sale_df['end_time'] = pd.to_datetime(self.sale_df['end_time'])

        # マスタ結合: ログをベースに商品・クリエイター情報を紐付け
        # suffixesを使ってカラム名の衝突を回避 (例: name -> name_creator)
        self.full_df = self.log_df.merge(self.product_df, on='product_id', how='left', suffixes=('', '_prod'))
        self.full_df = self.full_df.merge(self.creator_df, on='creator_id', how='left', suffixes=('', '_creator'))
        
        # 時系列ソート
        self.full_df.sort_values(['user_id', 'accessed_at'], inplace=True)
        print(f"Total Log Records: {len(self.full_df)}")

    # ---------------------------------------------------------
    # 2. ノイズ検出 (Noise Detection) ★最重要
    # ---------------------------------------------------------
    def detect_noise(self):
        print("--- [2/6] Detecting Noise & Suspicious Transactions ---")
        
        # --- A. Is_Self_Purchase (自己購入判定) ---
        # 1. IDが一致
        id_match = (self.full_df['user_id'] == self.full_df['creator_id'])
        
        # 2. 名前が一致 (もしユーザー名カラムがあれば)
        name_match = False
        if 'name' in self.full_df.columns and 'name_creator' in self.full_df.columns:
            # ユーザー名とクリエイター名が完全一致する場合も怪しいとみなす
            name_match = (self.full_df['name'] == self.full_df['name_creator'])
        
        self.full_df['is_self_purchase'] = id_match | name_match
        self.noise_report['self_purchase_count'] = self.full_df['is_self_purchase'].sum()

        # --- B. Dominant Buyer (太客・身内判定) ---
        # そのクリエイターの売上の80%以上を1人が占めている場合
        purchases = self.full_df[self.full_df['event_action'] == 'purchase']
        
        if not purchases.empty:
            # クリエイターごとの総販売数
            creator_total = purchases.groupby('creator_id').size()
            # クリエイター×ユーザーごとの購入数
            creator_user = purchases.groupby(['creator_id', 'user_id']).size()
            
            # 割合計算
            # creator_totalのインデックスを合わせて割り算
            dominance = (creator_user / creator_total.reindex(creator_user.index.get_level_values(0)).values)
            
            # 閾値0.8以上のペアを抽出
            suspicious_pairs = dominance[dominance >= 0.8].index.tolist() # list of (creator_id, user_id)
            
            # フラグ付与
            suspicious_df = pd.DataFrame(suspicious_pairs, columns=['creator_id', 'user_id'])
            suspicious_df['is_dominant_buyer'] = True
            
            self.full_df = self.full_df.merge(suspicious_df, on=['creator_id', 'user_id'], how='left')
            self.full_df['is_dominant_buyer'] = self.full_df['is_dominant_buyer'].fillna(False)
        else:
            self.full_df['is_dominant_buyer'] = False
            
        self.noise_report['dominant_buyer_transactions'] = self.full_df['is_dominant_buyer'].sum()

        # --- C. Direct_Traffic_Buy (閲覧なし購入) ---
        # セッションまたはユーザー×商品単位で、viewがないのにpurchaseがあるケース
        # groupby + apply(set) でアクション集合を確認
        actions_per_item = self.full_df.groupby(['user_id', 'product_id'])['event_action'].apply(set)
        
        # purchaseがあり、かつviewがないインデックスを特定
        no_view_indices = actions_per_item[actions_per_item.apply(lambda x: 'purchase' in x and 'view' not in x)].index
        
        no_view_df = pd.DataFrame(no_view_indices.tolist(), columns=['user_id', 'product_id'])
        no_view_df['is_direct_buy'] = True
        
        self.full_df = self.full_df.merge(no_view_df, on=['user_id', 'product_id'], how='left')
        self.full_df['is_direct_buy'] = self.full_df['is_direct_buy'].fillna(False)
        
        self.noise_report['direct_buy_transactions'] = self.full_df['is_direct_buy'].sum()

    # ---------------------------------------------------------
    # 3. 特徴量エンジニアリング (Feature Engineering)
    # ---------------------------------------------------------
    def engineer_features(self):
        print("--- [3/6] Feature Engineering ---")
        df = self.full_df

        # --- A. Material Features (デザイン軸) ---
        # 1. 複雑度: material_1 ~ 12 のうち値が入っている数
        mat_cols = [f'material_{i}' for i in range(1, 13)]
        existing_mat_cols = [c for c in mat_cols if c in df.columns]
        df['material_complexity'] = df[existing_mat_cols].notnull().sum(axis=1)

        # 2. デザイン展開数 (IP力): 同じ画像がいくつのカテゴリで商品化されているか
        if 'material_url' in df.columns:
            mat_counts = df.groupby('material_url')['item_category_id'].nunique().reset_index(name='material_cross_cat_count')
            df = df.merge(mat_counts, on='material_url', how='left')

        # --- B. Creator Features (クリエイター軸) ---
        # 1. 活動期間
        df['creator_tenure_days'] = (df['accessed_at'] - df['created_at']).dt.days
        
        # 2. 公式フラグ (数値化)
        if 'official' in df.columns:
            df['is_official'] = df['official'].astype(int)

        # 3. Creator Gini Index (顧客集中度)
        # 購入ログのみで計算
        purchases = df[df['event_action'] == 'purchase']
        gini_map = {}
        for cid, group in purchases.groupby('creator_id'):
            # そのクリエイターのユーザーごとの購入回数分布
            user_counts = group['user_id'].value_counts().values
            gini_map[cid] = self._calculate_gini(user_counts)
        
        # 全データにマッピング（購入がないクリエイターは0）
        df['creator_gini_index'] = df['creator_id'].map(gini_map).fillna(0)

        # --- C. Sale Features (タイミング・価格感応度) ---
        df['is_sale_target'] = False
        df['days_to_sale_end'] = -1.0 # セール外は負の値
        df['discount_amount'] = 0

        # Saleテーブルを用いて判定
        for _, sale in self.sale_df.iterrows():
            # 期間判定
            time_mask = (df['accessed_at'] >= sale['start_time']) & (df['accessed_at'] <= sale['end_time'])
            
            # アイテム対象判定 (itemカラムがカテゴリ名と一致するか、商品名に含まれるか)
            # ここではシンプルにカテゴリ名一致と仮定
            item_mask = df['item_category_name'] == sale['item']
            
            target_mask = time_mask & item_mask
            
            if target_mask.any():
                df.loc[target_mask, 'is_sale_target'] = True
                df.loc[target_mask, 'discount_amount'] = sale['amount_with_tax']
                # 残り時間(日)を計算
                df.loc[target_mask, 'days_to_sale_end'] = (sale['end_time'] - df.loc[target_mask, 'accessed_at']).dt.total_seconds() / 86400

        # --- D. User Behavior Features (行動ログ) ---
        # 1. Favorite to Purchase Ratio (慎重さ指標)
        user_stats = df.groupby('user_id')['event_action'].value_counts().unstack(fill_value=0)
        if 'favorite' in user_stats.columns and 'purchase' in user_stats.columns:
            user_stats['fav_to_buy_ratio'] = user_stats['purchase'] / (user_stats['favorite'] + 1)
            df = df.merge(user_stats[['fav_to_buy_ratio']], on='user_id', how='left')

        # 2. Color Preference (色の好み)
        # ユーザーごとの色購入傾向をOne-Hotエンコーディングして集計
        if 'exemplary_item_color_name' in df.columns:
            # データ量削減のためTop 20色以外はOtherにするなどの処理があると良いが、ここでは全色展開
            color_dummies = pd.get_dummies(df['exemplary_item_color_name'], prefix='color_pref')
            # ユーザーIDと結合してSumをとる
            user_color_pref = pd.concat([df['user_id'], color_dummies], axis=1).groupby('user_id').sum().reset_index()
            # メインテーブルに結合
            df = df.merge(user_color_pref, on='user_id', how='left')

        self.full_df = df

    # ---------------------------------------------------------
    # 4. データマート作成 (Filtering)
    # ---------------------------------------------------------
    def create_clean_datamart(self):
        print("--- [4/6] Creating Clean Data Mart (Filtering) ---")
        
        # ノイズフラグが立っていないデータのみを抽出
        # かつ、ジニ係数が高すぎる（特定ファン専用）クリエイターも除外
        
        criteria = (
            (~self.full_df['is_self_purchase']) &      # 自己購入でない
            (~self.full_df['is_dominant_buyer']) &     # 身内買いでない
            (~self.full_df['is_direct_buy']) &         # 検索経由である
            (self.full_df['creator_gini_index'] < 0.8) # 一般に売れている
        )
        
        self.model_data = self.full_df[criteria].copy()
        
        print(f"Original Records: {len(self.full_df)}")
        print(f"Cleaned Records:  {len(self.model_data)}")
        print(f"Removed Records:  {len(self.full_df) - len(self.model_data)}")
        
        return self.model_data

    # ---------------------------------------------------------
    # 5. Deep Dive Analysis (分析)
    # ---------------------------------------------------------
    def analyze_deep_dive(self):
        print("--- [5/6] Running Deep Dive Analysis ---")
        results = {}

        # ① マテリアル単位での「併せ買い」分析
        # 同じ material_url で、異なる item_category_id を買ったユーザーを探す
        print("Analysis 1: Cross-Selling Potential...")
        purchases = self.model_data[self.model_data['event_action'] == 'purchase']
        
        if not purchases.empty and 'material_url' in purchases.columns:
            # マテリアル×ユーザーごとのカテゴリ数
            mat_user_counts = purchases.groupby(['material_url', 'user_id'])['item_category_name'].nunique()
            # 2カテゴリ以上買っているケースを抽出
            cross_sells = mat_user_counts[mat_user_counts >= 2]
            
            if not cross_sells.empty:
                top_materials = cross_sells.index.get_level_values(0).value_counts().head(5)
                results['top_cross_sell_materials'] = top_materials
                print(f"Top 5 Materials used in Cross-Selling:\n{top_materials}")
            else:
                print("No significant cross-selling found.")

        # ② Price vs Profit vs Conversion (価格弾力性)
        print("\nAnalysis 2: Price & Profit Analysis...")
        # 商品ごとのViewとPurchaseを集計
        item_metrics = self.model_data.groupby(['product_id', 'price', 'profit']).agg(
            views=('event_action', lambda x: (x=='view').sum()),
            purchases=('event_action', lambda x: (x=='purchase').sum())
        ).reset_index()
        
        # CVR計算
        item_metrics['cvr'] = item_metrics['purchases'] / (item_metrics['views'] + 1)
        # 利益率（推定）
        item_metrics['profit_margin'] = item_metrics['profit'] / (item_metrics['price'] + 1)
        
        # 相関行列
        corr_matrix = item_metrics[['price', 'profit', 'profit_margin', 'cvr']].corr()
        results['price_correlation'] = corr_matrix
        print(f"Correlation Matrix:\n{corr_matrix}")

        # ③ Text Mining (Descriptionのキーワード抽出)
        print("\nAnalysis 3: Text Mining (Top Keywords)...")
        # 売上がトップ100の商品の説明文を抽出
        top_sellers = item_metrics.sort_values('purchases', ascending=False).head(100)
        target_ids = top_sellers['product_id'].values
        
        texts = self.model_data[self.model_data['product_id'].isin(target_ids)]['description'].fillna('').unique()
        
        if len(texts) > 0:
            try:
                # 日本語の場合は分かち書きが必要ですが、ここでは簡易的にsklearnの機能で実行
                # 本番環境では janome や MeCab で分かち書きした文字列リストを渡してください
                vectorizer = TfidfVectorizer(max_features=20) 
                # 日本語用に文字単位(analyzer='char')や、別途分かち書き処理を推奨
                # ここでは簡易的に例外処理で囲みます
                vectorizer.fit_transform(texts)
                keywords = vectorizer.get_feature_names_out()
                results['top_keywords'] = keywords
                print(f"Top Keywords: {keywords}")
            except Exception as e:
                print(f"Text analysis skipped or failed: {e}")

        return results

    # ---------------------------------------------------------
    # 6. 実行メイン (Run)
    # ---------------------------------------------------------
    def run(self):
        self.preprocess()
        self.detect_noise()
        self.engineer_features()
        clean_df = self.create_clean_datamart()
        analysis_results = self.analyze_deep_dive()
        
        print("\n--- Processing Complete ---")
        print("Noise Report:", self.noise_report)
        return clean_df, analysis_results


if __name__ == "__main__":
    print("=== Generating Dummy Data for Testing ===")
    
    # ダミーデータの作成
    # 1. Product DF
    products = pd.read_csv("products_20260204.csv")
    
    # 2. Creator DF
    creators = pd.read_csv("creators_20260204.csv")

    # 3. Log DF (ユーザー行動)
    logs = pd.read_csv("events_20260204.csv")
    
    # 4. Sale DF
    sales = pd.read_csv("time_discounts2025.csv")

    # クラスの初期化と実行
    analyzer = SuzuriAdvancedAnalyzer(logs, products, creators, sales)
    final_df, results = analyzer.run()

    # 確認
    print("\nSample of Cleaned Data (Top 3 rows):")
    print(final_df[['user_id', 'product_id', 'event_action', 'is_sale_target', 'creator_gini_index']].head(3))