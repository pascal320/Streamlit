import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# optional heavy deps will be imported lazily
_HAS_XGBOOST = False
_HAS_PROPHET = False
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except Exception:
    xgb = None

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    Prophet = None


def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            df = pd.read_excel(uploaded_file)
    else:
        # prefer clustered dataset if present
        if os.path.exists('ML_Clustered_Database_Horizon_Global_Consulting.csv'):
            df = pd.read_csv('ML_Clustered_Database_Horizon_Global_Consulting.csv')
        else:
            # fallback to original default
            df = pd.read_csv('Final_Database_Horizon_Global_Consulting.csv')
    return df


def load_saved_artifacts(search_dir='.'):
    """Look for common saved artifacts (pickled models, processed csvs) in the given directory.
    Returns a dict with keys: 'models' -> {name: model}, 'data' -> {name: path}
    """
    artifacts = {'models': {}, 'data': {}}
    try:
        # search for pickles
        pkl_files = glob.glob(os.path.join(search_dir, '*.pkl')) + glob.glob(os.path.join(search_dir, '*.joblib'))
        for p in pkl_files:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                artifacts['models'][name] = joblib.load(p)
            except Exception:
                try:
                    with open(p, 'rb') as fh:
                        artifacts['models'][name] = fh.read()
                except Exception:
                    pass

        # search for csvs that might be processed data
        csv_files = glob.glob(os.path.join(search_dir, '*processed*.csv')) + glob.glob(os.path.join(search_dir, '*processed*.txt'))
        csv_files += glob.glob(os.path.join(search_dir, '*.csv'))
        for p in csv_files:
            name = os.path.splitext(os.path.basename(p))[0]
            artifacts['data'][name] = p
    except Exception:
        pass
    return artifacts


def numeric_cols(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df):
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def compute_kpis(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train_simple_models(df, target_col, features=None, test_size=0.2, random_state=42, preloaded_models=None):
    """Train simple LR and RF baselines and (if provided) evaluate preloaded models on the same test split.

    preloaded_models: dict mapping name->model objects (as returned by load_saved_artifacts)
    """
    X = df[features] if features is not None else df[numeric_cols(df)].drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)

    preds = {}
    preds['LinearRegression'] = lr.predict(X_test)
    preds['RandomForest'] = rf.predict(X_test)

    # Evaluate any preloaded models on the same test split if possible
    if preloaded_models:
        for name, mdl in preloaded_models.items():
            try:
                p = predict_with_model(mdl, X_test, model_name=name)
                # ensure 1d
                p = np.asarray(p).reshape(-1,)
                if p.shape[0] == X_test.shape[0]:
                    preds[name] = p
            except Exception:
                # skip models that can't be applied to this feature set
                continue

    kpis = {name: compute_kpis(y_test, p) for name, p in preds.items()}
    return {'models': {'lr': lr, 'rf': rf}, 'X_test': X_test, 'y_test': y_test, 'preds': preds, 'kpis': kpis}


def predict_with_model(model, X, model_name=None, date_col=None):
    """Attempt to predict using a saved model object. Handles sklearn-like, xgboost.Booster, and Prophet where possible.

    - model: loaded object from joblib/pickle
    - X: DataFrame or 2D array
    - model_name: optional string filename to help heuristics
    - date_col: for prophet-style models, name of date column if needed
    Returns: 1D numpy array of predictions or raises Exception
    """
    # sklearn-like models
    if hasattr(model, 'predict') and not (Prophet and isinstance(model, Prophet)):
        try:
            return model.predict(X)
        except Exception:
            # try numpy array
            try:
                return model.predict(np.asarray(X))
            except Exception:
                pass

    # xgboost.Booster
    if _HAS_XGBOOST and hasattr(xgb, 'DMatrix') and isinstance(model, getattr(xgb, 'Booster', object)):
        try:
            dmat = xgb.DMatrix(X)
            return model.predict(dmat)
        except Exception:
            # try numpy
            dmat = xgb.DMatrix(np.asarray(X))
            return model.predict(dmat)

    # Prophet model (expects DataFrame with ds column)
    if _HAS_PROPHET and Prophet and isinstance(model, Prophet):
        # X must contain 'ds' column: if X is a DataFrame, try to build ds
        if isinstance(X, pd.DataFrame):
            if 'ds' not in X.columns:
                if date_col and date_col in X.columns:
                    future = pd.DataFrame({'ds': pd.to_datetime(X[date_col])})
                else:
                    # if index appears to be datetime-like
                    try:
                        future = pd.DataFrame({'ds': pd.to_datetime(X.index)})
                    except Exception:
                        raise ValueError('Prophet model requires a date (ds) column or datetime index')
            else:
                future = X[['ds']].copy()
            pred = model.predict(future)
            return pred['yhat'].values

    # fallback: attempt joblib-like loading/unwrapping
    if hasattr(model, 'predict'):
        return model.predict(X)

    raise RuntimeError('Unable to predict with provided model. Missing predict method or unsupported model type.')


def get_feature_importances_from_saved_model(model, feature_names=None):
    """Extract feature importances from a saved model if possible.
    Returns DataFrame with columns ['feature','importance'] or None.
    """
    # sklearn-style
    if hasattr(model, 'feature_importances_'):
        importances = getattr(model, 'feature_importances_')
        names = feature_names if feature_names is not None else [f'f{i}' for i in range(len(importances))]
        return pd.DataFrame({'feature': names, 'importance': importances}).sort_values('importance', ascending=False)

    # xgboost
    if _HAS_XGBOOST and isinstance(model, getattr(xgb, 'Booster', object)):
        try:
            score = model.get_score(importance_type='weight')
            items = [{'feature': k, 'importance': v} for k, v in score.items()]
            df_imp = pd.DataFrame(items).sort_values('importance', ascending=False)
            if df_imp.empty and feature_names is not None:
                # maybe trained xgb scikit API
                try:
                    arr = model.feature_importances_
                    return pd.DataFrame({'feature': feature_names, 'importance': arr}).sort_values('importance', ascending=False)
                except Exception:
                    pass
            return df_imp
        except Exception:
            pass

    # not available
    return None


# ----- lightweight time-series and forecasting helpers -----
def aggregate_timeseries(df, date_col, value_col, freq='M', agg='sum'):
    s = df[[date_col, value_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col])
    if freq == 'D':
        sdf = s.set_index(date_col).resample('D').sum().reset_index()
    elif freq == 'M':
        sdf = s.set_index(date_col).resample('M').sum().reset_index()
    elif freq == 'Y':
        sdf = s.set_index(date_col).resample('Y').sum().reset_index()
    else:
        sdf = s.set_index(date_col).resample(freq).sum().reset_index()
    sdf = sdf.rename(columns={date_col: 'ds', value_col: 'y'})
    return sdf


def make_lag_features(ts_df, lags=(1,7), rolling_windows=(3,7)):
    df = ts_df.copy().set_index('ds')
    for lag in range(1, max(lags)+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    for w in rolling_windows:
        df[f'roll_mean_{w}'] = df['y'].shift(1).rolling(window=w, min_periods=1).mean()
    df = df.dropna().reset_index()
    return df


def train_xgb_forecast(ts_df, horizon=12, params=None):
    # prepare lag features
    df_feats = make_lag_features(ts_df, lags=range(1,8), rolling_windows=(3,7))
    feature_cols = [c for c in df_feats.columns if c not in ['ds','y']]
    X = df_feats[feature_cols]
    y = df_feats['y']
    if _HAS_XGBOOST:
        model = xgb.XGBRegressor(n_estimators=params.get('n_estimators',100), max_depth=params.get('max_depth',3), learning_rate=params.get('learning_rate',0.1), random_state=0)
    else:
        model = RandomForestRegressor(n_estimators=params.get('n_estimators',100), random_state=0)
    model.fit(X, y)

    # iterative forecasting using last rows
    last = ts_df.set_index('ds').copy()
    preds = []
    cur = last.copy()
    for i in range(horizon):
        # build features from cur
        recent = cur['y']
        feat = {}
        for lag in range(1,8):
            feat[f'lag_{lag}'] = recent.iloc[-lag] if len(recent) >= lag else recent.mean()
        for w in (3,7):
            feat[f'roll_mean_{w}'] = recent.shift(1).rolling(window=w, min_periods=1).mean().iloc[-1]
        Xp = pd.DataFrame([feat])
        p = model.predict(Xp)[0]
        # advance
        next_idx = cur.index[-1] + (cur.index[-1] - cur.index[-2]) if len(cur) > 1 else cur.index[-1] + pd.Timedelta(days=1)
        cur.loc[next_idx] = p
        preds.append(p)
    return model, preds


def train_prophet(ts_df, periods=12, seasonality_mode='additive', changepoint_prior_scale=0.05):
    if not _HAS_PROPHET:
        raise RuntimeError('Prophet is not installed in the environment')
    m = Prophet(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_prior_scale)
    m.fit(ts_df.rename(columns={'ds':'ds','y':'y'}))
    future = m.make_future_dataframe(periods=periods, freq='M')
    f = m.predict(future)
    return m, f


def detect_default_columns(df):
    """Heuristic detection of common column roles: date, demand/quantity, price, profit, sku, trade partner."""
    cols = {c.lower(): c for c in df.columns}
    res = {'date': None, 'demand': None, 'price': None, 'profit': None, 'sku': None, 'trade_partner': None}
    # date
    for k in ['date', 'order_date', 'ds', 'timestamp']:
        if k in cols:
            res['date'] = cols[k]
            break
    # demand/quantity
    for k in ['quantity', 'qty', 'demand', 'units', 'inventory', 'sales', 'volume']:
        for col in df.columns:
            if k in col.lower():
                res['demand'] = col
                break
        if res['demand']:
            break
    # price
    for col in df.columns:
        if 'price' in col.lower() or 'list_price' in col.lower():
            res['price'] = col
            break
    # profit / dcm
    for col in df.columns:
        if any(t in col.lower() for t in ['profit', 'dcm', 'margin']):
            res['profit'] = col
            break
    # sku / product
    for col in df.columns:
        if any(t in col.lower() for t in ['sku', 'product', 'item', 'material', 'partno']):
            res['sku'] = col
            break
    # trade partner / customer
    for col in df.columns:
        if any(t in col.lower() for t in ['partner', 'customer', 'dealer', 'distributor', 'trade']):
            res['trade_partner'] = col
            break
    return res



def main():
    st.set_page_config(page_title='Whirlpool AI Insights', layout='wide', initial_sidebar_state='expanded')
    # suppress warnings in console for cleaner output
    warnings.filterwarnings('ignore')

    st.markdown("""
    <style>
    .block-container{padding:1rem 2rem}
    .stSidebar .css-1d391kg {background: #ffffff}
    </style>
    """, unsafe_allow_html=True)

    st.title('Whirlpool — AI Insights')
    st.caption('Minimal, light-themed interactive dashboard for analytics and price optimization')

    # Sidebar
    st.sidebar.header('Controls')
    uploaded_file = st.sidebar.file_uploader('Upload CSV/Excel (optional)', type=['csv', 'xlsx', 'xls'])
    prefer_saved = st.sidebar.checkbox('Prefer saved notebook artifacts (models/data) if available', value=False)
    force_retrain = st.sidebar.checkbox('Force local retrain (ignore saved artifacts)', value=True)
    notebook_only = st.sidebar.checkbox('Notebook-only mode (use saved artifacts only, no retrain)', value=False)
    page = st.sidebar.radio('Pages', ['Dashboard Overview', 'Forecasting Models', 'Feature Importance', 'Data Insights', 'Price Optimization'])
    st.sidebar.markdown('---')
    st.sidebar.write('Dataset & Filters')
    use_cluster = st.sidebar.checkbox('Use clustered dataset (if available)', value=True)

    df = None
    try:
        # Determine whether to actually use saved artifacts (allow forcing local retrain)
        prefer_saved_effective = prefer_saved and (not force_retrain)
        # If user prefers saved artifacts and there's a processed CSV, use that as default
        saved = load_saved_artifacts('.') if prefer_saved_effective else {'models': {}, 'data': {}}
        df = None
        if uploaded_file is not None:
            df = load_data(uploaded_file)
        else:
            # If user opts to use clustered dataset and it exists, load it
            if use_cluster and os.path.exists('ML_Clustered_Database_Horizon_Global_Consulting.csv'):
                df = pd.read_csv('ML_Clustered_Database_Horizon_Global_Consulting.csv')
            else:
                # check for a processed csv first when preferring saved artifacts
                if prefer_saved and saved.get('data'):
                    # pick a likely candidate: contains 'processed' or the first CSV available
                    candidates = [p for k, p in saved['data'].items() if 'processed' in k.lower()]
                    use_path = candidates[0] if candidates else next(iter(saved['data'].values()), None)
                    if use_path:
                        try:
                            df = pd.read_csv(use_path)
                        except Exception:
                            df = load_data(None)
                    else:
                        df = load_data(None)
                else:
                    df = load_data(None)
    except FileNotFoundError:
        st.error('Default dataset not found in workspace. Please upload a CSV file.')
        return
    except Exception as e:
        st.error(f'Error loading dataset: {e}')
        return

    st.sidebar.markdown(f'Data rows: {df.shape[0]} | cols: {df.shape[1]}')

    # common selectors
    num_cols = numeric_cols(df)
    cat_cols = categorical_cols(df)

    # detect sensible defaults so user doesn't have to pick every time
    defaults = detect_default_columns(df)

    # If user asked to prefer saved models, try to extract them for use later
    saved_models = saved.get('models', {}) if 'saved' in locals() else {}

    if page == 'Dashboard Overview':
        st.header('Overview — Forecast & Price Optimization')
        st.write('Upload your CSV (use the sidebar). Overview will show side-by-side Prophet (time-series) and price optimization for a selectable product.')

        if uploaded_file is None and df is None:
            st.info('Please upload a CSV in the sidebar to begin.')
        else:
            # minimal controls for overview
            # use detected defaults when available to avoid forcing user selection
            date_options = [c for c in df.columns if 'date' in c.lower()] + [c for c in df.columns]
            date_default = defaults.get('date') if defaults.get('date') in date_options else date_options[0]
            date_col = st.selectbox('Date column', date_options, index=date_options.index(date_default))
            demand_default = defaults.get('demand') if defaults.get('demand') in num_cols else (num_cols[0] if num_cols else None)
            demand_col = st.selectbox('Demand / quantity column', num_cols, index=num_cols.index(demand_default) if demand_default in num_cols else 0)
            sku_col = None
            if cat_cols:
                sku_default = defaults.get('sku') if defaults.get('sku') in cat_cols else None
                sku_opt = [None] + cat_cols
                sku_col = st.selectbox('SKU / product column (optional)', sku_opt, index=sku_opt.index(sku_default) if sku_default in sku_opt else 0)
            price_default = defaults.get('price') if defaults.get('price') in num_cols else (num_cols[0] if num_cols else None)
            price_col = st.selectbox('Price column (for optimization)', [c for c in num_cols], index=num_cols.index(price_default) if price_default in num_cols else 0)
            profit_default = defaults.get('profit') if defaults.get('profit') in num_cols and defaults.get('profit') != price_col else (next((c for c in num_cols if c != price_col), price_col) if num_cols else None)
            profit_col = st.selectbox('Profit/DCM column (for optimization)', [c for c in num_cols if c != price_col] or [price_col], index=([c for c in num_cols if c != price_col] or [price_col]).index(profit_default) if profit_default in ( [c for c in num_cols if c != price_col] or [price_col]) else 0)

            # forecasting tuning
            st.markdown('### Forecast tuning')
            prop_seasonality = st.selectbox('Prophet seasonality mode', ['additive', 'multiplicative'], index=0)
            prop_changepoint = st.slider('Prophet changepoint prior scale', 0.001, 0.5, 0.05)
            xgb_estimators = st.slider('XGBoost / RF n_estimators', 10, 500, 100)
            xgb_depth = st.slider('XGBoost max_depth', 1, 10, 3)

            # seasonality/date range selectors
            st.markdown('### Seasonality & date range')
            # ensure date parsed
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception:
                st.warning('Could not parse the selected date column. Forecasting may fail.')

            min_date = df[date_col].min()
            max_date = df[date_col].max()
            drange = st.date_input('Select date range', value=(min_date, max_date), min_value=min_date, max_value=max_date)

            # choose product for price optimization (random default)
            product_value = None
            if sku_col:
                unique_skus = df[sku_col].dropna().unique().tolist()
                # prefer detected sku value if present
                sku_default_value = None
                if defaults.get('sku') and defaults.get('sku') == sku_col and len(unique_skus) > 0:
                    sku_default_value = unique_skus[0]
                product_value = st.selectbox('Select product for price optimization', ['<random>'] + unique_skus, index=0 if sku_default_value is None else (['<random>'] + unique_skus).index(sku_default_value))
                if product_value == '<random>' and unique_skus:
                    import random
                    product_value = random.choice(unique_skus)

            # aggregation frequency for price optimization (Overview)
            agg_freq = st.selectbox('Aggregate by (Overview)', ['M', 'Y'], index=0)

            # trade partner filter defaults for Overview (safe defaults so variables exist)
            trade_partner_col = defaults.get('trade_partner') if defaults.get('trade_partner') in cat_cols else None
            trade_partner_value = None
            if trade_partner_col:
                tp_vals = df[trade_partner_col].dropna().unique().tolist()
                if tp_vals:
                    trade_partner_value = st.selectbox('Select trade partner (optional)', ['<all>'] + tp_vals)

            # run forecasts
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('Prophet forecast (demand)')
                try:
                    ts = aggregate_timeseries(df[(df[date_col] >= pd.to_datetime(drange[0])) & (df[date_col] <= pd.to_datetime(drange[1]))], date_col, demand_col, freq='M')
                    m, forecast_df = train_prophet(ts, periods=12, seasonality_mode=prop_seasonality, changepoint_prior_scale=prop_changepoint)
                    fig_p = px.line(forecast_df, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'], title='Prophet forecast (yhat, lower, upper)')
                    st.plotly_chart(fig_p, use_container_width=True)
                except Exception as e:
                    st.error(f'Prophet forecast failed: {e}')

            with col2:
                st.subheader('Price optimization (sample product)')
                try:
                    # filter product and trade partner
                    pop = df.copy()
                    if sku_col and product_value:
                        pop = pop[pop[sku_col] == product_value]
                    if trade_partner_col and trade_partner_value and trade_partner_value != '<all>':
                        pop = pop[pop[trade_partner_col] == trade_partner_value]
                    # aggregate monthly
                    resample_rule = 'M' if agg_freq == 'M' else 'Y'
                    pop_agg = pop.set_index(pd.to_datetime(pop[date_col])).resample(resample_rule).agg({price_col:'mean', profit_col:'sum'})
                    pop_agg = pop_agg.dropna()
                    if pop_agg.empty:
                        st.warning('Not enough data for selected product to run price optimization.')
                    else:
                        X = pop_agg[[price_col]].values.reshape(-1,1)
                        y = pop_agg[profit_col].values
                        # simple model choice
                        if _HAS_XGBOOST:
                            price_model = xgb.XGBRegressor(n_estimators=xgb_estimators, max_depth=xgb_depth, random_state=0)
                        else:
                            price_model = RandomForestRegressor(n_estimators=xgb_estimators, random_state=0)
                        price_model.fit(X, y)
                        grid = np.linspace(X.min(), X.max(), 200).reshape(-1,1)
                        preds = price_model.predict(grid)
                        opt = grid[np.argmax(preds)][0]
                        viz = pd.DataFrame({'price':grid.flatten(), 'pred_profit':preds})
                        fig_price = px.line(viz, x='price', y='pred_profit', title=f'Price optimization for product {product_value}')
                        fig_price.add_scatter(x=[opt], y=[preds.max()], mode='markers', name='Optimal', marker={'size':12, 'color':'green'})
                        st.plotly_chart(fig_price, use_container_width=True)
                        st.metric('Optimal price (est.)', f'{float(opt):.2f}')
                except Exception as e:
                    st.error(f'Price optimization failed: {e}')

    elif page == 'Forecasting Models':
        st.header('Forecasting Models')
        st.write('Train and compare models. Choose a numeric target and features.')

        if not num_cols:
            st.warning('No numeric columns available for forecasting.')
        else:
            # allow selecting target and filtering by category for focused training
            target = st.selectbox('Select target (numeric)', num_cols, index=0)
            # category filter
            category_col = st.selectbox('Filter by category (optional)', [None] + cat_cols)
            filter_value = None
            if category_col:
                values = df[category_col].dropna().unique().tolist()
                filter_value = st.selectbox(f'Choose {category_col} value', ['<all>'] + values)

            if category_col and filter_value and filter_value != '<all>':
                df_filtered = df[df[category_col] == filter_value].copy()
            else:
                df_filtered = df.copy()

            features = st.multiselect('Select features', [c for c in numeric_cols(df_filtered) if c != target], default=[c for c in numeric_cols(df_filtered) if c != target][:6])
            test_size = st.slider('Test set fraction', 0.05, 0.5, 0.2)
            if st.button('Train models'):
                with st.spinner('Training...'):
                    result = train_simple_models(df_filtered, target_col=target, features=features if features else None, test_size=test_size, preloaded_models=saved_models if prefer_saved else None)

                # Create ranked comparison table and chart
                kpi_df = pd.DataFrame({m: {'MAE': v[0], 'RMSE': v[1], 'R2': v[2]} for m, v in result['kpis'].items()}).T
                kpi_df = kpi_df.reset_index().rename(columns={'index':'model'}).sort_values('RMSE')
                st.subheader('Model comparison (ranked by RMSE)')
                st.dataframe(kpi_df.style.format({'MAE':'{:.3f}','RMSE':'{:.3f}','R2':'{:.3f}'}))
                fig_rmse = px.bar(kpi_df, x='model', y='RMSE', title='RMSE by model', text='RMSE')
                st.plotly_chart(fig_rmse, use_container_width=True)

                st.subheader('Select a model to use for predictions')
                selected_model_name = st.selectbox('Select model', kpi_df['model'].tolist(), index=0)
                selected_model_obj = None
                # trained baselines
                if selected_model_name in result['models']:
                    # note: result['models'] stores as {'lr':..., 'rf':...} not by display name; map names
                    if selected_model_name.lower().startswith('linear') or 'linear' in selected_model_name.lower():
                        selected_model_obj = result['models'].get('lr')
                    elif 'forest' in selected_model_name.lower() or 'random' in selected_model_name.lower():
                        selected_model_obj = result['models'].get('rf')
                # preloaded
                if selected_model_obj is None and selected_model_name in saved_models:
                    selected_model_obj = saved_models.get(selected_model_name)

                if selected_model_obj is None:
                    st.info('Selected model object not available for direct prediction. You can still inspect KPIs.')
                else:
                    st.write('Model ready for bulk prediction on the filtered dataset.')
                    if st.button('Predict on filtered dataset'):
                        # assemble X for prediction; prefer chosen features
                        X_pred = df_filtered[features] if features else df_filtered[numeric_cols(df_filtered)].drop(columns=[target], errors='ignore')
                        X_pred = X_pred.fillna(X_pred.median())
                        try:
                            preds = predict_with_model(selected_model_obj, X_pred, model_name=selected_model_name)
                            out = df_filtered.copy().reset_index(drop=True)
                            out['prediction'] = preds
                            st.subheader('Prediction sample')
                            st.dataframe(out.head(100))
                            csv = out.to_csv(index=False)
                            st.download_button('Download predictions CSV', csv, file_name=f'predictions_{selected_model_name}.csv')
                        except Exception as e:
                            st.error(f'Prediction failed: {e}')

    elif page == 'Feature Importance':
        st.header('Feature Importance')
        st.write('Train a RandomForest and inspect top features.')

        if not num_cols:
            st.warning('No numeric columns to compute feature importance.')
        else:
            target = st.selectbox('Select target (numeric)', num_cols, index=0)
            features = st.multiselect('Features to consider', [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:8])
            model_choice = st.selectbox('Model for importance', (['RandomForest', 'XGBoost'] if _HAS_XGBOOST else ['RandomForest']))
            n_estimators = st.slider('n_estimators', 10, 500, 150)
            max_depth = st.slider('max_depth (for XGBoost)', 1, 12, 6) if model_choice == 'XGBoost' else None
            if st.button('Compute importance'):
                if not features:
                    st.warning('Select at least one feature to compute importance.')
                else:
                    X = df[features].fillna(df[features].median())
                    y = df[target]
                    model = None
                    # try to reuse a saved model if it matches choice
                    for key, val in saved_models.items():
                        if model_choice.lower() in key.lower():
                            model = val
                            break
                    if model is None:
                        if model_choice == 'XGBoost' and _HAS_XGBOOST:
                            model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
                            model.fit(X, y)
                        else:
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
                            model.fit(X, y)

                    imp_df = get_feature_importances_from_saved_model(model, feature_names=features)
                    if imp_df is None or imp_df.empty:
                        st.warning('Could not extract feature importances from the model.')
                    else:
                        fig = px.bar(imp_df, x='importance', y='feature', orientation='h', title=f'Feature importance ({model_choice})')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(imp_df)

    elif page == 'Data Insights':
        st.header('Data Insights')
        st.write('Trend, anomaly detection, and categorical breakdowns.')
        st.subheader('Trend')
        date_candidates = [c for c in df.columns if 'date' in c.lower()]
        if date_candidates:
            date_col = st.selectbox('Choose date column', date_candidates)
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df_sorted = df.sort_values(date_col)
                metric = st.selectbox('Choose metric to trend', numeric_cols(df_sorted), index=0)
                fig = px.line(df_sorted, x=date_col, y=metric, title=f'{metric} over time')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning('Could not parse date column: ' + str(e))
        else:
            st.info('No date-like column detected. Showing index-based trend for a selected numeric column.')
            if num_cols:
                metric = st.selectbox('Choose metric to trend (index-based)', num_cols)
                fig = px.line(df.reset_index(), x='index', y=metric, title=f'{metric} over records')
                st.plotly_chart(fig, use_container_width=True)

        st.subheader('Anomaly Detection (simple z-score)')
        if num_cols:
            metric = st.selectbox('Choose metric for anomaly detection', num_cols, index=0)
            series = df[metric].fillna(method='ffill').fillna(method='bfill')
            z = (series - series.mean()) / (series.std() + 1e-9)
            anomalies = z.abs() > st.slider('Z-score threshold', 2.0, 5.0, 3.0)
            out_df = pd.DataFrame({'value': series, 'zscore': z, 'anomaly': anomalies})
            fig = px.scatter(out_df.reset_index(), x='index', y='value', color='anomaly', title=f'Anomalies in {metric}')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader('Categorical breakdown (pie)')
        if cat_cols:
            cat = st.selectbox('Choose categorical column for pie', cat_cols)
            counts = df[cat].value_counts().reset_index()
            counts.columns = [cat, 'count']
            fig = px.pie(counts, values='count', names=cat, title=f'Distribution of {cat}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No categorical columns detected to show pie chart.')

    elif page == 'Price Optimization':
        st.header('Price Optimization')
        st.write('Visualize profit vs. price and compute an approximate optimal price.')

        if not num_cols:
            st.warning('No numeric columns found for price optimization.')
        else:
            # choose price and profit columns
            price_col = st.selectbox('Select price column', [c for c in num_cols], index=0)
            profit_col = st.selectbox('Select profit (DCM) column', [c for c in num_cols if c != price_col] or [price_col], index=0)
            # aggregation
            agg_freq = st.selectbox('Aggregate by', ['M', 'Y'], index=0)
            # trade partner filter
            trade_partner_col = None
            if cat_cols:
                tp_default = defaults.get('trade_partner') if defaults.get('trade_partner') in cat_cols else None
                trade_partner_col = st.selectbox('Trade partner column (optional)', [None] + cat_cols, index=([None] + cat_cols).index(tp_default) if tp_default in cat_cols else 0)
            trade_partner_value = None
            if trade_partner_col:
                tp_vals = df[trade_partner_col].dropna().unique().tolist()
                if tp_vals:
                    trade_partner_value = st.selectbox('Select trade partner', ['<all>'] + tp_vals)
            st.write('If the dataset has a transactional or sku-level structure, results are per-row — aggregate as needed.')

            # allow selecting a product/category to focus optimization
            product_col = None
            if cat_cols:
                prod_default = defaults.get('sku') if defaults.get('sku') in cat_cols else None
                product_col = st.selectbox('Product identifier column (optional)', [None] + cat_cols, index=([None] + cat_cols).index(prod_default) if prod_default in cat_cols else 0)
            product_value = None
            if product_col:
                values = df[product_col].dropna().unique().tolist()
                if values:
                    # prefer selection of a detected sku value if available
                    product_value = st.selectbox('Select product', ['<all>'] + values, index=0)

            # select saved price-related models if present
            price_model_keys = [k for k in saved_models.keys() if any(t in k.lower() for t in ['price', 'profit', 'dcm', 'opt', 'xgboost', 'boost', 'quantity', 'price_opt'])]
            use_saved_or_train = 'Saved model' if price_model_keys else 'Train new model'
            if not notebook_only:
                use_saved_or_train = st.radio('Use saved model or train locally?', (['Saved model', 'Train new model'] if price_model_keys else ['Train new model']))
            else:
                # notebook-only forces saved model usage
                use_saved_or_train = 'Saved model'

            selected_saved = None
            if use_saved_or_train == 'Saved model':
                selected_saved = st.selectbox('Choose saved model for price optimization', ['<none>'] + price_model_keys)

            # price range override
            p_min = float(df[price_col].min())
            p_max = float(df[price_col].max())
            prange = st.slider('Price range to evaluate', float(p_min), float(p_max), (float(p_min), float(p_max)))

            if st.button('Compute optimization'):
                # filter dataset
                sub = df[[price_col, profit_col] + ([product_col] if product_col else [])].dropna()
                if product_col and product_value and product_value != '<all>':
                    sub = sub[sub[product_col] == product_value]

                if sub.empty:
                    st.warning('No data available for selected columns / product filter.')
                else:
                    X = sub[[price_col]].values.reshape(-1, 1)
                    y = sub[profit_col].values

                    model = None
                    # try to use selected saved model
                    if use_saved_or_train == 'Saved model' and selected_saved and selected_saved != '<none>':
                        saved_obj = saved_models.get(selected_saved)
                        if saved_obj is not None:
                            model = saved_obj
                    # if no saved model selected or notebook_only is False and user chose train
                    if model is None and use_saved_or_train == 'Train new model':
                        model = RandomForestRegressor(n_estimators=150, random_state=0)
                        model.fit(X, y)

                    # prepare evaluation grid
                    grid = np.linspace(prange[0], prange[1], 200).reshape(-1, 1)
                    preds = None
                    try:
                        # try Model predict directly on grid
                        preds = predict_with_model(model, grid, model_name=selected_saved)
                    except Exception:
                        # try predict with dataframe column name
                        try:
                            grid_df = pd.DataFrame({price_col: grid.flatten()})
                            preds = predict_with_model(model, grid_df, model_name=selected_saved)
                        except Exception as e:
                            st.error(f'Failed to predict with the selected model: {e}')
                            preds = None

                    if preds is None:
                        st.error('Prediction could not be performed with the chosen model. Try training a local model or choose a different saved model.')
                    else:
                        preds = np.asarray(preds).reshape(-1,)
                        opt_idx = int(np.argmax(preds))
                        opt_price = float(grid[opt_idx][0])
                        opt_profit = float(preds[opt_idx])

                        avg_price = float(sub[price_col].mean())
                        initial_price = float(sub[price_col].iloc[0])

                        viz_df = pd.DataFrame({'price': grid.flatten(), 'pred_profit': preds})
                        fig = px.line(viz_df, x='price', y='pred_profit', title='Predicted profit vs Price')
                        # annotate points if model supports predicting single values
                        try:
                            fig.add_scatter(x=[initial_price], y=[float(predict_with_model(model, np.array([[initial_price]]), model_name=selected_saved)[0])], mode='markers', name='Initial price', marker={'size':12, 'color':'orange'})
                        except Exception:
                            pass
                        try:
                            fig.add_scatter(x=[avg_price], y=[float(predict_with_model(model, np.array([[avg_price]]), model_name=selected_saved)[0])], mode='markers', name='Average price', marker={'size':12, 'color':'blue'})
                        except Exception:
                            pass
                        fig.add_scatter(x=[opt_price], y=[opt_profit], mode='markers', name='Optimal price', marker={'size':14, 'color':'green'})
                        st.plotly_chart(fig, use_container_width=True)

                        st.metric('Optimal price', f'{opt_price:.2f}')
                        st.metric('Predicted profit at optimal', f'{opt_profit:.2f}')

    # (Artifact Diagnostics removed to simplify UI — dashboard will train models locally by default)

    # Footer: show sample of data if requested
    if st.sidebar.checkbox('Show data sample'):
        st.subheader('Data sample')
        st.dataframe(df.head(200))


if __name__ == '__main__':
    main()
