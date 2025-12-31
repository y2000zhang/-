import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import winsorize, skew, kurtosis, pearsonr
import statsmodels.api as sm
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# --- 1. 数据处理核心函数 ---
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None

def check_missing_values(df):
    missing_pct = df.isnull().mean()
    warning_vars = missing_pct[missing_pct > 0.1]
    return missing_pct, warning_vars

def apply_winsorization(df, columns, limits=0.01):
    df_winsorized = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_winsorized[col]):
            valid_mask = df_winsorized[col].notnull()
            df_winsorized.loc[valid_mask, col] = winsorize(
                df_winsorized.loc[valid_mask, col], 
                limits=(limits, limits)
            )
    return df_winsorized

def get_descriptive_stats(df, selected_columns):
    subset = df[selected_columns]
    stats = subset.describe(percentiles=[.25, .5, .75]).T
    stats = stats.rename(columns={
        'count': 'N', 'mean': 'Mean', 'std': 'SD', 
        'min': 'Min', '25%': 'P25', '50%': 'Median', 
        '75%': 'P75', 'max': 'Max'
    })
    stats['Skewness'] = subset.apply(lambda x: skew(x.dropna()))
    stats['Kurtosis'] = subset.apply(lambda x: kurtosis(x.dropna()))
    return stats

def diag_iv_strength(df, iv, x):
    data = df[[iv, x]].dropna()
    X_iv = sm.add_constant(data[iv])
    model = sm.OLS(data[x], X_iv).fit()
    return model.fvalue, model.params[1], model.pvalues[1]

def diag_skewness_suggestion(stats_df):
    suggestions = []
    for var, row in stats_df.iterrows():
        if abs(row['Skewness']) > 1:
            suggestions.append(f"⚠️ 变量 **{var}** 分布严重偏态 (Skew={row['Skewness']:.2f})，建议取对数 (Log) 处理。")
    return suggestions

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Descriptive_Stats')
    return output.getvalue()

# --- 2. Streamlit 界面逻辑 ---
def run_app():
    st.set_page_config(page_title="CSMAR 实证助手", layout="wide")
    st.title("📊 探索性数据分析 (EDA) 交互式平台")
    
    with st.sidebar:
        st.header("1. 数据预处理")
        uploaded_file = st.file_uploader("上传 CSMAR 数据 (CSV 或 XLSX)", type=['csv', 'xlsx'])
        winsor_pct = st.selectbox("双侧缩尾比例 (Winsorize)", [0, 0.01, 0.05], index=1)
        
        st.header("2. 定义变量角色")
        target_y = st.text_input("因变量 (Y)")
        main_x = st.text_input("核心解释变量 (X)")
        controls = st.multiselect("控制变量 (Controls)", [])
        iv_var = st.text_input("工具变量 (IV, 可选)")

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if winsor_pct > 0:
                df = apply_winsorization(df, numeric_cols, limits=winsor_pct)
            
            missing_pct, _ = check_missing_values(df)
            tab1, tab2, tab3 = st.tabs(["📋 描述性统计", "📈 相关性分析", "🔍 专项诊断"])

            with tab1:
                selected_vars = [v for v in [target_y, main_x] + controls if v]
                if selected_vars:
                    desc_df = get_descriptive_stats(df, selected_vars)
                    st.dataframe(desc_df.style.format("{:.3f}"))
                    for sug in diag_skewness_suggestion(desc_df):
                        st.info(sug)
                    st.download_button("导出 Excel", data=to_excel(desc_df), file_name="stats.xlsx")

            with tab2:
                if target_y and main_x:
                    fig_scatter = px.scatter(df, x=main_x, y=target_y, trendline="ols")
                    st.plotly_chart(fig_scatter)

            with tab3:
                if iv_var and main_x:
                    f_stat, _, _ = diag_iv_strength(df, iv_var, main_x)
                    st.metric("第一阶段 F 统计量", f"{f_stat:.2f}")

    else:
        st.info("请在左侧上传数据文件。")

if __name__ == "__main__":
    run_app()
