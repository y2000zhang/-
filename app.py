import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import winsorize, skew, kurtosis, pearsonr
import statsmodels.api as sm
from io import BytesIO

# --- 1. 数据加载与基础清洗 ---
def load_data(uploaded_file):
    """支持 CSV 和 Excel 加载"""
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
    """缺失值预警逻辑"""
    missing_pct = df.isnull().mean()
    warning_vars = missing_pct[missing_pct > 0.1]
    return missing_pct, warning_vars

# --- 2. 金融实证核心：缩尾处理 (Winsorize) ---
def apply_winsorization(df, columns, limits=0.01):
    """
    对指定列进行双侧缩尾处理
    limits: 0.01 表示 [1%, 99%] 缩尾
    """
    df_winsorized = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_winsorized[col]):
            # 过滤掉缺失值后再进行缩尾，保持索引一致
            valid_mask = df_winsorized[col].notnull()
            df_winsorized.loc[valid_mask, col] = winsorize(
                df_winsorized.loc[valid_mask, col], 
                limits=(limits, limits)
            )
    return df_winsorized

# --- 3. 学术标准描述性统计 ---
def get_descriptive_stats(df, selected_columns):
    """
    生成符合学术期刊标准的描述性统计表
    包含：N, Mean, SD, Min, P25, Median, P75, Max, Skew, Kurtosis
    """
    subset = df[selected_columns]
    stats = subset.describe(percentiles=[.25, .5, .75]).T
    
    # 重新命名和添加指标
    stats = stats.rename(columns={
        'count': 'N', 'mean': 'Mean', 'std': 'SD', 
        'min': 'Min', '25%': 'P25', '50%': 'Median', 
        '75%': 'P75', 'max': 'Max'
    })
    
    # 计算偏度和峰度
    stats['Skewness'] = subset.apply(lambda x: skew(x.dropna()))
    stats['Kurtosis'] = subset.apply(lambda x: kurtosis(x.dropna()))
    
    return stats

# --- 4. 专项诊断逻辑 ---
def diag_iv_strength(df, iv, x):
    """
    IV专项：计算第一阶段 F 统计量初步判断弱工具变量
    """
    data = df[[iv, x]].dropna()
    X_iv = sm.add_constant(data[iv])
    model = sm.OLS(data[x], X_iv).fit()
    f_stat = model.fvalue
    return f_stat, model.params[1], model.pvalues[1]

def diag_skewness_suggestion(stats_df):
    """偏度诊断建议"""
    suggestions = []
    for var, row in stats_df.iterrows():
        if abs(row['Skewness']) > 1:
            suggestions.append(f"⚠️ 变量 **{var}** 分布严重偏态 (Skew={row['Skewness']:.2f})，建议取对数 (Log) 处理。")
    return suggestions

# --- 5. 导出功能 ---
def to_excel(df):
    """将 DataFrame 转换为 Excel 二进制流"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=True, sheet_name='Descriptive_Stats')
    processed_data = output.getvalue()
    return processed_data
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 假设第一部分的函数已载入

def run_app():
    st.set_page_config(page_title="CSMAR 实证助手 - EDA专家", layout="wide")
    
    st.title("📊 探索性数据分析 (EDA) 交互式平台")
    st.markdown("""
    *本模块专为管理科学与金融实证研究设计，支持 CSMAR/Wind 数据格式，集成缩尾处理与学术级诊断。*
    """)

    # --- 侧边栏：文件上传与参数设置 ---
    with st.sidebar:
        st.header("1. 数据预处理")
        uploaded_file = st.file_uploader("上传 CSMAR 数据 (CSV 或 XLSX)", type=['csv', 'xlsx'])
        
        winsor_pct = st.selectbox("双侧缩尾比例 (Winsorize)", [0, 0.01, 0.05], index=1, 
                                 help="金融实证通常使用 1% 缩尾以消除极端值影响")
        
        st.header("2. 定义变量角色")
        target_y = st.text_input("因变量 (Y)", placeholder="例如: ROA")
        main_x = st.text_input("核心解释变量 (X)", placeholder="例如: Digital_Index")
        controls = st.multiselect("控制变量 (Controls)", [])
        iv_var = st.text_input("工具变量 (IV, 可选)")
        m_var = st.text_input("中介变量 (M, 可选)")
        id_var = st.text_input("个体ID (如 Stkcd)")
        time_var = st.text_input("时间变量 (如 Year)")

    if uploaded_file:
        df = load_data(uploaded_file)
        # 更新控制变量可选列表
        all_cols = df.columns.tolist()
        
        # --- 数据预处理逻辑 ---
        # 1. 缩尾处理
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if winsor_pct > 0:
            df = apply_winsorization(df, numeric_cols, limits=winsor_pct)
            st.sidebar.success(f"已完成 {winsor_pct*100}% 缩尾处理")

        # 2. 缺失值预警
        missing_pct, warning_vars = check_missing_values(df)
        
        # --- 右侧主界面布局 ---
        tab1, tab2, tab3 = st.tabs(["📋 描述性统计", "📈 分布与相关性", "🔍 专项诊断 (IV/M)"])

        # Tab 1: 描述性统计
        with tab1:
            st.subheader("学术标准描述性统计表")
            selected_vars = [v for v in [target_y, main_x] + controls if v]
            if selected_vars:
                desc_df = get_descriptive_stats(df, selected_vars)
                st.dataframe(desc_df.style.format("{:.3f}").highlight_null(color='red'))
                
                # 诊断提示
                suggestions = diag_skewness_suggestion(desc_df)
                for sug in suggestions:
                    st.info(sug)
                
                # 缺失值红色预警
                for var in selected_vars:
                    if missing_pct[var] > 0.1:
                        st.error(f"❌ **{var}** 缺失值比例为 {missing_pct[var]:.2%}: **可能存在样本选择偏差！**")
                
                # 下载按钮
                st.download_button("导出 Excel 学术表", data=to_excel(desc_df), 
                                 file_name="Descriptive_Stats.xlsx", mime="application/vnd.ms-excel")
            else:
                st.warning("请在左侧侧边栏指定变量名以生成统计表。")

        # Tab 2: 交互式可视化
        with tab2:
            if target_y and main_x:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{target_y} 分布图 (含KDE)**")
                    fig_y = px.histogram(df, x=target_y, marginal="rug", kde=True, 
                                       title=f"{target_y} 分布特征", color_discrete_sequence=['#1f77b4'])
                    st.plotly_chart(fig_y, use_container_width=True)
                
                with col2:
                    st.write(f"**{main_x} 与 {target_y} 拟合关系**")
                    # 大数据优化处理
                    opacity = 0.3 if len(df) > 2000 else 0.7
                    fig_scatter = px.scatter(df, x=main_x, y=target_y, trendline="ols",
                                           opacity=opacity, title="核心变量散点图及 95% 置信区间")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # 计算相关系数
                    corr_val, p_val = pearsonr(df[main_x].dropna(), df[target_y].dropna())
                    sig_text = "显著" if p_val < 0.05 else "不显著"
                    st.write(f"Pearson相关系数: **{corr_val:.3f}** (p={p_val:.3f}, {sig_text})")

                # 相关性热力图
                st.write("**变量相关性矩阵 (Heatmap)**")
                if len(selected_vars) > 1:
                    corr_matrix = df[selected_vars].corr()
                    fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                       color_continuous_scale='RdBu_r', range_color=[-1,1])
                    st.plotly_chart(fig_heat, use_container_width=True)
                    
                    # 共线性检查
                    high_corr = (corr_matrix.abs() > 0.7) & (corr_matrix != 1.0)
                    if high_corr.any().any():
                        st.warning("⚠️ 警告：检测到变量间相关系数 > 0.7，可能存在多重共线性风险。")

        # Tab 3: IV/中介专项
        with tab3:
            if iv_var and main_x:
                st.subheader("工具变量 (IV) 第一阶段强度初探")
                f_stat, coef, p = diag_iv_strength(df, iv_var, main_x)
                st.metric("第一阶段 F 统计量", f"{f_stat:.2f}")
                if f_stat < 10:
                    st.error("💡 F < 10: 存在弱工具变量风险 (Weak IV Instrument)。")
                else:
                    st.success("💡 F > 10: 工具变量通过弱识别初步检验。")
                
            if m_var and main_x and target_y:
                st.subheader("中介效应 (Mediation) 路径初探")
                st.info(f"路径预览: {main_x} ➔ {m_var} ➔ {target_y}")
                # 此处可进一步添加简化的路径图绘制

    else:
        st.info("👋 请上传数据集开始分析。建议首先检查 Stkcd 和 Year 的格式。")

if __name__ == "__main__":
    run_app()
