import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

# --- 核心逻辑函数 (复用并增强) ---
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

# --- UI 界面 ---
def run_app():
    st.set_page_config(page_title="CSMAR 实证助手", layout="wide")
    st.title("📊 探索性数据分析 (EDA) 交互式平台")
    
    # 初始化变量
    df = None
    
    with st.sidebar:
        st.header("1. 数据预处理")
        uploaded_file = st.file_uploader("上传 CSMAR 数据 (CSV 或 XLSX)", type=['csv', 'xlsx'])
        winsor_pct = st.selectbox("双侧缩尾比例 (Winsorize)", [0, 0.01, 0.05], index=1)
        
        # 只有上传文件后才显示变量选择
        if uploaded_file:
            df_raw = load_data(uploaded_file)
            if df_raw is not None:
                all_cols = df_raw.columns.tolist()
                
                st.header("2. 定义变量角色")
                target_y = st.selectbox("因变量 (Y)", options=[None] + all_cols)
                main_x = st.selectbox("核心解释变量 (X)", options=[None] + all_cols)
                # 修复核心：将 all_cols 传给 multiselect
                controls = st.multiselect("控制变量 (Controls)", options=all_cols)
                
                st.markdown("---")
                st.subheader("高级变量 (用于专项诊断)")
                iv_var = st.selectbox("工具变量 (IV, 可选)", options=[None] + all_cols)
                m_var = st.selectbox("中介变量 (M, 可选)", options=[None] + all_cols)
                
                # 执行缩尾
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
                df = apply_winsorization(df_raw, numeric_cols, limits=winsor_pct) if winsor_pct > 0 else df_raw
    
    if uploaded_file and df is not None:
        tab1, tab2, tab3 = st.tabs(["📋 描述性统计", "📈 相关性分析", "🔍 专项诊断"])

        # --- Tab 1: 描述性统计 ---
        with tab1:
            analysis_vars = [v for v in [target_y, main_x] + controls if v]
            if analysis_vars:
                st.subheader("学术标准描述性统计表")
                subset = df[analysis_vars]
                stats = subset.describe(percentiles=[.25, .5, .75]).T
                stats = stats.rename(columns={'count': 'N', 'mean': 'Mean', 'std': 'SD', '50%': 'Median'})
                stats['Skewness'] = subset.apply(lambda x: skew(x.dropna()))
                
                st.dataframe(stats.style.format("{:.3f}"))
                
                # 智能提示
                for var in analysis_vars:
                    if abs(skew(df[var].dropna())) > 1:
                        st.warning(f"💡 变量 **{var}** 偏度过高，实证研究中通常建议对其取对数。")
            else:
                st.info("请在左侧选择 Y 和 X 变量。")

        # --- Tab 2: 相关性分析 ---
        with tab2:
            if target_y and main_x:
                col1, col2 = st.columns([2, 1])
                with col1:
                    # 散点图增加 95% 置信区间
                    fig = px.scatter(df, x=main_x, y=target_y, trendline="ols", 
                                   title=f"{main_x} 与 {target_y} 的线性关系及95%置信区间",
                                   template="simple_white", opacity=0.5)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📝 自动分析报告")
                    # 计算相关性
                    valid_df = df[[main_x, target_y]].dropna()
                    r, p = pearsonr(valid_df[main_x], valid_df[target_y])
                    
                    st.write(f"- **Pearson系数**: `{r:.3f}`")
                    st.write(f"- **P值**: `{p:.3f}`")
                    
                    if p < 0.05:
                        res = "正相关" if r > 0 else "负相关"
                        st.success(f"结论：两者在 5% 水平上显著{res}。初步支撑研究假设。")
                    else:
                        st.error("结论：两者相关性不显著。请检查是否存在非线性关系或样本量不足。")
            else:
                st.warning("请先指定 Y 和 X 变量。")

        # --- Tab 3: 专项诊断 ---
        with tab3:
            st.subheader("学术专项诊断报告")
            
            # IV 诊断
            if iv_var and main_x:
                st.markdown("#### 1. 工具变量 (IV) 强度检验")
                data = df[[iv_var, main_x]].dropna()
                model = sm.OLS(data[main_x], sm.add_constant(data[iv_var])).fit()
                f_stat = model.fvalue
                st.metric("第一阶段 F 统计量", f"{f_stat:.2f}")
                if f_stat < 10:
                    st.error("⚠️ F < 10：存在**弱工具变量**风险，IV 与 X 的相关性不足。")
                else:
                    st.success("✅ F > 10：初步排除了弱工具变量问题。")
            
            # 中介分析提示
            if m_var and main_x and target_y:
                st.markdown("#### 2. 中介效应 (Mediation) 初探")
                st.info(f"正在分析路径：{main_x} ➔ {m_var} ➔ {target_y}")
                r1, _ = pearsonr(df[main_x].dropna(), df[m_var].dropna())
                r2, _ = pearsonr(df[m_var].dropna(), df[target_y].dropna())
                st.write(f"- 路径 A ({main_x}➔{m_var}) 相关性: `{r1:.3f}`")
                st.write(f"- 路径 B ({m_var}➔{target_y}) 相关性: `{r2:.3f}`")
            
            if not iv_var and not m_var:
                st.info("在左侧侧边栏选择 **工具变量** 或 **中介变量** 后，此处将自动显示学术检验结果。")

    else:
        st.info("👋 欢迎！请在左侧上传 CSMAR 数据文件并定义变量角色开始分析。")

if __name__ == "__main__":
    run_app()
