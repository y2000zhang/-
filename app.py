import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. 核心逻辑函数 ---
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

def apply_winsorization(df, columns, limits=0.01):
    """对指定列进行双侧缩尾处理"""
    df_winsorized = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_winsorized[col]):
            # 确保列不为空
            if df_winsorized[col].notnull().any():
                valid_mask = df_winsorized[col].notnull()
                df_winsorized.loc[valid_mask, col] = winsorize(
                    df_winsorized.loc[valid_mask, col], 
                    limits=(limits, limits)
                )
    return df_winsorized

def check_vif(df, variables):
    """计算多重共线性 VIF"""
    if len(variables) < 2: 
        return None
    # 必须剔除缺失值否则 VIF 无法计算
    data = df[variables].dropna()
    if data.empty or len(data) < len(variables):
        return None
    try:
        X = sm.add_constant(data)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data[vif_data["feature"] != 'const']
    except:
        return None

# --- 2. UI 界面逻辑 ---
def run_app():
    st.set_page_config(page_title="CSMAR 实证助手", layout="wide")
    st.title("📊 探索性数据分析 (EDA) 交互式平台")
    st.markdown("*量身定制的金融/管理实证研究数据探索工具*")
    
    df = None
        
    with st.sidebar:
        st.header("1. 数据预处理")
        uploaded_file = st.file_uploader("上传 CSMAR/Wind 数据 (CSV 或 XLSX)", type=['csv', 'xlsx'])
        winsor_pct = st.selectbox("双侧缩尾比例 (Winsorize)", [0, 0.01, 0.05], index=1, 
                                 help="建议实证研究使用 1% 缩尾以消除极端值")
        
        if uploaded_file:
            df_raw = load_data(uploaded_file)
            if df_raw is not None:
                all_cols = df_raw.columns.tolist()
                
                st.header("2. 定义变量角色")
                target_y = st.selectbox("因变量 (Y)", options=[None] + all_cols)
                main_x = st.selectbox("核心解释变量 (X)", options=[None] + all_cols)
                controls = st.multiselect("控制变量 (Controls)", options=all_cols)
                
                st.markdown("---")
                st.subheader("高级变量 (用于专项诊断)")
                iv_var = st.selectbox("工具变量 (IV, 可选)", options=[None] + all_cols)
                m_var = st.selectbox("中介变量 (M, 可选)", options=[None] + all_cols)
                
                # 执行预处理
                numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
                if winsor_pct > 0:
                    df = apply_winsorization(df_raw, numeric_cols, limits=winsor_pct)
                    st.sidebar.success(f"已完成 {winsor_pct*100}% 缩尾处理")
                else:
                    df = df_raw.copy()
    
    if uploaded_file and df is not None:
        tab1, tab2, tab3 = st.tabs(["📋 描述性统计", "📈 相关性与分布", "🔍 专项诊断"])

        # --- Tab 1: 描述性统计 ---
        with tab1:
            analysis_vars = [v for v in [target_y, main_x] + controls if v]
            if analysis_vars:
                st.subheader("学术标准描述性统计表")
                subset = df[analysis_vars]
                stats = subset.describe(percentiles=[.25, .5, .75]).T
                stats = stats.rename(columns={'count': 'N', 'mean': 'Mean', 'std': 'SD', '50%': 'Median'})
                # 计算偏度
                stats['Skewness'] = subset.apply(lambda x: skew(x.dropna()))
                
                st.dataframe(stats.style.format("{:.3f}").highlight_null(color='red'))
                
                # 智能诊断提示
                for var in analysis_vars:
                    if abs(skew(df[var].dropna())) > 1:
                        st.warning(f"💡 变量 **{var}** 偏度过高 ({skew(df[var].dropna()):.2f})，建议实证中取对数 (Log) 处理。")
            else:
                st.info("请在左侧侧边栏指定 Y 和 X 变量以生成统计表。")

        # --- Tab 2: 相关性与分布 --- 
        with tab2:
            if target_y and main_x:
                # 1. 核心关系散点图
                st.subheader("一、核心回归关系探索")
                fig_scatter = px.scatter(df, x=main_x, y=target_y, trendline="ols", 
                                       marginal_y="violin", 
                                       title=f"{main_x} ➔ {target_y} 拟合趋势图",
                                       opacity=0.4)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 2. 相关性矩阵与 VIF
                st.markdown("---")
                st.subheader("二、相关性矩阵与多重共线性")
                all_selected = [v for v in [target_y, main_x] + controls if v]
                
                if len(all_selected) > 1:
                    col_heat, col_vif = st.columns([1, 1])
                    with col_heat:
                        corr_matrix = df[all_selected].corr()
                        fig_heat = px.imshow(corr_matrix, text_auto=".2f", 
                                           color_continuous_scale='RdBu_r', range_color=[-1,1],
                                           title="Pearson 相关系数矩阵")
                        st.plotly_chart(fig_heat, use_container_width=True)
                    
                    with col_vif:
                        st.markdown("#### 🛡️ VIF 共线性诊断")
                        num_vars = df[all_selected].select_dtypes(include=[np.number]).columns.tolist()
                        vif_res = check_vif(df, num_vars)
                        if vif_res is not None:
                            st.dataframe(vif_res.style.format({"VIF": "{:.2f}"}))
                            max_vif = vif_res['VIF'].max()
                            if max_vif > 10: st.error(f"严重风险：最大 VIF ({max_vif:.2f}) > 10")
                            elif max_vif > 5: st.warning("中度风险：VIF > 5")
                            else: st.success("诊断通过：无严重多重共线性")

                # 3. 独立坐标轴箱线图 (优化量纲差异)
                st.markdown("---")
                st.subheader("三、变量结构分析 (独立坐标轴)")
                st.info("💡 每个变量使用独立坐标轴展示，方便观察不同量纲下的离群点分布。")
                
                num_cols = 2
                for i in range(0, len(all_selected), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        if i + j < len(all_selected):
                            var_name = all_selected[i + j]
                            with cols[j]:
                                fig_single = px.box(df, x=var_name, orientation="h", 
                                                  title=f"变量 {var_name} 分布",
                                                  color_discrete_sequence=['#1f77b4'])
                                fig_single.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
                                st.plotly_chart(fig_single, use_container_width=True)
            else:
                st.info("请先设置核心变量 X 和 Y。")

        # --- Tab 3: 专项诊断 ---
        with tab3:
            st.subheader("学术专项诊断报告")
            
            # IV 诊断
            if iv_var and main_x:
                st.markdown("#### 1. 工具变量 (IV) 强度检验")
                data_iv = df[[iv_var, main_x]].dropna()
                if not data_iv.empty:
                    model_iv = sm.OLS(data_iv[main_x], sm.add_constant(data_iv[iv_var])).fit()
                    st.metric("第一阶段 F 统计量", f"{model_iv.fvalue:.2f}")
                    if model_iv.fvalue < 10:
                        st.error("⚠️ 弱工具变量风险：F 统计量小于 10。")
                    else:
                        st.success("✅ 通过检验：工具变量相关性强度达标。")
            
            # 中介分析提示
            if m_var and main_x and target_y:
                st.markdown("#### 2. 中介效应 (Mediation) 路径初探")
                st.info(f"路径：{main_x} (X) ➔ {m_var} (M) ➔ {target_y} (Y)")
                data_m = df[[main_x, m_var, target_y]].dropna()
                if not data_m.empty:
                    r_xm, _ = pearsonr(data_m[main_x], data_m[m_var])
                    r_my, _ = pearsonr(data_m[m_var], data_m[target_y])
                    st.write(f"- 路径 A (X➔M) 相关性: `{r_xm:.3f}`")
                    st.write(f"- 路径 B (M➔Y) 相关性: `{r_my:.3f}`")
            
            if not iv_var and not m_var:
                st.info("在左侧侧边栏选择工具变量 (IV) 或中介变量 (M) 即可开启诊断。")

    else:
        st.info("👋 欢迎！请在左侧上传数据集并定义变量角色开始分析。")

if __name__ == "__main__":
    run_app()
