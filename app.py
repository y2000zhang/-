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

# --- 增强：计算多重共线性 VIF ---
def check_vif(df, variables):
    if len(variables) < 2: return None
    data = df[variables].dropna()
    # 增加常数项
    X = sm.add_constant(data)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data[vif_data["feature"] != 'const']

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
        # 在原代码导入部分增加
        
# --- UI 增强 ---
# 在 run_app() 的 Tab 2 增加内容        
        with tab2:
            if target_y and main_x:
                # 1. 保持原有的散点图
                st.subheader("一、核心关系探索")
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_scatter = px.scatter(df, x=main_x, y=target_y, trendline="ols", 
                                           marginal_y="box", # 侧边增加箱线图
                                           title=f"{main_x} 与 {target_y} 的分布与趋势",
                                           opacity=0.3)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # 2. 增加：全变量相关性热力图
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
                        st.markdown("#### 🛡️ 多重共线性 (VIF) 诊断")
                        # 排除非数值型后计算
                        num_vars = df[all_selected].select_dtypes(include=[np.number]).columns.tolist()
                        vif_res = check_vif(df, num_vars)
                        if vif_res is not None:
                            st.dataframe(vif_res.style.format({"VIF": "{:.2f}"}))
                            # 诊断标准
                            max_vif = vif_res['VIF'].max()
                            if max_vif > 10:
                                st.error(f"警告：最大 VIF ({max_vif:.2f}) > 10，存在严重共线性风险！")
                            elif max_vif > 5:
                                st.warning("提示：存在中度共线性风险 (VIF > 5)。")
                            else:
                                st.success("共线性诊断通过：所有变量 VIF 均处于安全范围。")
        
                # 3. 增加：变量对比箱线图（检查缩尾效果）
                st.markdown("---")
                st.subheader("三、变量结构分析")
                fig_box = px.box(df[all_selected], orientation="h", title="变量分布箱线图 (用于识别异常值)")
                st.plotly_chart(fig_box, use_container_width=True)
                st.info("💡 箱线图说明：若缩尾后仍存在大量远距离离群点，建议在实证模型中对该变量进行 Log 处理或更严格的缩尾。")

        
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
