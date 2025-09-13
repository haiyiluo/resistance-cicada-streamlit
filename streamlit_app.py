"""
ResistanceCascade Streamlit Web App
这是你的模型的网页版本
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# 导入你的模型
# 确保resistance_cascade文件夹在同一目录下
from resistance_cascade.model import ResistanceCascade

# 设置页面配置
st.set_page_config(
    page_title="抵抗级联模型",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .stButton > button {
        background-color: #667eea;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton > button:hover {
        background-color: #764ba2;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 标题和介绍
st.title("🔥 抵抗级联互动模型")
st.markdown("""
这个模型展示了抗议活动如何在人群中传播。你可以调整参数来探索不同条件下的级联动态。
""")

# 侧边栏 - 参数设置
st.sidebar.header("⚙️ 模型参数设置")

# 基本参数
st.sidebar.subheader("基本参数")

# 使用列来并排显示参数
col1, col2 = st.sidebar.columns(2)

with col1:
    width = st.number_input("网格宽度", min_value=20, max_value=100, value=40, step=10)
    citizen_density = st.slider("公民密度", 0.1, 0.9, 0.7, 0.05)
    seed = st.number_input("随机种子", min_value=0, max_value=99999, value=42)

with col2:
    height = st.number_input("网格高度", min_value=20, max_value=100, value=40, step=10)
    max_iters = st.slider("最大步数", 100, 1000, 500, 50)
    
# 关键参数
st.sidebar.subheader("🎯 关键参数")

epsilon = st.sidebar.slider(
    "信息不确定性 (ε)", 
    min_value=0.1, 
    max_value=1.5, 
    value=0.5, 
    step=0.1,
    help="人们对实际情况的误判程度。值越高，信息越不准确。"
)

security_density = st.sidebar.slider(
    "安全部队密度", 
    min_value=0.0, 
    max_value=0.1, 
    value=0.02, 
    step=0.01,
    help="警察/安全部队在人群中的比例。"
)

pp_mean = st.sidebar.slider(
    "私人偏好均值", 
    min_value=-1.0, 
    max_value=0.0, 
    value=-0.5, 
    step=0.1,
    help="平均而言，人们内心支持(-1)还是反对(0)政权。"
)

threshold = st.sidebar.slider(
    "激活阈值", 
    min_value=2.0, 
    max_value=5.0, 
    value=3.5, 
    step=0.1,
    help="人们参与抗议需要的鼓励程度。"
)

# 添加参数说明
with st.sidebar.expander("❓ 参数说明"):
    st.write("""
    - **信息不确定性**：模拟信息管制和谣言的影响
    - **安全部队密度**：镇压能力的强弱
    - **私人偏好**：民众的真实态度
    - **激活阈值**：参与的心理门槛
    """)

# 主界面布局
tab1, tab2, tab3 = st.tabs(["🚀 运行模拟", "📊 历史记录", "📖 模型说明"])

with tab1:
    # 运行按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 开始模拟", type="primary", use_container_width=True)
    
    if run_button:
        # 创建占位符
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # 运行模型
        status_placeholder.info("🔄 正在初始化模型...")
        
        # 创建模型实例
        model = ResistanceCascade(
            width=width,
            height=height,
            citizen_density=citizen_density,
            security_density=security_density,
            epsilon=epsilon,
            private_preference_distribution_mean=pp_mean,
            threshold=threshold,
            seed=seed,
            max_iters=max_iters,
            multiple_agents_per_cell=True
        )
        
        # 收集数据
        time_steps = []
        active_counts = []
        support_counts = []
        oppose_counts = []
        jail_counts = []
        
        # 运行模拟
        start_time = time.time()
        step = 0
        
        while model.running and step < max_iters:
            model.step()
            
            # 记录数据
            time_steps.append(step)
            active_counts.append(model.active_count)
            support_counts.append(model.support_count)
            oppose_counts.append(model.oppose_count)
            jail_counts.append(model.count_jail(model))
            
            # 更新进度
            progress = (step + 1) / max_iters
            progress_bar.progress(progress)
            
            # 更新状态
            if step % 10 == 0:
                status_placeholder.info(f"🔄 模拟进行中... 步骤 {step}/{max_iters}")
            
            step += 1
        
        # 模拟完成
        simulation_time = time.time() - start_time
        status_placeholder.success(f"✅ 模拟完成！用时 {simulation_time:.2f} 秒")
        progress_bar.empty()
        
        # 显示结果
        st.subheader("📊 模拟结果")
        
        # 关键指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "革命发生", 
                "是 ✅" if model.revolution else "否 ❌",
                delta="成功" if model.revolution else "失败"
            )
        
        with col2:
            max_participation = max(active_counts) / model.citizen_count * 100
            st.metric(
                "最高参与率", 
                f"{max_participation:.1f}%",
                delta=f"{max_participation - 5:.1f}%" if max_participation > 5 else None
            )
        
        with col3:
            st.metric(
                "达到峰值时间", 
                f"第 {active_counts.index(max(active_counts))} 步",
                delta="快速" if active_counts.index(max(active_counts)) < 50 else "缓慢"
            )
        
        with col4:
            st.metric(
                "最终活跃人数", 
                f"{model.active_count} 人",
                delta=f"{model.active_count - model.citizen_count * 0.1:.0f}"
            )
        
        # 动态图表
        st.subheader("📈 人群动态变化")
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：人群状态变化
        ax1.plot(time_steps, active_counts, label='活跃抗议者', color='#FE6100', linewidth=2)
        ax1.plot(time_steps, support_counts, label='支持者', color='#648FFF', linewidth=2)
        ax1.plot(time_steps, oppose_counts, label='反对者', color='#A020F0', linewidth=2)
        ax1.plot(time_steps, jail_counts, label='被捕者', color='#000000', linewidth=2, linestyle='--')
        
        ax1.set_xlabel('时间步')
        ax1.set_ylabel('人数')
        ax1.set_title('人群状态演变')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 右图：参与率变化
        participation_rate = [a / model.citizen_count * 100 for a in active_counts]
        ax2.fill_between(time_steps, participation_rate, alpha=0.3, color='#FE6100')
        ax2.plot(time_steps, participation_rate, color='#FE6100', linewidth=2)
        ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% 临界线')
        
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('参与率 (%)')
        ax2.set_title('抗议参与率变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 数据表格
        with st.expander("📊 查看详细数据"):
            # 创建数据框
            df = pd.DataFrame({
                '时间步': time_steps[::10],  # 每10步显示一次
                '活跃人数': active_counts[::10],
                '支持人数': support_counts[::10],
                '反对人数': oppose_counts[::10],
                '被捕人数': jail_counts[::10],
                '参与率(%)': [f"{x:.1f}" for x in participation_rate[::10]]
            })
            st.dataframe(df, use_container_width=True)
            
            # 下载按钮
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 下载数据 (CSV)",
                data=csv,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # 保存到session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({
            'timestamp': datetime.now(),
            'params': {
                'epsilon': epsilon,
                'security_density': security_density,
                'pp_mean': pp_mean,
                'threshold': threshold
            },
            'results': {
                'revolution': model.revolution,
                'max_participation': max_participation,
                'peak_time': active_counts.index(max(active_counts)),
                'final_active': model.active_count
            }
        })

with tab2:
    st.subheader("📊 历史运行记录")
    
    if 'history' in st.session_state and st.session_state.history:
        # 转换为数据框
        history_data = []
        for h in st.session_state.history:
            record = {
                '时间': h['timestamp'].strftime('%H:%M:%S'),
                'ε': h['params']['epsilon'],
                '安全密度': h['params']['security_density'],
                '私人偏好': h['params']['pp_mean'],
                '阈值': h['params']['threshold'],
                '革命': '✅' if h['results']['revolution'] else '❌',
                '最高参与率': f"{h['results']['max_participation']:.1f}%",
                '峰值时间': h['results']['peak_time']
            }
            history_data.append(record)
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # 参数对比图
        if len(history_data) > 1:
            st.subheader("📈 参数影响分析")
            
            # 选择要分析的参数
            param_to_analyze = st.selectbox(
                "选择要分析的参数",
                ['ε', '安全密度', '私人偏好', '阈值']
            )
            
            # 创建散点图
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_data = [h['params']['epsilon'] if param_to_analyze == 'ε' 
                     else h['params']['security_density'] if param_to_analyze == '安全密度'
                     else h['params']['pp_mean'] if param_to_analyze == '私人偏好'
                     else h['params']['threshold'] 
                     for h in st.session_state.history]
            
            y_data = [h['results']['max_participation'] for h in st.session_state.history]
            colors = ['green' if h['results']['revolution'] else 'red' 
                     for h in st.session_state.history]
            
            scatter = ax.scatter(x_data, y_data, c=colors, s=100, alpha=0.6)
            ax.set_xlabel(param_to_analyze)
            ax.set_ylabel('最高参与率 (%)')
            ax.set_title(f'{param_to_analyze} 对参与率的影响')
            ax.grid(True, alpha=0.3)
            
            # 添加图例
            green_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='g', markersize=10, label='革命成功')
            red_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='r', markersize=10, label='革命失败')
            ax.legend(handles=[green_patch, red_patch])
            
            st.pyplot(fig)
        
        # 清除历史记录按钮
        if st.button("🗑️ 清除历史记录"):
            st.session_state.history = []
            st.experimental_rerun()
    else:
        st.info("还没有运行记录。运行一些模拟后，这里会显示历史数据。")

with tab3:
    st.subheader("📖 模型说明")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 模型原理
        
        这是一个**基于主体的模型(Agent-Based Model)**，模拟个体如何决定是否参与抗议：
        
        1. **个体决策**：每个人根据周围人的行为和自己的偏好决定
        2. **信息传播**：人们观察周围的情况，但信息可能不准确
        3. **镇压风险**：安全部队会逮捕活跃的抗议者
        4. **级联效应**：当足够多的人参与时，会引发连锁反应
        
        ### 关键机制
        
        - **阈值模型**：人们有不同的参与门槛
        - **空间相互作用**：只能看到附近的人
        - **动态演化**：情况随时间不断变化
        """)
    
    with col2:
        st.markdown("""
        ### 参数详解
        
        **信息不确定性 (ε)**
        - 低值（0.1-0.3）：信息透明，人们了解真实情况
        - 中值（0.4-0.7）：存在一定误判
        - 高值（0.8-1.5）：严重的信息混乱
        
        **安全部队密度**
        - 0-1%：几乎没有镇压
        - 2-4%：中等镇压
        - 5-10%：强力镇压
        
        **私人偏好均值**
        - -1.0：强烈反对政权
        - -0.5：温和反对
        - 0.0：中立
        
        **激活阈值**
        - 2-3：容易被动员
        - 3-4：需要一定鼓励
        - 4-5：很难被动员
        """)
    
    st.markdown("---")
    
    # 添加一些使用提示
    st.info("""
    💡 **使用提示**:
    - 尝试不同的参数组合，观察结果的变化
    - 特别注意"临界点"现象 - 小的参数变化可能导致完全不同的结果
    - 革命成功通常需要：低镇压 + 高不满 + 适度的信息不确定性
    """)
    
    # 联系信息
    st.markdown("---")
    st.markdown("🔗 [返回主网站](https://yourusername.github.io) | 📧 联系: your-email@example.com")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>基于 Agent-Based Model (ABM) 技术 | 使用 Streamlit 构建</p>
    </div>
    """, 
    unsafe_allow_html=True
)