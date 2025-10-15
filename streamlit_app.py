"""
ResistanceCascade Streamlit Web App
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import your model
# Make sure the resistance_cascade folder is in the same directory
from resistance_cascade.model import ResistanceCascade

# Set page configuration
st.set_page_config(
    page_title="Resistance Cascade Model",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
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

# Title and introduction
st.title("🔥 Resistance Cascade Interactive Model")
st.markdown("""
This model demonstrates how protest activities spread through populations. You can adjust parameters to explore cascade dynamics under different conditions.
""")

# Sidebar - Parameter settings
st.sidebar.header("⚙️ Model Parameter Settings")

# Basic parameters
st.sidebar.subheader("Basic Parameters")

# Use columns to display parameters side by side
col1, col2 = st.sidebar.columns(2)

with col1:
    width = st.number_input("Grid Width", min_value=20, max_value=100, value=40, step=10)
    citizen_density = st.slider("Citizen Density", 0.1, 0.9, 0.7, 0.05)
    seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)

with col2:
    height = st.number_input("Grid Height", min_value=20, max_value=100, value=40, step=10)
    max_iters = st.slider("Maximum Steps", 100, 1000, 500, 50)
    
# Key parameters
st.sidebar.subheader("🎯 Key Parameters")

epsilon = st.sidebar.slider(
    "Information Uncertainty (ε)", 
    min_value=0.1, 
    max_value=1.5, 
    value=0.5, 
    step=0.1,
    help="The degree to which people misjudge the actual situation. Higher values indicate less accurate information."
)

security_density = st.sidebar.slider(
    "Security Force Density", 
    min_value=0.0, 
    max_value=0.1, 
    value=0.02, 
    step=0.01,
    help="Proportion of police/security forces in the population."
)

pp_mean = st.sidebar.slider(
    "Private Preference Mean", 
    min_value=-1.0, 
    max_value=0.0, 
    value=-0.5, 
    step=0.1,
    help="On average, whether people privately support(-1) or oppose(0) the regime."
)

threshold = st.sidebar.slider(
    "Activation Threshold", 
    min_value=2.0, 
    max_value=5.0, 
    value=3.5, 
    step=0.1,
    help="The level of encouragement needed for people to participate in protests."
)

# Add parameter explanation
with st.sidebar.expander("❓ Parameter Explanation"):
    st.write("""
    - **Information Uncertainty**: Simulates the effects of information control and rumors
    - **Security Force Density**: Strength of suppression capability
    - **Private Preference**: People's true attitudes
    - **Activation Threshold**: Psychological barrier to participation
    """)

# Main interface layout
tab1, tab2, tab3 = st.tabs(["🚀 Run Simulation", "📊 History Records", "📖 Model Explanation"])

with tab1:
    # Run button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button("🚀 Start Simulation", type="primary", use_container_width=True)
    
    if run_button:
        # Create placeholders
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Run model
        status_placeholder.info("🔄 Initializing model...")
        
        # Create model instance
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
        
        # Collect data
        time_steps = []
        active_counts = []
        support_counts = []
        oppose_counts = []
        jail_counts = []
        
        # Run simulation
        start_time = time.time()
        step = 0
        
        while model.running and step < max_iters:
            model.step()
            
            # Record data
            time_steps.append(step)
            active_counts.append(model.active_count)
            support_counts.append(model.support_count)
            oppose_counts.append(model.oppose_count)
            jail_counts.append(model.count_jail(model))
            
            # Update progress
            progress = (step + 1) / max_iters
            progress_bar.progress(progress)
            
            # Update status
            if step % 10 == 0:
                status_placeholder.info(f"🔄 Simulation in progress... Step {step}/{max_iters}")
            
            step += 1
        
        # Simulation completed
        simulation_time = time.time() - start_time
        status_placeholder.success(f"✅ Simulation completed! Time taken: {simulation_time:.2f} seconds")
        progress_bar.empty()
        
        # Display results
        st.subheader("📊 Simulation Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Revolution Occurred", 
                "Yes ✅" if model.revolution else "No ❌",
                delta="Success" if model.revolution else "Failed"
            )
        
        with col2:
            max_participation = max(active_counts) / model.citizen_count * 100
            st.metric(
                "Peak Participation Rate", 
                f"{max_participation:.1f}%",
                delta=f"{max_participation - 5:.1f}%" if max_participation > 5 else None
            )
        
        with col3:
            st.metric(
                "Time to Peak", 
                f"Step {active_counts.index(max(active_counts))}",
                delta="Fast" if active_counts.index(max(active_counts)) < 50 else "Slow"
            )
        
        with col4:
            st.metric(
                "Final Active Count", 
                f"{model.active_count} people",
                delta=f"{model.active_count - model.citizen_count * 0.1:.0f}"
            )
        
        # Dynamic charts
        st.subheader("📈 Population Dynamics")
        
        # Create charts
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left chart: Population state changes
        ax1.plot(time_steps, active_counts, label='Active Protesters', color='#FE6100', linewidth=2)
        ax1.plot(time_steps, support_counts, label='Supporters', color='#648FFF', linewidth=2)
        ax1.plot(time_steps, oppose_counts, label='Opponents', color='#A020F0', linewidth=2)
        ax1.plot(time_steps, jail_counts, label='Arrested', color='#000000', linewidth=2, linestyle='--')
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Number of People')
        ax1.set_title('Population State Evolution')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Right chart: Participation rate changes
        participation_rate = [a / model.citizen_count * 100 for a in active_counts]
        ax2.fill_between(time_steps, participation_rate, alpha=0.3, color='#FE6100')
        ax2.plot(time_steps, participation_rate, color='#FE6100', linewidth=2)
        ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% Critical Line')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Participation Rate (%)')
        ax2.set_title('Protest Participation Rate Changes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Data table
        with st.expander("📊 View Detailed Data"):
            # Create dataframe
            df = pd.DataFrame({
                'Time Step': time_steps[::10],  # Display every 10 steps
                'Active Count': active_counts[::10],
                'Support Count': support_counts[::10],
                'Oppose Count': oppose_counts[::10],
                'Arrested Count': jail_counts[::10],
                'Participation Rate(%)': [f"{x:.1f}" for x in participation_rate[::10]]
            })
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Save to session state
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
    st.subheader("📊 Historical Run Records")
    
    if 'history' in st.session_state and st.session_state.history:
        # Convert to dataframe
        history_data = []
        for h in st.session_state.history:
            record = {
                'Time': h['timestamp'].strftime('%H:%M:%S'),
                'ε': h['params']['epsilon'],
                'Security Density': h['params']['security_density'],
                'Private Preference': h['params']['pp_mean'],
                'Threshold': h['params']['threshold'],
                'Revolution': '✅' if h['results']['revolution'] else '❌',
                'Peak Participation': f"{h['results']['max_participation']:.1f}%",
                'Peak Time': h['results']['peak_time']
            }
            history_data.append(record)
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
        
        # Parameter comparison chart
        if len(history_data) > 1:
            st.subheader("📈 Parameter Impact Analysis")
            
            # Select parameter to analyze
            param_to_analyze = st.selectbox(
                "Select parameter to analyze",
                ['ε', 'Security Density', 'Private Preference', 'Threshold']
            )
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_data = [h['params']['epsilon'] if param_to_analyze == 'ε' 
                     else h['params']['security_density'] if param_to_analyze == 'Security Density'
                     else h['params']['pp_mean'] if param_to_analyze == 'Private Preference'
                     else h['params']['threshold'] 
                     for h in st.session_state.history]
            
            y_data = [h['results']['max_participation'] for h in st.session_state.history]
            colors = ['green' if h['results']['revolution'] else 'red' 
                     for h in st.session_state.history]
            
            scatter = ax.scatter(x_data, y_data, c=colors, s=100, alpha=0.6)
            ax.set_xlabel(param_to_analyze)
            ax.set_ylabel('Peak Participation Rate (%)')
            ax.set_title(f'{param_to_analyze} Impact on Participation Rate')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            green_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='g', markersize=10, label='Revolution Success')
            red_patch = plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='r', markersize=10, label='Revolution Failed')
            ax.legend(handles=[green_patch, red_patch])
            
            st.pyplot(fig)
        
        # Clear history button
        if st.button("🗑️ Clear History Records"):
            st.session_state.history = []
            st.experimental_rerun()
    else:
        st.info("No run records yet. Historical data will be displayed here after running some simulations.")

with tab3:
    st.subheader("📖 Model Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Model Principles
        
        This is an **Agent-Based Model** that simulates how individuals decide whether to participate in protests:
        
        1. **Individual Decision-Making**: Each person decides based on surrounding behavior and personal preferences
        2. **Information Spread**: People observe their surroundings, but information may be inaccurate
        3. **Suppression Risk**: Security forces arrest active protesters
        4. **Cascade Effect**: When enough people participate, it triggers a chain reaction
        
        ### Key Mechanisms
        
        - **Threshold Model**: People have different participation thresholds
        - **Spatial Interaction**: Can only see nearby people
        - **Dynamic Evolution**: Situation continuously changes over time
        """)
    
    with col2:
        st.markdown("""
        ### Parameter Details
        
        **Information Uncertainty (ε)**
        - Low values (0.1-0.3): Transparent information, people understand the real situation
        - Medium values (0.4-0.7): Some misjudgment exists
        - High values (0.8-1.5): Severe information confusion
        
        **Security Force Density**
        - 0-1%: Almost no suppression
        - 2-4%: Medium suppression
        - 5-10%: Strong suppression
        
        **Private Preference Mean**
        - -1.0: Strongly oppose regime
        - -0.5: Moderately oppose
        - 0.0: Neutral
        
        **Activation Threshold**
        - 2-3: Easily mobilized
        - 3-4: Requires some encouragement
        - 4-5: Difficult to mobilize
        """)
    
    st.markdown("---")
    
    # Add usage tips
    st.info("""
    💡 **Usage Tips**:
    - Try different parameter combinations and observe result changes
    - Pay special attention to "tipping point" phenomena - small parameter changes may lead to completely different outcomes
    - Revolution success usually requires: low suppression + high dissatisfaction + moderate information uncertainty
    """)
    
    # Contact information
    st.markdown("---")
    st.markdown("🔗 [Return to Main Website](https://yourusername.github.io) | 📧 Contact: lhyiris@outlook.com")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Based on Agent-Based Model (ABM) Technology | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)

