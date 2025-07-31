import os
import sys
import json
import subprocess
import tempfile
import glob
import streamlit as st
import pandas as pd
from PIL import Image

# Define frontend test cases
TEST_CASES = {
    "Basic Sinusoidal": "(2 + sin(x*2*pi)*cos(y*2*pi))",
    "Exp-Log Blend": "(2 + sin(exp(x)*2*pi)*cos(log(2*y+0.1)*pi))",
    "High-Freq Sinusoidal": "(2 + sin(x*4*pi)*cos(y*4*pi) + 0.3*sin(x*6*pi))",
    "Smooth Blend Quadrants": (
        "1*(1 - smooth_blend(x,0.5,0.01))*smooth_blend(y,0.5,0.01) + "
        "2.5*smooth_blend(x,0.5,0.01)*smooth_blend(y,0.5,0.01) + "
        "3*(1 - smooth_blend(x,0.5,0.01))*(1 - smooth_blend(y,0.5,0.01)) + "
        "4*smooth_blend(x,0.5,0.01)*(1 - smooth_blend(y,0.5,0.01))"
    )
}


def run_backend(args):
    """Invoke backend script with given arguments list."""
    try:
        subprocess.run([sys.executable, "backend.py"] + args, check=True)
        return True
    except subprocess.CalledProcessError:
        st.error("Backend error: check console for details.")
        return False


def main():
    st.set_page_config(page_title="Density-Equalizing Mapping", layout="wide")
    st.title("Density-Equalizing Mapping")

    # --- Session state setup ---
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    if 'original_ready' not in st.session_state:
        st.session_state.original_ready = False
    if 'mapped_ready' not in st.session_state:
        st.session_state.mapped_ready = False
    if 'auto_ckpt_loaded' not in st.session_state:
        st.session_state.auto_ckpt_loaded = False
        st.session_state.auto_ckpt_path = None

    # --- Sidebar inputs ---
    st.sidebar.header("Inputs")
    
    # Auto-load checkpoint from repository
    def find_auto_checkpoint():
        """Find and load checkpoint files from the current directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for ext in ['*.pth', '*.pt']:
            ckpt_files = glob.glob(os.path.join(current_dir, ext))
            if ckpt_files:
                return ckpt_files[0]  # Return the first found checkpoint
        return None
    
    # Try to auto-load checkpoint if not already done
    if not st.session_state.auto_ckpt_loaded:
        auto_ckpt = find_auto_checkpoint()
        if auto_ckpt:
            st.session_state.auto_ckpt_path = auto_ckpt
            st.session_state.auto_ckpt_loaded = True
    
    # Checkpoint upload with status indicator
    col_upload, col_status = st.sidebar.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload checkpoint", type=["pth", "pt"])
    with col_status:
        if st.session_state.auto_ckpt_loaded or uploaded:
            st.markdown('<div style="background-color: #4CAF50; color: white; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px;">✓</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color: #f44336; color: white; padding: 2px 6px; border-radius: 3px; text-align: center; font-size: 12px;">✗</div>', unsafe_allow_html=True)
    
    
    # Determine checkpoint path
    if uploaded:
        ckpt_path = os.path.join(st.session_state.temp_dir, uploaded.name)
        with open(ckpt_path, 'wb') as f:
            f.write(uploaded.getbuffer())
    elif st.session_state.auto_ckpt_loaded and st.session_state.auto_ckpt_path:
        ckpt_path = st.session_state.auto_ckpt_path
        # Show info about auto-loaded checkpoint
        st.sidebar.info(f"Auto-loaded: {os.path.basename(ckpt_path)}")
    else:
        ckpt_path = None

    # Function selector (dropdown + editable)
    preset_options = list(TEST_CASES.keys())
    # Check if there is a custom option
    if 'preset_sel' not in st.session_state:
        st.session_state.preset_sel = preset_options[0]
    if 'fn_expr' not in st.session_state:
        st.session_state.fn_expr = TEST_CASES[st.session_state.preset_sel]
    if 'previous_fn_expr' not in st.session_state:
        st.session_state.previous_fn_expr = st.session_state.fn_expr

    # Define preset change callback
    def on_preset_change():
        new_sel = st.session_state.preset_selector
        if new_sel != st.session_state.preset_sel:
            st.session_state.preset_sel = new_sel
            if new_sel in TEST_CASES:
                # switch to preset option
                st.session_state.fn_expr = TEST_CASES[new_sel]
            else:
                # switch to Custom mode
                st.session_state.fn_expr = st.session_state.previous_fn_expr
    
    # function expression input change callback
    def on_fn_expr_change():
        new_expr = st.session_state.fn_expr_input
        if new_expr != st.session_state.fn_expr:
            st.session_state.fn_expr = new_expr
            # 如果是在Custom模式下
            if st.session_state.preset_sel == "Custom":
                st.session_state.previous_fn_expr = new_expr

    # Preset selection dropdown
    st.sidebar.selectbox(
        "Preset f(x,y)", 
        options=preset_options + ["Custom"], 
        index=(preset_options + ["Custom"]).index(st.session_state.preset_sel) if st.session_state.preset_sel in (preset_options + ["Custom"]) else 0,
        key="preset_selector",
        on_change=on_preset_change
    )
    
    # Custom function input
    is_custom = st.session_state.preset_sel == "Custom"
    if is_custom:
        st.sidebar.text_input(
            "f(x,y)", 
            value=st.session_state.fn_expr, 
            key="fn_expr_input",
            on_change=on_fn_expr_change
        )
    else:
        st.sidebar.text_input(
            "f(x,y)", 
            value=st.session_state.fn_expr, 
            key="fn_expr_input",
            disabled=True
        )
    
    # Store the current function expression in session state
    fn_expr = st.session_state.fn_expr

    # Grid size and fine-tune epochs
    n_test = st.sidebar.slider("Grid size N_test", 16, 512, 128)
    ft_epochs = st.sidebar.slider("Fine-tune epochs", 0, 50, 20)

    # --- Action buttons ---
    if 'tab_index' not in st.session_state:
        st.session_state.tab_index = 0

    # Define tab switch function
    def switch_to_tab(tab_name):
        js_code = f"""
        <script>
            // Select the tab container
            console.log("Switching to tab: {tab_name}")
            var tabContainer = window.parent.document.querySelector('.stTabs');
            
            // Select the tab buttons
            var tabButtons = tabContainer.querySelectorAll('[role="tab"]');
            
            // Find the button for the target tab and click it
            tabButtons.forEach(function(button) {{
                if (button.innerText.trim() === "{tab_name}") {{
                    button.click();
                }}
            }});
        </script>
        """
        # Execute the JavaScript code
        st.components.v1.html(js_code, height=0, width=0)

    # Create columns for buttons and status indicators
    col1_btn, col1_status, col2_btn, col2_status = st.columns([4, 4, 4, 4])
    
    with col1_btn:
        if st.button("Generate Original Domain"):
            if not ckpt_path:
                st.sidebar.warning("Please upload a checkpoint file.")
            else:
                with st.spinner("Running original domain generation..."):
                    args = [
                        "--mode", "original",
                        "--ckpt", ckpt_path,
                        "--fn", fn_expr,
                        "--N", str(n_test),
                        "--out", st.session_state.temp_dir,
                        "--ft_epochs", str(ft_epochs)
                    ]
                    if run_backend(args):
                        st.session_state.original_ready = True
                        st.session_state.mapped_ready = False  # Reset mapped state
                        # Store the current function expression as the previous expression
                        st.session_state.previous_fn_expr = fn_expr
                        # Switch to Original tab
                        switch_to_tab("Original")
    
    # Original domain generation status
    with col1_status:
        if st.session_state.original_ready:
            st.markdown('✅', unsafe_allow_html=True)
    
    with col2_btn:
        if st.button("Generate Mapped Domain"):
            if not ckpt_path:
                st.sidebar.warning("Please upload a checkpoint file.")
            else:
                with st.spinner("Running mapped domain generation..."):
                    args = [
                        "--mode", "mapped",
                        "--ckpt", ckpt_path,
                        "--fn", fn_expr,
                        "--N", str(n_test),
                        "--out", st.session_state.temp_dir,
                        "--ft_epochs", str(ft_epochs)
                    ]
                    if run_backend(args):
                        st.session_state.mapped_ready = True
                        # Store the current function expression as the previous expression
                        st.session_state.previous_fn_expr = fn_expr
                        # Switch to Mapped tab
                        switch_to_tab("Mapped")

    # Mapped domain generation status
    with col2_status:
        if st.session_state.mapped_ready:
            st.markdown('✅', unsafe_allow_html=True)


    # --- Tabs for output ---
    tab_labels = ["Original", "Mapped", "Comparison"]

    # Ensure tab_index is within valid range
    if 'tab_index' in st.session_state:
        if st.session_state.tab_index < 0 or st.session_state.tab_index >= len(tab_labels):
            st.session_state.tab_index = 0

    # Get the current selected tab index
    selected_tab = st.session_state.get('tab_index', 0)

    # Create tabs using the tabs component
    tabs = st.tabs(tab_labels)

    # Tab: Original
    with tabs[0]:
        orig_path = os.path.join(st.session_state.temp_dir, "original.png")
        if st.session_state.original_ready and os.path.exists(orig_path):
            st.image(orig_path, caption="Original Domain", use_container_width=True)
        else:
            st.info("Click 'Generate Original Domain' to visualize.")

    # Tab: Mapped
    with tabs[1]:
        mapped_path = os.path.join(st.session_state.temp_dir, "mapped.png")
        metrics_path = os.path.join(st.session_state.temp_dir, "metrics.json")
        hist_path = os.path.join(st.session_state.temp_dir, "hist_compare.png")
        if st.session_state.mapped_ready and os.path.exists(mapped_path):
            st.image(mapped_path, caption="Mapped Domain", use_container_width=True)
        else:
            st.info("Click 'Generate Mapped Domain' to view results.")

    # Tab: Comparison
    with tabs[2]:
        if st.session_state.original_ready and st.session_state.mapped_ready:
            colA, colB = st.columns(2)
            # Original column
            with colA:
                if os.path.exists(orig_path):
                    st.image(orig_path, caption="Original Domain", use_container_width=True)
                orig_hist = os.path.join(st.session_state.temp_dir, "hist_compare.png")
                if os.path.exists(orig_hist):
                    img = Image.open(orig_hist)
                    w, h = img.size
                    left = img.crop((0, 0, w//2, h))
                    st.image(left, caption="Original Histogram", use_container_width=True)
            # Mapped column
            with colB:
                if os.path.exists(mapped_path):
                    st.image(mapped_path, caption="Mapped Domain", use_container_width=True)
                mapped_hist = os.path.join(st.session_state.temp_dir, "hist_compare.png")
                if os.path.exists(mapped_hist):
                    img = Image.open(mapped_hist)
                    w, h = img.size
                    right = img.crop((w//2, 0, w, h))
                    st.image(right, caption="Mapped Histogram", use_container_width=True)
            # Metrics table
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                comp_df = pd.DataFrame([{
                    "Time (s)": metrics.get("Time (s)"),
                    "Std/Mean Orig": metrics.get("Std/Mean Orig"),
                    "Std/Mean Map": metrics.get("Std/Mean Map"),
                    "Max |μ|": metrics.get("Max |μ|"),
                    "Mean |μ|": metrics.get("Mean |μ|")
                }])

                st.dataframe(
                    comp_df,
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown(
                    """
                    <style>
                    .stDataFrame thead tr th, .stDataFrame tbody tr td {
                        text-align: center !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("Generate both Original and Mapped domains to compare.")

if __name__ == '__main__':
    main()
