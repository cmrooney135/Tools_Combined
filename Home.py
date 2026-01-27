
import streamlit as st

st.set_page_config(page_title="Tools Home", page_icon = "🏠", layout="wide")

# ---------- Card styling for st.button ----------
st.markdown("""
<style>
/* Make all st.button look like clean "cards" */
.stButton > button {
  width: 100% !important;
  text-align: left !important;
  border-radius: 14px !important;
  border: 1px solid #e6e6e6 !important;
  background: #ffffff !important;
  color: #111827 !important; /* slate-900 */
  padding: 14px 16px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
  transition: box-shadow .15s ease, transform .05s ease !important;
  white-space: pre-wrap !important; /* allow \\n to wrap to a second line */
  line-height: 1.15 !important;
  font-size: 1rem !important;
}
.stButton > button:hover {
  box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
  transform: translateY(-1px);
}
.stButton > button:focus { outline: none !important; }

/* Slightly emphasize the title line inside the label */
.card-title {
  font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.title("PTL Cable Tools Home")
st.write("")

# ---------- Centered Home ----------
l, c, r = st.columns([1, 2, 1])
with c:
    st.markdown('<div style="display:flex;flex-direction:column;align-items:center;">'
                '<div style="font-size:40px;line-height:1;margin-bottom:6px;">🏠</div>'
                '</div>', unsafe_allow_html=True)
    # Entire card clickable
    if st.button("Home\nOverview and entry point for all PTL cable tools.", use_container_width=True, key="home"):
        st.switch_page("Home.py")

st.write("---")

# ---------- Two Columns: Tesla (left) | Paradise (right) ----------
tesla_col, paradise_col = st.columns(2)

# --- TESLA ---
with tesla_col:
    st.markdown('#### ⚡ Tesla')

    if st.button("📈  Tesla Summary\nSummary statistics and overview histograms for DCR and continuity.",
                 use_container_width=True, key="tesla_summary"):
        st.switch_page("pages/tesla_summary.py")

    if st.button("🛠️  Tesla Tools\nHeatmaps and defect maps for any number of Tesla cables (high-click version).",
                 use_container_width=True, key="tesla_tools"):
        st.switch_page("pages/tesla_tools.py")

    if st.button("⚡  Tesla Tools (operator version)\nHeatmaps and defect maps for any number of Tesla cables (low-click version)",
                 use_container_width=True, key="tesla_ops"):
        st.switch_page("pages/tesla_tools_few_clicks.py")

# --- PARADISE ---
with paradise_col:
    st.markdown('#### 🧰 Paradise')

    if st.button("📊  Paradise Summary\nSummary statistics and overview histograms for DCR and continuity.",
                 use_container_width=True, key="paradise_summary"):
        st.switch_page("pages/paradise_summary.py")

    if st.button("🧰  Paradise Tools\nHeatmaps and defect maps for any number of paradise cables (high-click version)",
                 use_container_width=True, key="paradise_tools"):
        st.switch_page("pages/paradise_tools.py")

    if st.button("⚙️  Paradise Tools (operator version)\nHeatmaps and defect maps for any number of paradise cables (low-click version)",
                 use_container_width=True, key="paradise_ops"):
        st.switch_page("pages/paradise_tools_few_clicks.py")

