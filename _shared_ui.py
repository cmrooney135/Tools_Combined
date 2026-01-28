
# _shared_ui.py
import streamlit as st

def top_bar(page_icon: str, title: str, home_page_path: str = "Home.py"):
    """
    Renders a top bar with:
      - a small home icon button (left) that routes to Home
      - the page icon + title (center/left, responsive)
    """
    # Small home icon button (left)
    col_home, col_title, col_spacer = st.columns([0.12, 1.0, 0.05])
    with col_home:
        st.markdown(
            """
            <style>
            .home-btn > button {
              border-radius: 12px !important;
              padding: 6px 10px !important;
              font-size: 18px !important;
              line-height: 1 !important;
              border: 1px solid #e6e6e6 !important;
              background: #ffffff !important;
              color: #111827 !important;
              box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
            }
            .home-btn > button:hover {
              box-shadow: 0 6px 16px rgba(0,0,0,0.08) !important;
              transform: translateY(-1px);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button("🏠", key="go_home_btn", help="Go to Home", type="secondary"):
            st.switch_page(home_page_path)

    # Title with page icon (center)
    with col_title:
        st.title(f"{page_icon} {title}")
