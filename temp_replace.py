# -*- coding: utf-8 -*-
from pathlib import Path
text = Path('app.py').read_text(encoding='utf-8')
replacements = {
    'st.subheader("BTC/USDT chart")': 'st.subheader("Graphique BTC/USDT")',
    'st.selectbox("Interval", INTERVALS, index=0)': 'st.selectbox("Intervalle", INTERVALS, index=0)',
    'st.subheader("Market snapshot")': 'st.subheader("Instantané marché")',
    'kpi_block("Price", f"{ticker[\'price\']:, .2f} $"': None,
}
