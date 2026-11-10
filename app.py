import streamlit as st

import us_quant_screener_v2 as m

st.set_page_config(page_title="퀀트 스크리너", page_icon="📈", layout="centered")
st.title("📈 미국주식 퀀트 스크리너")
st.caption("볼린저 바닥 + RSI 과매도 + 저평가 우량주 · S&P 500 기준")

with st.expander("설정", expanded=False):
    account = st.number_input("계좌 금액 (USD)", 1000, 10_000_000, 10000, step=1000)
    rsi_max = st.slider("RSI 상한 (낮을수록 엄격)", 20, 45, 35)
    pctb_max = st.slider("볼린저 %B 상한 (낮을수록 바닥)", 0.0, 0.4, 0.15, 0.01)
    top_n = st.slider("표시할 종목 수", 1, 10, 3)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def run(account, rsi_max, pctb_max):
    m.CFG["account"] = account
    m.CFG["rsi_max"] = rsi_max
    m.CFG["pctb_max"] = pctb_max
    uni = m.get_universe()
    tickers = [t for t, s in uni.items() if s not in m.EXCLUDED_SECTORS]
    regime = m.market_regime()
    cands = m.technical_scan(tickers)
    if cands.empty:
        return None, regime
    res = m.fundamental_filter(cands)
    if res.empty:
        return None, regime
    return m.score_and_size(res), regime


if st.button("🔍 오늘의 후보 찾기", type="primary", use_container_width=True):
    with st.spinner("분석 중입니다... (2~5분 걸려요)"):
        res, regime = run(account, rsi_max, pctb_max)
    st.session_state["out"] = (res, regime)

if "out" in st.session_state:
    res, regime = st.session_state["out"]
    st.info(regime)
    if res is None or len(res) == 0:
        st.warning("조건을 모두 통과한 종목이 없습니다. 설정에서 RSI나 %B 기준을 완화해 보세요.")
    else:
        for tk, r in res.head(top_n).iterrows():
            st.subheader(f"{tk} · {r['name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("종합점수", r["score"])
            c2.metric("현재가", f"${r['price']:.2f}")
            c3.metric("F-Score", f"{int(r['Fscore'])}/9")
            c1.metric("RSI", f"{r['RSI14']:.0f}")
            c2.metric("PER", f"{r['PER']:.1f}")
            c3.metric("PBR", f"{r['PBR']:.1f}")
            st.write(
                f"매출 {r['rev_g']:+.1%} · 영업이익 {r['op_g']:+.1%} · Altman Z {r['AltmanZ']:.1f} · "
                f"200일선 {'위' if r['uptrend'] else '아래'}"
            )
            st.write(f"🛑 손절가 **${r['stop']:.2f}** · 계좌 기준 **{int(r['shares'])}주**")
            st.caption(f"뉴스: {r['news'] or '없음'}")
            st.divider()
        with st.expander("전체 후보 표"):
            st.dataframe(res.drop(columns=["news"]).round(3))
    st.caption("※ 참고용 자동 스크리닝이며 투자 권유가 아닙니다. 매수 전 공시·뉴스를 직접 확인하세요.")
