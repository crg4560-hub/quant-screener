"""
미국 주식 퀀트 스크리너 v2  (여러 퀀트 서적의 아이디어를 종합)

반영한 아이디어 (개념 해석이며 책 내용을 그대로 옮긴 것은 아닙니다)
- Bollinger, 'Bollinger on Bollinger Bands' : %B로 밴드 위치 측정, 밴드 터치만으로 매수하지 말고
                                               RSI 등 다른 지표로 확인
- Connors & Alvarez, 'Short Term Trading Strategies That Work' : 단기 과매도(RSI2) + 장기 상승추세 눌림목,
                                               평균회귀 청산 규칙
- Greenblatt, 'The Little Book That Beats the Market' : 마법공식 (이익수익률 + 투하자본수익률)
- O'Shaughnessy, 'What Works on Wall Street' : 복합 가치지표, 주주환원수익률, 가치 + 모멘텀 결합
- Piotroski F-Score (2000 논문, 여러 퀀트서가 인용) : 9점 재무건전성 점수
- Gray & Carlisle, 'Quantitative Value' : 싸 보이지만 부실한 종목(가치 함정) 제거 -> Altman Z
- Antonacci 'Dual Momentum' / Faber 추세추종 : 시장 국면(SPY 200일선, 12개월 수익률) 확인
- Chan, 'Quantitative Trading' : 백테스트 시 비용 반영, 편향 경고, 위험 기반 포지션 사이징

설치:  pip install yfinance pandas numpy lxml requests
실행:  python us_quant_screener_v2.py
       python us_quant_screener_v2.py --account 20000
       python us_quant_screener_v2.py --tickers AAPL MSFT KO
       python us_quant_screener_v2.py --backtest

※ 참고용 도구입니다. 투자 판단과 책임은 본인에게 있습니다.
"""
import argparse
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ───────────── 설정값 (여기만 바꿔서 조절) ─────────────
CFG = {
    # [Bollinger + Connors] 기술적 진입 조건
    "bb_window": 20,
    "bb_std": 2.0,
    "pctb_max": 0.15,          # %B 이하 = 밴드 바닥권
    "rsi_period": 14,
    "rsi_max": 35,             # RSI14 이하 = 과매도 (밴드 신호의 확인용)
    "max_1m_drop": -0.30,      # 최근 1개월 -30%보다 더 빠진 '추락 중' 종목 제외
    "require_uptrend": False,  # True면 200일선 위 눌림목만 (Connors식, 더 보수적)
    # 하드 필터: 우량 + 저평가
    "min_market_cap": 5e9,
    "per_max": 25,
    "pbr_max": 5,
    "rev_growth_min": 0.0,     # 매출 성장 > 0
    "op_growth_min": 0.0,      # 영업이익 성장 > 0
    "op_margin_min": 0.08,
    "roe_min": 0.08,
    "debt_to_equity_max": 150,
    "fscore_min": 6,           # [Piotroski] 9점 만점
    "altman_z_min": 1.8,       # [Gray] 부실 위험 구간(1.8 미만) 제외
    # 종합점수 가중치 (합계 1.0)
    "weights": {
        "value": 0.25,         # [O'Shaughnessy] 복합 가치
        "magic": 0.15,         # [Greenblatt] 이익수익률 + ROC
        "quality": 0.25,       # [Piotroski] F-Score
        "momentum": 0.10,      # [O'Shaughnessy/Antonacci] 12-1개월 모멘텀
        "oversold": 0.15,      # 얼마나 바닥권인가
        "sh_yield": 0.10,      # 배당 + 자사주매입
    },
    "uptrend_bonus": 0.05,     # 200일선 위 눌림목이면 점수 가산
    # [Chan] 포지션 사이징
    "account": 10000,
    "risk_per_trade": 0.01,    # 한 종목에서 계좌의 1%만 잃도록
    "atr_stop_mult": 2.5,
    "max_weight": 0.10,        # 한 종목 최대 비중
    # 백테스트
    "bt_years": 5,
    "bt_max_hold": 20,
    "bt_exit_pctb": 0.5,       # 중심선 회복 시 청산 (평균회귀)
    "bt_exit_rsi": 55,
    "bt_cost": 0.001,          # 편도 0.1% (수수료+슬리피지 가정)
}
EXCLUDED_SECTORS = {"Financials", "Financial Services", "Utilities"}  # [Greenblatt] 금융/유틸리티 제외


# ───────────── 데이터 / 지표 ─────────────
def get_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    df = pd.read_html(StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df.set_index("Symbol")["GICS Sector"].to_dict()


def download_ohlc(tickers, period):
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, list(tickers)])
    close = raw["Close"].dropna(how="all", axis=1)
    return {f: raw[f][close.columns] for f in ["Open", "High", "Low", "Close"]}


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def bollinger(close, w, k):
    ma = close.rolling(w).mean()
    sd = close.rolling(w).std()
    upper, lower = ma + k * sd, ma - k * sd
    return (close - lower) / (upper - lower), (upper - lower) / ma  # %B, 밴드폭


def atr(high, low, close, n=14):
    pc = close.shift(1)
    tr = np.maximum.reduce([(high - low).values, (high - pc).abs().values,
                            (low - pc).abs().values])
    tr = pd.DataFrame(tr, index=close.index, columns=close.columns)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ───────────── 1) 시장 국면 [Antonacci / Faber] ─────────────
def market_regime():
    try:
        spy = yf.download("SPY", period="2y", auto_adjust=True, progress=False)["Close"].squeeze()
        above = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]
        ret12 = spy.iloc[-1] / spy.iloc[-253] - 1
        msg = f"[시장 국면] SPY 200일선 {'위' if above else '아래'}, 12개월 수익률 {ret12:+.1%}"
        if not above or ret12 < 0:
            msg += "\n  ⚠ 약세 국면: 바닥 신호가 '더 떨어지는 구간'일 수 있으니 비중을 줄이세요."
    except Exception as e:
        msg = f"[시장 국면] 조회 실패: {e}"
    print(msg)
    return msg


# ───────────── 2) 기술적 스캔 [Bollinger + Connors] ─────────────
def technical_scan(tickers):
    print(f"[1/3] 가격 데이터 다운로드 ({len(tickers)}종목)...")
    d = download_ohlc(tickers, "2y")
    c = d["Close"]
    pctb, bw = bollinger(c, CFG["bb_window"], CFG["bb_std"])
    last = pd.DataFrame({
        "price": c.iloc[-1],
        "pctB": pctb.iloc[-1],
        "bandwidth": bw.iloc[-1],
        "RSI14": rsi(c, CFG["rsi_period"]).iloc[-1],
        "RSI2": rsi(c, 2).iloc[-1],
        "uptrend": c.iloc[-1] > c.rolling(200).mean().iloc[-1],
        "mom_12_1": c.iloc[-22] / c.iloc[-253] - 1,
        "ret_1m": c.iloc[-1] / c.iloc[-22] - 1,
        "ATR": atr(d["High"], d["Low"], c).iloc[-1],
    })
    hit = last[(last.pctB <= CFG["pctb_max"]) & (last.RSI14 <= CFG["rsi_max"])
               & (last.ret_1m >= CFG["max_1m_drop"])]
    if CFG["require_uptrend"]:
        hit = hit[hit.uptrend]
    print(f"      기술적 조건 통과: {len(hit)}종목")
    return hit


# ───────────── 3) 재무 분석 ─────────────
def pick(df, names, col=0, default=np.nan):
    if df is None or df.empty or col >= df.shape[1]:
        return default
    for n in names:
        if n in df.index:
            v = df.loc[n].iloc[col]
            if pd.notna(v):
                return float(v)
    return default


def pct_change(a, b):
    return (a - b) / abs(b) if pd.notna(a) and pd.notna(b) and b != 0 else np.nan


def fundamentals(tk):
    t = yf.Ticker(tk)
    info = t.info or {}
    fin, bs, cf = t.financials, t.balance_sheet, t.cashflow
    g = lambda df, names, col=0, default=np.nan: pick(df, names, col, default)

    rev0, rev1 = g(fin, ["Total Revenue"]), g(fin, ["Total Revenue"], 1)
    op0, op1 = g(fin, ["Operating Income", "EBIT"]), g(fin, ["Operating Income", "EBIT"], 1)
    ebit0 = g(fin, ["EBIT", "Operating Income"])
    ni0, ni1 = g(fin, ["Net Income"]), g(fin, ["Net Income"], 1)
    gp0, gp1 = g(fin, ["Gross Profit"]), g(fin, ["Gross Profit"], 1)
    ta0, ta1 = g(bs, ["Total Assets"]), g(bs, ["Total Assets"], 1)
    ca0, ca1 = g(bs, ["Current Assets"]), g(bs, ["Current Assets"], 1)
    cl0, cl1 = g(bs, ["Current Liabilities"]), g(bs, ["Current Liabilities"], 1)
    ltd0 = g(bs, ["Long Term Debt", "Total Debt"], 0, 0)
    ltd1 = g(bs, ["Long Term Debt", "Total Debt"], 1, 0)
    sh0 = g(bs, ["Ordinary Shares Number", "Share Issued"])
    sh1 = g(bs, ["Ordinary Shares Number", "Share Issued"], 1)
    re0 = g(bs, ["Retained Earnings"])
    tl0 = g(bs, ["Total Liabilities Net Minority Interest"])
    cash0 = g(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], 0, 0)
    cfo0 = g(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    buyback = g(cf, ["Repurchase Of Capital Stock"], 0, 0)
    divpaid = g(cf, ["Common Stock Dividend Paid", "Cash Dividends Paid"], 0, 0)

    mcap = info.get("marketCap") or np.nan
    ev = info.get("enterpriseValue") or np.nan

    # [Piotroski] F-Score 9점
    roa0, roa1 = ni0 / ta0, ni1 / ta1
    f = [
        roa0 > 0,
        cfo0 > 0,
        roa0 > roa1,
        cfo0 > ni0,
        ltd0 / ta0 <= ltd1 / ta1,
        (ca0 / cl0) > (ca1 / cl1),
        sh0 <= sh1,
        (gp0 / rev0) > (gp1 / rev1),
        (rev0 / ta0) > (rev1 / ta1),
    ]
    fscore = int(sum(bool(x) for x in f))

    # [Greenblatt] 마법공식 (근사): 이익수익률 = EBIT/EV, ROC = EBIT/투하자본
    invested = ta0 - cl0 - cash0
    ey = ebit0 / ev if pd.notna(ev) and ev > 0 else np.nan
    roc = ebit0 / invested if pd.notna(invested) and invested > 0 else np.nan

    # [Gray] Altman Z-Score (부실 위험)
    z = (1.2 * (ca0 - cl0) / ta0 + 1.4 * re0 / ta0 + 3.3 * ebit0 / ta0
         + 0.6 * mcap / tl0 + 1.0 * rev0 / ta0)

    fcf = info.get("freeCashflow")
    p_fcf = mcap / fcf if fcf and fcf > 0 and pd.notna(mcap) else np.nan
    sh_yield = -(buyback + divpaid) / mcap if pd.notna(mcap) and mcap > 0 else np.nan

    news = []
    try:
        for n in (t.news or [])[:3]:
            title = n.get("title") or n.get("content", {}).get("title")
            if title:
                news.append(title)
    except Exception:
        pass

    return {
        "ticker": tk, "name": info.get("shortName"), "sector": info.get("sector"),
        "mkt_cap_B": mcap / 1e9 if pd.notna(mcap) else np.nan,
        "PER": info.get("trailingPE"), "PBR": info.get("priceToBook"),
        "PSR": info.get("priceToSalesTrailing12Months"),
        "EV_EBITDA": info.get("enterpriseToEbitda"), "P_FCF": p_fcf,
        "EY": ey, "ROC": roc, "Fscore": fscore, "AltmanZ": z,
        "rev_g": pct_change(rev0, rev1), "op_g": pct_change(op0, op1),
        "op_margin": info.get("operatingMargins"), "ROE": info.get("returnOnEquity"),
        "debt_eq": info.get("debtToEquity"), "sh_yield": sh_yield,
        "news": " | ".join(news),
    }


def fundamental_filter(cands):
    print(f"[2/3] 재무 데이터 조회 ({len(cands)}종목)...")
    rows = []
    for tk in cands.index:
        try:
            r = fundamentals(tk)
            if r["sector"] in EXCLUDED_SECTORS:
                continue
            rows.append(r)
        except Exception as e:
            print(f"      {tk} 건너뜀: {e}")
    if not rows:
        return pd.DataFrame()
    df = cands.join(pd.DataFrame(rows).set_index("ticker"), how="inner")

    c = CFG
    num = lambda col, na: pd.to_numeric(df[col], errors="coerce").fillna(na)
    ok = (
        (num("mkt_cap_B", 0) * 1e9 >= c["min_market_cap"])
        & num("PER", np.nan).between(0.01, c["per_max"])
        & (num("PBR", 99) <= c["pbr_max"])
        & (num("rev_g", -1) > c["rev_growth_min"])
        & (num("op_g", -1) > c["op_growth_min"])
        & (num("op_margin", 0) >= c["op_margin_min"])
        & (num("ROE", 0) >= c["roe_min"])
        & (num("debt_eq", 0) <= c["debt_to_equity_max"])
        & (df.Fscore >= c["fscore_min"])
        & (df.AltmanZ.fillna(-9) >= c["altman_z_min"])
    )
    out = df[ok].copy()
    print(f"      재무/가치/안전성 통과: {len(out)}종목")
    return out


# ───────────── 4) 종합 점수 + 포지션 사이징 ─────────────
def score_and_size(df):
    w = CFG["weights"]
    hi = lambda s: pd.to_numeric(s, errors="coerce").rank(pct=True)          # 클수록 좋음
    lo = lambda s: 1 - pd.to_numeric(s, errors="coerce").rank(pct=True)     # 작을수록 좋음

    value = pd.concat([lo(df.PER), lo(df.PBR), lo(df.PSR), lo(df.EV_EBITDA), lo(df.P_FCF)],
                      axis=1).mean(axis=1)
    magic = (hi(df.EY) + hi(df.ROC)) / 2
    quality = df.Fscore / 9
    momentum = hi(df.mom_12_1)
    oversold = (lo(df.RSI14) + lo(df.pctB)) / 2
    shy = hi(df.sh_yield)

    s = (w["value"] * value.fillna(0.5) + w["magic"] * magic.fillna(0.5)
         + w["quality"] * quality + w["momentum"] * momentum.fillna(0.5)
         + w["oversold"] * oversold.fillna(0.5) + w["sh_yield"] * shy.fillna(0.3))
    s = s + CFG["uptrend_bonus"] * df.uptrend.astype(float)
    df["score"] = (s * 100).round(1)

    # [Chan] 위험 기반 포지션 사이징
    stop = df.price - CFG["atr_stop_mult"] * df.ATR
    risk_ps = (df.price - stop).replace(0, np.nan)
    sh_risk = CFG["account"] * CFG["risk_per_trade"] / risk_ps
    sh_cap = CFG["account"] * CFG["max_weight"] / df.price
    df["stop"] = stop.round(2)
    df["shares"] = np.floor(np.minimum(sh_risk, sh_cap)).fillna(0).astype(int)
    return df.sort_values("score", ascending=False)


# ───────────── 백테스트 [Chan] (기술적 진입/청산 규칙만) ─────────────
def backtest(tickers):
    print(f"[백테스트] {len(tickers)}종목, 최근 {CFG['bt_years']}년 (기술적 규칙만 검증)")
    d = download_ohlc(tickers, f"{CFG['bt_years'] + 1}y")
    O, H, L, C = d["Open"], d["High"], d["Low"], d["Close"]
    pctb, _ = bollinger(C, CFG["bb_window"], CFG["bb_std"])
    R, MA200, A = rsi(C, CFG["rsi_period"]), C.rolling(200).mean(), atr(H, L, C)
    trades = []
    for tk in C.columns:
        o, h, l, c = O[tk].values, H[tk].values, L[tk].values, C[tk].values
        pb, rr, m2, at = pctb[tk].values, R[tk].values, MA200[tk].values, A[tk].values
        n, i = len(c), 200
        while i < n - 2:
            sig = pb[i] <= CFG["pctb_max"] and rr[i] <= CFG["rsi_max"]
            if sig and CFG["require_uptrend"]:
                sig = c[i] > m2[i]
            if not sig or np.isnan(o[i + 1]) or np.isnan(at[i]):
                i += 1
                continue
            entry = o[i + 1]  # 신호 다음 날 시가 체결 (미래 참조 방지)
            stop = entry - CFG["atr_stop_mult"] * at[i]
            last = min(n - 1, i + 1 + CFG["bt_max_hold"])
            exit_px, j_exit = c[last], last
            for j in range(i + 1, last + 1):
                if l[j] <= stop:
                    exit_px, j_exit = min(stop, o[j]), j
                    break
                if j > i + 1 and (pb[j] >= CFG["bt_exit_pctb"] or rr[j] >= CFG["bt_exit_rsi"]):
                    exit_px, j_exit = c[j], j
                    break
            ret = exit_px / entry - 1 - 2 * CFG["bt_cost"]
            trades.append({"ticker": tk, "entry_date": C.index[i + 1], "exit_date": C.index[j_exit],
                           "hold_days": j_exit - i, "ret": ret})
            i = j_exit + 1
    t = pd.DataFrame(trades)
    if t.empty:
        print("거래가 없습니다. 조건을 완화해 보세요.")
        return
    wins, losses = t[t.ret > 0].ret.sum(), -t[t.ret <= 0].ret.sum()
    print(f"  거래 수      : {len(t)}")
    print(f"  승률         : {(t.ret > 0).mean():.1%}")
    print(f"  평균 수익률  : {t.ret.mean():+.2%}   (중앙값 {t.ret.median():+.2%})")
    print(f"  손익비(PF)   : {wins / losses:.2f}" if losses > 0 else "  손익비(PF)   : inf")
    print(f"  평균 보유일  : {t.hold_days.mean():.1f}   최악 거래 {t.ret.min():+.1%}")
    t.to_csv("backtest_trades.csv", index=False)
    print("  → backtest_trades.csv 저장")
    print("  ⚠ 주의: 현재 S&P500 구성종목만 사용(생존편향), 과거 재무 데이터는 반영 안 됨(재무 필터 미검증).")
    print("          과최적화를 피하려면 기간을 나눠(예: 앞 3년/뒤 2년) 결과가 비슷한지 확인하세요.")


# ───────────── main ─────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", help="직접 지정 (기본: S&P500)")
    ap.add_argument("--account", type=float, help="계좌 금액(USD)")
    ap.add_argument("--backtest", action="store_true", help="기술적 규칙 백테스트")
    args = ap.parse_args()
    if args.account:
        CFG["account"] = args.account

    if args.tickers:
        tickers = args.tickers
    else:
        uni = get_universe()
        tickers = [t for t, s in uni.items() if s not in EXCLUDED_SECTORS]

    if args.backtest:
        backtest(tickers)
        return

    market_regime()
    cands = technical_scan(tickers)
    if cands.empty:
        print("기술적 조건 통과 종목이 없습니다. CFG의 pctb_max / rsi_max를 완화해 보세요.")
        return
    res = fundamental_filter(cands)
    if res.empty:
        print("재무 조건까지 통과한 종목이 없습니다. CFG 기준을 완화해 보세요.")
        cands.to_csv("tech_candidates.csv")
        return

    res = score_and_size(res)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n[3/3] 최종 후보 (종합점수 순)")
    cols = ["name", "score", "price", "pctB", "RSI14", "PER", "PBR", "Fscore", "AltmanZ",
            "rev_g", "op_g", "uptrend", "stop", "shares"]
    print(res[cols].round(3))
    print("\n최근 뉴스 헤드라인 (이슈 확인용):")
    for tk, r in res.iterrows():
        print(f"- {tk}: {r.news or '없음'}")
    res.to_csv("screener_result_v2.csv")
    print("\n→ screener_result_v2.csv 저장 완료")


if __name__ == "__main__":
    main()
