"""
market_sentiment_cycle_v2.py
市场情绪周期判断器 V2（实盘版 + 离线演示模式）
================================================
功能：
  1) 用 akshare 拉取当日涨停池 / 昨日涨停表现 / 概念板块强度
  2) 6 维打分 → 判定当前情绪周期阶段
  3) 统计概念板块涨停家数 → 输出最强主线（绝对主线 / 次线）
  4) 在主线板块内识别龙头梯队与中军名单

数据源：东方财富（akshare）
联网失败或沙盒环境 → 自动切到 _demo_data 离线快照演示。

⚠️ 免责声明：仅供学习研究，不构成任何投资建议。
"""

import sys
import time
import datetime
import pandas as pd
import numpy as np

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False

# 离线演示数据
try:
    from _demo_data import (
        get_demo_zt_pool, get_demo_zt_prev, get_demo_dt, get_demo_concepts
    )
    HAS_DEMO = True
except ImportError:
    HAS_DEMO = False


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def _to_num(s):
    if s is None:
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip().replace("%", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _today_str():
    return datetime.datetime.now().strftime("%Y%m%d")


def _latest_trade_date(target_date=None):
    if not HAS_AK:
        return target_date or _today_str()
    try:
        df = ak.tool_trade_date_hist_sina()
        dates = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d").tolist()
        d = target_date or _today_str()
        valid = [x for x in dates if x <= d]
        return valid[-1] if valid else d
    except Exception:
        return target_date or _today_str()


def _safe_ak(fn, *a, **kw):
    """调 akshare 接口，失败返回 None，不抛异常"""
    if not HAS_AK:
        return None
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"  ! 接口异常({fn.__name__}): {e}")
        return None


# ─────────────────────────────────────────────
# 第 1 层：情绪周期判断
# ─────────────────────────────────────────────
def fetch_sentiment_data(date_str, offline=False):
    """返回 dict: zt_pool / zt_prev / zt_dt"""
    if offline or not HAS_AK:
        print(f"[*] 使用离线演示数据 ({date_str})")
        return {
            "zt_pool": get_demo_zt_pool(),
            "zt_prev": get_demo_zt_prev(),
            "zt_dt": get_demo_dt(),
            "source": "demo",
        }

    print(f"[*] 正在拉取 {date_str} 的涨停/跌停/昨日涨停数据 …")
    zt_pool = _safe_ak(ak.stock_zt_pool_em, date=date_str)
    zt_prev = _safe_ak(ak.stock_zt_pool_previous_em, date=date_str)
    zt_dt   = _safe_ak(ak.stock_zt_pool_dtgc_em, date=date_str)

    # 任一关键数据拉不到 → 退回演示
    if zt_pool is None or zt_pool.empty:
        print("  ! 实盘数据不可用，自动切换离线演示模式")
        return {
            "zt_pool": get_demo_zt_pool(),
            "zt_prev": get_demo_zt_prev(),
            "zt_dt": get_demo_dt(),
            "source": "demo",
        }
    return {
        "zt_pool": zt_pool,
        "zt_prev": zt_prev or pd.DataFrame(),
        "zt_dt": zt_dt,
        "source": "live",
    }


def calc_sentiment_score(data):
    """6 维打分 → 总分 + 明细 + 计数"""
    zt = data["zt_pool"]
    prev = data["zt_prev"]
    dt = data["zt_dt"]

    zt_count = len(zt) if zt is not None and not zt.empty else 0
    dt_count = len(dt) if dt is not None and not dt.empty else 0

    # 1) 涨停跌停比
    if dt_count == 0:
        ratio_score = 100 if zt_count > 0 else 30
    else:
        ratio = zt_count / dt_count
        ratio_score = int(np.clip(ratio * 25, 0, 100))
    ratio_score = max(ratio_score, 5)

    # 2) 连板高度
    if zt.empty or "连板数" not in zt.columns:
        height = 0
    else:
        height = int(zt["连板数"].astype(float).max()) if zt_count else 0
    height_score = int(np.clip(height * 18, 0, 100))

    # 3) 涨跌比
    total = zt_count + dt_count
    breadth_score = int(zt_count / total * 100) if total > 0 else 30

    # 4) 封板坚决度（封板资金中位数）
    if not zt.empty and "封板资金" in zt.columns:
        cap = pd.to_numeric(zt["封板资金"], errors="coerce").dropna()
        median_cap = cap.median() if not cap.empty else 0
        cap_score = int(np.clip(median_cap / 5e7 * 50, 0, 100))
    else:
        cap_score = 40

    # 5) 炸板率（反向）
    if not zt.empty and "炸板次数" in zt.columns:
        blasted = (pd.to_numeric(zt["炸板次数"], errors="coerce") > 0).sum()
        blast_rate = blasted / zt_count if zt_count else 0
        blast_score = int((1 - blast_rate) * 100)
    else:
        blast_score = 60

    # 6) 赚钱效应（昨日涨停今日平均涨幅）
    if prev is not None and not prev.empty and "今日涨跌幅" in prev.columns:
        avg_ret = pd.to_numeric(prev["今日涨跌幅"], errors="coerce").mean()
        profit_score = int(np.clip((avg_ret + 2) / 7 * 100, 0, 100))
    else:
        profit_score = 50

    total_score = (
        ratio_score * 0.20 + height_score * 0.20 + breadth_score * 0.15
        + cap_score * 0.15 + blast_score * 0.10 + profit_score * 0.20
    )
    total_score = int(round(total_score))

    detail = {
        "涨停/跌停比": ratio_score,
        "连板高度": height_score,
        "涨跌比": breadth_score,
        "封板坚决度": cap_score,
        "封板成功率": blast_score,
        "赚钱效应": profit_score,
    }
    counts = {"zt": zt_count, "dt": dt_count, "height": height}
    return total_score, detail, counts


def classify_cycle(total_score, zt_count, dt_count, height):
    if total_score >= 75 and zt_count >= 50 and height >= 3:
        return "高潮期", "市场情绪亢奋，涨停潮+高连板", "🔴"
    if total_score >= 65 and zt_count >= 30:
        return "发酵期", "赚钱效应扩散，主线清晰", "🟡"
    if total_score >= 50:
        return "分歧期", "多空拉锯，注意切换", "🟠"
    if total_score >= 35 and zt_count < 30:
        return "退潮期", "涨停萎缩，追高易亏", "⚫"
    if total_score < 35 or (dt_count > zt_count * 0.6 and zt_count < 20):
        return "冰点期", "恐慌极致，等待反转", "🔵"
    return "低位震荡", "情绪偏弱，观望为主", "⚪"


# ─────────────────────────────────────────────
# 第 2 层：主线识别
# ─────────────────────────────────────────────
def build_concept_strength(zt_pool, offline=False):
    """
    返回 DataFrame: 概念, 涨停家数, 板块涨跌幅, 综合得分, 代表龙头
    策略：
      - 基础：涨停池"所属行业"统计涨停家数（最稳）
      - 增强：概念板块列表提供"板块涨跌幅"和"领涨股票"
      - 演示模式：用 _demo_data 里手工整理的真实涨幅
    """
    if zt_pool is None or zt_pool.empty:
        return pd.DataFrame()

    # 1) 行业涨停家数（始终可用）
    industry_zt = pd.DataFrame()
    if "所属行业" in zt_pool.columns:
        industry_zt = zt_pool.groupby("所属行业").size().reset_index(name="涨停家数")

    # 2) 概念板块列表（联网 / 演示）
    concept_df = pd.DataFrame()
    if offline or not HAS_AK:
        if HAS_DEMO:
            concept_df = get_demo_concepts()
    else:
        concept_df = _safe_ak(ak.stock_board_concept_name_em)
        if concept_df is None:
            concept_df = get_demo_concepts() if HAS_DEMO else pd.DataFrame()

    # 3) 合并：以行业涨停为骨架，用概念表补"板块涨跌幅"和"领涨股票"
    #    概念名 → 行业名 做模糊匹配
    zt_names = set(zt_pool["名称"].astype(str).tolist()) if "名称" in zt_pool.columns else set()

    # 先把概念表整理成 dict: name -> {chg, leader}
    concept_map = {}
    if concept_df is not None and not concept_df.empty:
        col_chg = "板块涨跌幅" if "板块涨跌幅" in concept_df.columns else "涨跌幅"
        col_leader = "领涨股票" if "领涨股票" in concept_df.columns else None
        for _, r in concept_df.iterrows():
            nm = str(r.get("板块名称", "") or "").strip()
            chg = _to_num(r.get(col_chg))
            leader = str(r.get(col_leader, "") or "") if col_leader else ""
            if nm:
                concept_map[nm] = {
                    "chg": chg if not pd.isna(chg) else 0.0,
                    "leader": leader,
                }

    rows = []
    matched_concepts = set()
    if not industry_zt.empty:
        for _, r in industry_zt.iterrows():
            ind = str(r["所属行业"])
            zt_n = int(r["涨停家数"])
            # 找概念表里名字相近的
            chg, leader = 0.0, ""
            for cname, cval in concept_map.items():
                if cname == ind or cname in ind or ind in cname:
                    chg = cval["chg"]
                    leader = cval["leader"]
                    matched_concepts.add(cname)
                    break
            rows.append({
                "概念": ind,
                "涨停家数": zt_n,
                "板块涨跌幅": chg,
                "代表龙头": leader if leader in zt_names else "",
            })

    # 概念表中未匹配到的、且涨幅够大的，也补进来
    for cname, cval in concept_map.items():
        if cname in matched_concepts:
            continue
        if cval["chg"] < 1.5:
            continue
        rows.append({
            "概念": cname,
            "涨停家数": 1 if cval["leader"] in zt_names else 0,
            "板块涨跌幅": cval["chg"],
            "代表龙头": cval["leader"] if cval["leader"] in zt_names else "",
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["涨停家数"] = df["涨停家数"].fillna(0).astype(int)
    df["板块涨跌幅"] = df["板块涨跌幅"].fillna(0)
    max_zt = df["涨停家数"].max() or 1
    max_chg = df["板块涨跌幅"].max() or 1
    df["综合得分"] = (
        df["涨停家数"] / max_zt * 60
        + df["板块涨跌幅"] / max_chg * 40
    ).round(1)
    return df.sort_values("综合得分", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# 第 3 层：龙头 / 中军
# ─────────────────────────────────────────────
def pick_leaders_and_zhongjun(zt_pool, top_concepts):
    """在主线概念对应的涨停股里识别龙头与中军"""
    if zt_pool is None or zt_pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    zt = zt_pool.copy()
    zt["连板数"] = pd.to_numeric(zt["连板数"], errors="coerce").fillna(0)

    # 龙头：连板>=2，按连板↓、封板资金↓排序
    lead = zt[zt["连板数"] >= 2].copy()
    if lead.empty:
        lead = zt.head(5).copy()
    for c in ["封板资金", "最新价", "流通市值"]:
        if c in lead.columns:
            lead[c] = pd.to_numeric(lead[c], errors="coerce").fillna(0)

    if "封板资金" in lead.columns:
        lead = lead.sort_values(["连板数", "封板资金"], ascending=[False, False])
    else:
        lead = lead.sort_values("连板数", ascending=False)
    lead = lead.head(6).reset_index(drop=True)

    lead_out = lead.rename(columns={
        "连板数": "连板", "最新价": "现价",
        "封板资金": "封单(亿)", "所属行业": "行业"
    })
    if "封单(亿)" in lead_out.columns:
        lead_out["封单(亿)"] = (lead_out["封单(亿)"] / 1e8).round(2)
    keep = [c for c in ["名称","代码","连板","现价","封单(亿)","行业"] if c in lead_out.columns]
    lead_out = lead_out[keep]

    # 中军：流通市值 100~800 亿的大票涨停
    if "流通市值" in zt.columns:
        zj = zt[(zt["流通市值"] >= 1e10) & (zt["流通市值"] <= 8e10)].copy()
        zj["流通市值"] = pd.to_numeric(zj["流通市值"], errors="coerce")
        zj = zj.sort_values("流通市值", ascending=False).head(6)
        zj_out = zj.rename(columns={
            "流通市值": "流通市值(亿)", "最新价": "现价",
            "连板数": "连板", "所属行业": "行业"
        })
        zj_out["流通市值(亿)"] = (zj_out["流通市值(亿)"] / 1e8).round(1)
        keep2 = [c for c in ["名称","代码","流通市值(亿)","现价","连板","行业"] if c in zj_out.columns]
        zj_out = zj_out[keep2]
    else:
        zj_out = pd.DataFrame()

    return lead_out, zj_out


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def run(date_str=None, force_offline=False):
    if date_str is None:
        date_str = _latest_trade_date()
    offline = force_offline or not HAS_AK
    print(f"\n========== 市场情绪周期复盘 {date_str} ==========\n")

    # 1. 情绪
    data = fetch_sentiment_data(date_str, offline=offline)
    zt_pool = data["zt_pool"]
    src = data.get("source", "demo")
    print(f"  · 涨停家数: {len(zt_pool)}  (数据源: {'实盘' if src=='live' else '离线演示'})")
    time.sleep(0.3)

    score, detail, cnt = calc_sentiment_score(data)
    stage, stage_desc, emoji = classify_cycle(
        score, cnt["zt"], cnt["dt"], cnt["height"]
    )

    print(f"\n{emoji} 当前情绪周期: 【{stage}】 — {stage_desc}")
    print(f"   综合情绪分: {score} / 100")
    print("   分项明细:")
    for k, v in detail.items():
        bar = "█" * (v // 5)
        print(f"     · {k:^8s}: {v:>3d}  {bar}")

    # 2. 主线
    print("\n[*] 正在识别主线板块 …")
    concept_df = build_concept_strength(zt_pool, offline=(src == "demo"))
    time.sleep(0.3)

    top_n = 8
    main_lines = concept_df.head(top_n).reset_index(drop=True)
    print(f"\n📊 板块强度 TOP {top_n}:")
    for i, r in main_lines.iterrows():
        star = "★" if i < 2 else "·"
        print(
            f"  {star} {r['概念']:^12s}  "
            f"涨停 {int(r['涨停家数']):>2d}  "
            f"板块涨 {r['板块涨跌幅']:>5.2f}%  "
            f"得分 {r['综合得分']:>5.1f}  "
            f"领涨: {r.get('代表龙头','')}"
        )

    # 3. 龙头 / 中军
    top_concepts = main_lines.head(3)["概念"].tolist() if not main_lines.empty else []
    print("\n[*] 正在筛选龙头与中军 …")
    leaders, zhongjun = pick_leaders_and_zhongjun(zt_pool, top_concepts)

    print("\n🚀 龙头梯队 (高连板 / 强封单):")
    if not leaders.empty:
        print(leaders.to_string(index=False))
    else:
        print("  (今日无清晰龙头梯队)")

    print("\n🛡️  中军名单 (大市值涨停票):")
    if not zhongjun.empty:
        print(zhongjun.to_string(index=False))
    else:
        print("  (今日无明显中军)")

    # 4. 一句话复盘
    print("\n========== 一句话复盘 ==========")
    if not main_lines.empty:
        line1 = main_lines.iloc[0]
        print(
            f"· 绝对主线: {line1['概念']}"
            f"(涨停 {int(line1['涨停家数'])} 家, 板块 +{line1['板块涨跌幅']:.2f}%)"
        )
        if len(main_lines) > 1:
            line2 = main_lines.iloc[1]
            print(
                f"· 次主线:   {line2['概念']}"
                f"(涨停 {int(line2['涨停家数'])} 家, 板块 +{line2['板块涨跌幅']:.2f}%)"
            )
    if not leaders.empty:
        t = leaders.iloc[0]
        price_col = "现价" if "现价" in t.index else leaders.columns[-1]
        print(
            f"· 空间龙头: {t['名称']}({t['代码']}) "
            f"{int(t['连板'])}连板 @ {t[price_col]}"
        )
    print(f"· 情绪阶段: {emoji} {stage} (分 {score}/100)")
    if src == "demo":
        print("  ⚠️ 当前为离线演示数据，实盘请在联网环境运行。")
    print()

    return {
        "date": date_str,
        "source": src,
        "stage": stage,
        "score": score,
        "detail": detail,
        "main_lines": main_lines,
        "leaders": leaders,
        "zhongjun": zhongjun,
    }


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else None
    offline = "--offline" in sys.argv
    run(d, force_offline=offline)
