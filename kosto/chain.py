"""產業鏈族群標籤（例：電子中游-半導體｜IC/晶圓製造）＋題材（AI、雲端、低軌衛星…）.

資料來源：櫃買中心「產業價值鏈資訊平台」(ic.tpex.org.tw) 的公司 → 產業鏈區段對照（kosto/data/ic_*.txt）。
一家公司常同時列在多條產業鏈；主族群依「官方產業別最相符的產業鏈 → 該鏈中出現區段最多 → 區段最專精（成員最少）」選出，
其餘主題型產業鏈（人工智慧、雲端運算…）列為題材。不在平台上的公司以官方產業別代替。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("data")

THEMES = {"人工智慧": "AI", "雲端運算": "雲端", "資通訊安全": "資安", "大數據": "大數據", "區塊鏈": "區塊鏈",
          "金融科技": "金融科技", "體驗科技": "AR/VR", "運動科技": "運動科技", "太空衛星科技": "低軌衛星",
          "自動化": "機器人/自動化", "再生醫療": "再生醫療", "電動車輛": "電動車", "太陽能": "太陽能", "風力發電": "風電",
          "汽電共生": "汽電共生", "智慧電網": "智慧電網", "能源元件": "儲能電池", "LED照明": "LED"}
PURE_THEME = {"人工智慧", "雲端運算", "資通訊安全", "大數據", "區塊鏈", "金融科技", "體驗科技", "運動科技", "太空衛星科技",
              "自動化", "汽電共生", "智慧電網", "再生醫療"}
ELEC = {"半導體", "被動元件", "通信網路", "連接器", "電腦週邊", "平面顯示器", "觸控面板", "印刷電路板", "LED照明", "能源元件",
        "軟體服務", "電子商務", "人工智慧", "雲端運算", "資通訊安全", "大數據", "區塊鏈", "體驗科技", "太空衛星科技"}
BIO = {"製藥", "醫療器材", "食品生技", "再生醫療"}
FIN = {"金融", "金融科技"}
GREEN = {"太陽能", "風力發電", "汽電共生", "智慧電網"}
# 官方產業別 → 最可能的產業鏈（依序）
SECTOR_CHAINS = {
    "半導體業": ["半導體"], "電腦及週邊設備業": ["電腦週邊"], "光電業": ["平面顯示器", "LED照明", "觸控面板", "太陽能"],
    "通信網路業": ["通信網路"], "電子零組件業": ["被動元件", "連接器", "印刷電路板", "電腦週邊", "通信網路", "能源元件", "半導體"],
    "電子通路業": ["半導體", "電腦週邊", "通信網路"], "資訊服務業": ["軟體服務", "電子商務"],
    "其他電子業": ["電腦週邊", "通信網路", "其他"], "其他電子類": ["電腦週邊", "通信網路", "其他"],
    "電子工業": ["半導體", "電腦週邊", "通信網路", "被動元件", "印刷電路板", "連接器", "平面顯示器", "觸控面板", "LED照明", "能源元件", "軟體服務"],
    "生技醫療業": ["製藥", "醫療器材", "食品生技", "再生醫療"], "化學生技醫療": ["製藥", "醫療器材", "食品生技", "石化及塑橡膠"],
    "水泥工業": ["水泥"], "食品工業": ["食品"], "塑膠工業": ["石化及塑橡膠"], "化學工業": ["石化及塑橡膠"], "橡膠工業": ["石化及塑橡膠"],
    "紡織纖維": ["紡織"], "電機機械": ["電機機械", "自動化"], "電器電纜": ["電機機械", "風力發電", "智慧電網"], "鋼鐵工業": ["鋼鐵"],
    "汽車工業": ["汽車", "電動車輛"], "建材營造": ["建材營造"], "航運業": ["交通運輸及航運"], "觀光餐旅": ["休閒娛樂", "食品"],
    "觀光事業": ["休閒娛樂"], "金融保險": ["金融"], "金融業": ["金融"], "貿易百貨": ["貿易百貨"], "油電燃氣業": ["油電燃氣"],
    "造紙工業": ["造紙"], "玻璃陶瓷": ["建材營造", "平面顯示器"], "綠能環保": ["太陽能", "風力發電", "能源元件", "其他"],
    "綠能環保類": ["太陽能", "風力發電", "能源元件", "其他"], "數位雲端": ["軟體服務", "電子商務"], "數位雲端類": ["軟體服務", "電子商務"],
    "運動休閒": ["休閒娛樂", "紡織"], "運動休閒類": ["休閒娛樂", "紡織"], "居家生活": ["貿易百貨", "其他"], "居家生活類": ["貿易百貨", "其他"],
    "文化創意業": ["文化創意"], "農業科技業": ["其他"],
}
# 多產業鏈大型股的主業（平台未標示主從，人工指定；其餘依規則推定）
OVERRIDE = {
    "2317": ("電子下游-電腦週邊", "伺服器、EMS代工"), "2308": ("電子上游-電腦週邊", "電源供應器、散熱"),
    "2357": ("電子下游-電腦週邊", "筆記型電腦、主機板"), "2382": ("電子下游-電腦週邊", "伺服器、筆記型電腦"),
    "3231": ("電子下游-電腦週邊", "伺服器、筆記型電腦"), "2376": ("電子上游-電腦週邊", "主機板、顯示卡"),
    "2324": ("電子下游-電腦週邊", "筆記型電腦、伺服器"), "2353": ("電子下游-電腦週邊", "筆記型電腦、桌上型電腦"),
    "4938": ("電子下游-電腦週邊", "EMS代工、筆記型電腦"), "2356": ("電子下游-電腦週邊", "伺服器、筆記型電腦"),
    "2301": ("電子上游-電腦週邊", "電源供應器、光電元件"), "2377": ("電子上游-電腦週邊", "主機板、筆記型電腦"),
    "1513": ("傳產下游-電機機械", "重電、機電系統工程"), "1519": ("傳產下游-電機機械", "重電（變壓器）"),
    "1503": ("傳產下游-電機機械", "重電、馬達"), "1504": ("傳產下游-電機機械", "馬達、重電"),
    "2207": ("傳產下游-汽車", "汽車代理銷售"), "1605": ("傳產中游-鋼鐵", "不鏽鋼、電線電纜"),
    "2371": ("傳產下游-電機機械", "重電、家電"), "2395": ("電子下游-電腦週邊", "工業電腦"),
    "2345": ("電子下游-通信網路", "網路交換器"), "3045": ("電子下游-通信網路", "電信服務業"),
    "4904": ("電子下游-通信網路", "電信服務業"), "2412": ("電子下游-通信網路", "電信服務業"),
}
SECTOR_GROUP = {"半導體業": "電子", "電腦及週邊設備業": "電子", "光電業": "電子", "通信網路業": "電子", "電子零組件業": "電子",
                "電子通路業": "電子", "資訊服務業": "電子", "其他電子業": "電子", "其他電子類": "電子", "電子工業": "電子",
                "數位雲端": "電子", "數位雲端類": "電子", "生技醫療業": "生技", "化學生技醫療": "生技", "金融保險": "金融",
                "金融業": "金融", "綠能環保": "綠能", "綠能環保類": "綠能"}


@lru_cache(maxsize=1)
def _load():
    segs = []
    for line in (DATA / "ic_segments.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            chain, sid, pos, name = line.split("|")
            segs.append((chain, sid, pos, name))
    mp = {}
    for line in (DATA / "ic_map.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            code, _, idx = line.partition(":")
            mp[code] = [int(x) for x in idx.split(",") if x != ""]
    members = [0] * len(segs)
    for v in mp.values():
        for k in v:
            members[k] += 1
    return segs, mp, members


def _group(chain: str) -> str:
    if chain in ELEC:
        return "電子"
    if chain in BIO:
        return "生技"
    if chain in FIN:
        return "金融"
    if chain in GREEN:
        return "綠能"
    return "傳產"


def _pos(p: str) -> str:
    return p[:2] if p[:2] in ("上游", "中游", "下游") else ""


def label(ticker: str, sector: str = "") -> dict:
    """{'group': 族群(電子中游-半導體), 'segment': 區段, 'label': 完整標籤, 'themes': [題材…], 'chain': 產業鏈}"""
    segs, mp, members = _load()
    entries = mp.get(str(ticker), [])
    sector = (sector or "").strip()
    if sector.isdigit():sector=''  # unresolved numeric industry code is not an industry name
    if not entries:
        g = SECTOR_GROUP.get(sector, "傳產" if sector not in ("", "NA", "其他") else "其他")
        name = sector if sector not in ("", "NA") else "其他"
        return {"group": f"{g}-{name}" if g != "其他" else name, "segment": "", "label": f"{g}-{name}" if g != "其他" else name,
                "themes": [], "chain": ""}
    by_chain: dict[str, list[int]] = {}
    for k in entries:
        by_chain.setdefault(segs[k][0], []).append(k)
    core = [c for c in by_chain if c not in PURE_THEME] or list(by_chain)
    pref = SECTOR_CHAINS.get(sector, [])
    in_pref = [c for c in core if c in pref]
    pool = in_pref or core

    def key(c):
        return (-len(by_chain[c]), pref.index(c) if c in pref else 99)
    chain = sorted(pool, key=key)[0]
    ks = by_chain[chain]
    cnt = {}
    for k in ks:
        p = _pos(segs[k][2])
        cnt[p] = cnt.get(p, 0) + 1
    order_pos = {"下游": 0, "中游": 1, "上游": 2, "": 3}
    best_pos = sorted(cnt, key=lambda p: (-cnt[p], order_pos.get(p, 3)))[0]
    ks = sorted([k for k in ks if _pos(segs[k][2]) == best_pos], key=lambda k: (segs[k][3].startswith("其他"), members[k], k))
    pos = best_pos
    g = _group(chain)
    head = f"{g}{pos}-{chain}" if pos else f"{g}-{chain}"
    seg_names = []
    for k in ks:
        nm = segs[k][3]
        if nm not in seg_names:
            seg_names.append(nm)
    seg = "／".join(seg_names[:2])
    if chain == "其他":
        g = SECTOR_GROUP.get(sector, "傳產")
        nm = sector if sector not in ("", "NA", "其他", "電子工業") else "其他"
        head = f"{g}-{nm}"
        seg = "" if seg.startswith("其他") and seg == "其他" else seg
    if str(ticker) in OVERRIDE:
        head, seg = OVERRIDE[str(ticker)]
    themes = []
    for c in by_chain:
        if c != chain and c in THEMES and THEMES[c] not in themes:
            themes.append(THEMES[c])
    return {"group": head, "segment": seg, "label": f"{head}｜{seg}" if seg else head, "themes": themes[:4], "chain": chain}


def labels(tickers, sectors) -> list[dict]:
    return [label(t, s) for t, s in zip(tickers, sectors)]
