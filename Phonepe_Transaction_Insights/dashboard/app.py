import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PhonePe Pulse", page_icon="💜", layout="wide", initial_sidebar_state="expanded")

P, D, C, A, G = "#5f259f", "#13002b", "#1e0645", "#8b2fc9", "#f7c948"

st.markdown(f"""<style>
html,body,[class*="css"]{{background:{D};color:#e8d5f5;font-family:'Segoe UI',sans-serif;}}
section[data-testid="stSidebar"]{{background:{C};border-right:2px solid {P};}}
.kpi{{background:linear-gradient(135deg,{C},{P});border-radius:14px;padding:18px 14px;text-align:center;border:1px solid {A};margin:4px;}}
.kpi h3{{font-size:13px;color:#cbb8e8;margin:0;}}
.kpi h1{{font-size:26px;font-weight:700;color:{G};margin:4px 0 0;}}
.stTabs [data-baseweb="tab"]{{background:{C};color:#cbb8e8;border-radius:8px 8px 0 0;padding:8px 18px;}}
.stTabs [aria-selected="true"]{{background:{P};color:white;}}
div[data-testid="stMetric"]{{background:{C};border-radius:10px;padding:10px;border:1px solid {A};}}
.stSelectbox>div>div{{background:{C};color:#e8d5f5;border:1px solid {P};}}
h1,h2,h3{{color:#e8d5f5;}}
.block-container{{padding:1.5rem 2rem;}}
</style>""", unsafe_allow_html=True)

# ── LOAD DATA FROM CSV ───────────────────────────────────────────────
@st.cache_data
def load():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    agg_t = pd.read_csv(f"{base}/agg_transaction.csv")
    agg_u = pd.read_csv(f"{base}/agg_users.csv")
    agg_i = pd.read_csv(f"{base}/agg_insurance.csv")
    map_t = pd.read_csv(f"{base}/map_transaction.csv")
    map_u = pd.read_csv(f"{base}/map_users.csv")
    map_i = pd.read_csv(f"{base}/map_insurance.csv")
    top_t = pd.read_csv(f"{base}/top_transaction.csv")
    top_u = pd.read_csv(f"{base}/top_users.csv")
    top_i = pd.read_csv(f"{base}/top_insurance.csv")
    return agg_t, agg_u, agg_i, map_t, map_u, map_i, top_t, top_u, top_i

agg_t, agg_u, agg_i, map_t, map_u, map_i, top_t, top_u, top_i = load()

# ── HELPERS ──────────────────────────────────────────────────────────
def theme(fig, h=380):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,6,69,0.6)",
        font=dict(color="#e8d5f5", size=12), height=h,
        margin=dict(l=30,r=30,t=40,b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=P),
        xaxis=dict(gridcolor="rgba(95,37,159,0.3)", linecolor=P),
        yaxis=dict(gridcolor="rgba(95,37,159,0.3)", linecolor=P)
    )
    return fig

def kpi(col, label, value, delta=None):
    col.markdown(f'<div class="kpi"><h3>{label}</h3><h1>{value}</h1>{"<p style=color:#90ee90;font-size:12px>"+delta+"</p>" if delta else ""}</div>', unsafe_allow_html=True)

def fmt(n):
    if n>=1e12: return f"Rs.{n/1e12:.2f}T"
    if n>=1e9:  return f"Rs.{n/1e9:.2f}B"
    if n>=1e7:  return f"Rs.{n/1e7:.2f}Cr"
    return f"Rs.{n:,.0f}"

def fmtn(n):
    if n>=1e9: return f"{n/1e9:.2f}B"
    if n>=1e6: return f"{n/1e6:.2f}M"
    if n>=1e3: return f"{n/1e3:.1f}K"
    return str(int(n))

def period(df):
    df = df.copy()
    df["period"] = df["year"].astype(str)+"-Q"+df["quarter"].astype(str)
    return df

def filt(df, state_col="state", year_col="year", qtr_col="quarter"):
    d = df.copy()
    if sel_state != "All": d = d[d[state_col]==sel_state]
    if sel_year  != "All": d = d[d[year_col]==sel_year]
    if sel_qtr   != "All": d = d[d[qtr_col]==sel_qtr]
    return d

CSCALE = [[0,"#13002b"],[0.5,"#5f259f"],[1,"#f7c948"]]

# ── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='text-align:center;padding:16px 0 20px'>
        <div style='display:inline-flex;align-items:center;justify-content:center;
                    width:80px;height:80px;border-radius:50%;
                    background:linear-gradient(135deg,{P},{A});
                    box-shadow:0 0 24px rgba(95,37,159,0.7);margin-bottom:10px'>
            <svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="22" cy="22" r="20" fill="#5f259f" stroke="#f7c948" stroke-width="2"/>
                <text x="22" y="17" text-anchor="middle" font-size="9" fill="#f7c948" font-weight="bold" font-family="Arial">UPI</text>
                <path d="M12 22 Q22 14 32 22 Q22 30 12 22 Z" fill="#f7c948" opacity="0.9"/>
                <circle cx="22" cy="22" r="3" fill="#13002b"/>
                <path d="M19 28 L22 34 L25 28" stroke="#f7c948" stroke-width="2" fill="none" stroke-linecap="round"/>
            </svg>
        </div>
        <div style='font-size:22px;font-weight:700;color:{G}'>PhonePe Pulse</div>
        <div style='font-size:11px;color:#cbb8e8'>Transaction Intelligence Dashboard</div>
        <hr style='border-color:{P};margin:12px 0'>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", ["Overview","Transactions","Users","Insurance","Geo Maps","Business Cases"], label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{P}'>", unsafe_allow_html=True)
    st.markdown("**Global Filters**")

    all_states = ["All"] + sorted(agg_t["state"].unique().tolist())
    all_years  = ["All"] + sorted(agg_t["year"].unique().tolist())
    all_qtrs   = ["All", 1, 2, 3, 4]

    sel_state = st.selectbox("State",   all_states)
    sel_year  = st.selectbox("Year",    all_years)
    sel_qtr   = st.selectbox("Quarter", all_qtrs)

    st.markdown(f"""<div style='margin-top:30px;padding:10px;background:{C};border-radius:10px;border:1px solid {P};font-size:11px;color:#cbb8e8;text-align:center'>
        CSV Mode · No DB Required<br>Data: 2018-2024 · 9 Tables
    </div>""", unsafe_allow_html=True)

# ── PAGE 1: OVERVIEW ─────────────────────────────────────────────────
if page == "Overview":
    st.markdown(f"<h1 style='color:{G}'>PhonePe Pulse - Overview</h1>", unsafe_allow_html=True)

    ft = filt(agg_t); fu = filt(agg_u); fi = filt(agg_i); fm = filt(map_u)

    total_amt   = ft["transaction_amount"].sum()
    total_cnt   = ft["transaction_count"].sum()
    avg_txn     = total_amt/total_cnt if total_cnt else 0
    total_users = fu["registered_users"].sum()
    app_opens   = fm["app_opens"].sum()
    states_cnt  = ft["state"].nunique()
    dist_cnt    = filt(map_t)["district"].nunique()
    ins_cnt     = fi["policy_count"].sum()
    ins_amt     = fi["premium_amount"].sum()
    avg_prem    = ins_amt/ins_cnt if ins_cnt else 0
    brands      = fu["brand"].nunique()
    yoy = agg_t.groupby("year")["transaction_amount"].sum()
    yoy_g = f"+{((yoy.iloc[-1]-yoy.iloc[-2])/yoy.iloc[-2]*100):.1f}% YoY" if len(yoy)>1 else ""

    r1 = st.columns(4)
    kpi(r1[0],"Total Txn Amount",   fmt(total_amt),   yoy_g)
    kpi(r1[1],"Total Txn Count",    fmtn(total_cnt))
    kpi(r1[2],"Avg Txn Value",      f"Rs.{avg_txn:,.0f}")
    kpi(r1[3],"Registered Users",   fmtn(total_users))

    r2 = st.columns(4)
    kpi(r2[0],"App Opens",          fmtn(app_opens))
    kpi(r2[1],"Active States",      str(states_cnt))
    kpi(r2[2],"Districts",          str(dist_cnt))
    kpi(r2[3],"Mobile Brands",      str(brands))

    r3 = st.columns(4)
    kpi(r3[0],"Total Policies",     fmtn(ins_cnt))
    kpi(r3[1],"Insurance Premium",  fmt(ins_amt))
    kpi(r3[2],"Avg Premium/Policy", f"Rs.{avg_prem:,.0f}")
    kpi(r3[3],"Years Covered",      f"{agg_t['year'].min()}-{agg_t['year'].max()}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    qt = period(agg_t.groupby(["year","quarter"])["transaction_amount"].sum().reset_index())
    f1 = px.area(qt, x="period", y="transaction_amount", title="Quarterly Txn Amount Trend",
                 color_discrete_sequence=[A], labels={"transaction_amount":"Amount","period":"Quarter"})
    f1.update_traces(fill='tozeroy', fillcolor="rgba(95,37,159,0.3)")
    c1.plotly_chart(theme(f1), use_container_width=True)

    tt = agg_t.groupby("transaction_type")["transaction_amount"].sum().reset_index()
    f2 = px.pie(tt, names="transaction_type", values="transaction_amount", title="Txn Type Distribution",
                color_discrete_sequence=px.colors.sequential.Purples_r, hole=0.45)
    f2.update_traces(textfont_color="white")
    c2.plotly_chart(theme(f2), use_container_width=True)

# ── PAGE 2: TRANSACTIONS ─────────────────────────────────────────────
elif page == "Transactions":
    st.markdown(f"<h1 style='color:{G}'>Transaction Analytics</h1>", unsafe_allow_html=True)
    ft = filt(agg_t)

    t1,t2,t3,t4 = st.tabs(["Trends","Type Analysis","State Rankings","Heatmap"])

    with t1:
        qt = period(ft.groupby(["year","quarter"])[["transaction_amount","transaction_count"]].sum().reset_index())
        c1,c2 = st.columns(2)
        f1 = px.line(qt,x="period",y="transaction_amount",title="Quarterly Amount Trend",markers=True,color_discrete_sequence=[G])
        f2 = px.bar(qt,x="period",y="transaction_count",title="Quarterly Count Trend",color="transaction_count",color_continuous_scale=CSCALE)
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        yy = ft.groupby("year")[["transaction_amount","transaction_count"]].sum().reset_index()
        yy["yoy_amt"] = yy["transaction_amount"].pct_change()*100
        f3 = px.bar(yy,x="year",y="yoy_amt",title="YoY Growth % (Amount)",color="yoy_amt",
                    color_continuous_scale=[[0,"#d62728"],[0.5,P],[1,G]])
        st.plotly_chart(theme(f3,280),use_container_width=True)

    with t2:
        tt = ft.groupby("transaction_type")[["transaction_amount","transaction_count"]].sum().reset_index()
        c1,c2 = st.columns(2)
        f1 = px.pie(tt,names="transaction_type",values="transaction_amount",title="Share by Amount",hole=0.4,color_discrete_sequence=px.colors.sequential.Purp)
        f2 = px.bar(tt.sort_values("transaction_count",ascending=True),x="transaction_count",y="transaction_type",
                    orientation="h",title="Count by Type",color="transaction_count",color_continuous_scale=CSCALE)
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        ttype_q = period(ft.groupby(["year","quarter","transaction_type"])["transaction_amount"].sum().reset_index())
        f3 = px.area(ttype_q,x="period",y="transaction_amount",color="transaction_type",title="Txn Type Over Time",
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(theme(f3,320),use_container_width=True)

    with t3:
        st_t = ft.groupby("state")[["transaction_amount","transaction_count"]].sum().reset_index()
        c1,c2 = st.columns(2)
        f1 = px.bar(st_t.nlargest(10,"transaction_amount"),x="transaction_amount",y="state",orientation="h",
                    title="Top 10 States by Amount",color="transaction_amount",color_continuous_scale=CSCALE)
        f2 = px.bar(st_t.nlargest(10,"transaction_count"),x="transaction_count",y="state",orientation="h",
                    title="Top 10 States by Count",color="transaction_count",color_continuous_scale=CSCALE)
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        avg_v = ft.groupby("state").apply(lambda x: x["transaction_amount"].sum()/x["transaction_count"].sum()).reset_index(name="avg_val")
        f3 = px.bar(avg_v.nlargest(15,"avg_val"),x="state",y="avg_val",title="Top 15 States - Avg Txn Value",
                    color="avg_val",color_continuous_scale=CSCALE)
        st.plotly_chart(theme(f3,300),use_container_width=True)

    with t4:
        heat = ft.groupby(["state","year"])["transaction_amount"].sum().unstack(fill_value=0)
        heat = heat.loc[heat.sum(axis=1).nlargest(15).index]
        f = px.imshow(heat/1e12,title="State x Year Heatmap (Rs.T)",color_continuous_scale=CSCALE,
                      labels=dict(color="Rs.T"),aspect="auto",text_auto=".1f")
        st.plotly_chart(theme(f,500),use_container_width=True)
        seas = ft.groupby("quarter")["transaction_amount"].mean().reset_index()
        seas["quarter"] = seas["quarter"].map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
        f2 = px.bar(seas,x="quarter",y="transaction_amount",title="Seasonality - Avg Amount by Quarter",
                    color="transaction_amount",color_continuous_scale=CSCALE)
        st.plotly_chart(theme(f2,280),use_container_width=True)

# ── PAGE 3: USERS ────────────────────────────────────────────────────
elif page == "Users":
    st.markdown(f"<h1 style='color:{G}'>User Analytics</h1>", unsafe_allow_html=True)
    fu = filt(agg_u); fm = filt(map_u)

    t1,t2,t3 = st.tabs(["Brand Analysis","State Analysis","Growth Trends"])

    with t1:
        br = fu.groupby("brand")["registered_users"].sum().reset_index().sort_values("registered_users",ascending=False)
        c1,c2 = st.columns(2)
        f1 = px.pie(br.head(8),names="brand",values="registered_users",title="Brand Market Share",hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Vivid)
        f2 = px.bar(br.head(15),x="brand",y="registered_users",title="Brand-wise Registered Users",
                    color="registered_users",color_continuous_scale=CSCALE)
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        br_y = period(fu.groupby(["year","quarter","brand"])["registered_users"].sum().reset_index())
        top5b = br.head(5)["brand"].tolist()
        f3 = px.line(br_y[br_y["brand"].isin(top5b)],x="period",y="registered_users",color="brand",
                     title="Top 5 Brands - Growth Over Time",markers=True,
                     color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(theme(f3,320),use_container_width=True)

    with t2:
        su = fm.groupby("state")[["registered_users","app_opens"]].sum().reset_index()
        c1,c2 = st.columns(2)
        f1 = px.bar(su.nlargest(15,"registered_users"),x="state",y="registered_users",
                    title="Top 15 States - Registered Users",color="registered_users",color_continuous_scale=CSCALE)
        f2 = px.bar(su.nlargest(15,"app_opens"),x="state",y="app_opens",
                    title="Top 15 States - App Opens",color="app_opens",color_continuous_scale=CSCALE)
        f1.update_xaxes(tickangle=45); f2.update_xaxes(tickangle=45)
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        du = fm.groupby("district")["registered_users"].sum().reset_index().nlargest(10,"registered_users")
        f3 = px.bar(du,x="registered_users",y="district",orientation="h",title="Top 10 Districts - Users",
                    color="registered_users",color_continuous_scale=CSCALE)
        st.plotly_chart(theme(f3,320),use_container_width=True)

    with t3:
        qu = period(fu.groupby(["year","quarter"])["registered_users"].sum().reset_index())
        qa = period(fm.groupby(["year","quarter"])["app_opens"].sum().reset_index())
        c1,c2 = st.columns(2)
        f1 = px.area(qu,x="period",y="registered_users",title="User Registration Growth",color_discrete_sequence=[A])
        f2 = px.area(qa,x="period",y="app_opens",title="App Opens Growth",color_discrete_sequence=[G])
        f1.update_traces(fill='tozeroy',fillcolor="rgba(139,47,201,0.3)")
        f2.update_traces(fill='tozeroy',fillcolor="rgba(247,201,72,0.3)")
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        heat_u = fu.groupby(["state","year"])["registered_users"].sum().unstack(fill_value=0)
        heat_u = heat_u.loc[heat_u.sum(axis=1).nlargest(12).index]
        f3 = px.imshow(heat_u/1e6,title="State x Year User Heatmap (Millions)",
                       color_continuous_scale=CSCALE,text_auto=".1f",aspect="auto")
        st.plotly_chart(theme(f3,420),use_container_width=True)

# ── PAGE 4: INSURANCE ────────────────────────────────────────────────
elif page == "Insurance":
    st.markdown(f"<h1 style='color:{G}'>Insurance Analytics</h1>", unsafe_allow_html=True)
    fi = filt(agg_i); fmi = filt(map_i)

    t1,t2,t3 = st.tabs(["Growth","State View","District View"])

    with t1:
        qi = period(fi.groupby(["year","quarter"])[["policy_count","premium_amount"]].sum().reset_index())
        qi["avg_prem"] = qi["premium_amount"]/qi["policy_count"]
        c1,c2 = st.columns(2)
        f1 = px.bar(qi,x="period",y="policy_count",title="Quarterly Policy Count",
                    color="policy_count",color_continuous_scale=CSCALE)
        f2 = px.line(qi,x="period",y="premium_amount",title="Quarterly Premium Growth",
                     markers=True,color_discrete_sequence=[G])
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        f3 = px.line(qi,x="period",y="avg_prem",title="Avg Premium per Policy Trend",
                     markers=True,color_discrete_sequence=[A])
        yi = fi.groupby("year")[["policy_count","premium_amount"]].sum().reset_index()
        yi["yoy"] = yi["premium_amount"].pct_change()*100
        f4 = px.bar(yi,x="year",y="yoy",title="YoY Insurance Premium Growth %",
                    color="yoy",color_continuous_scale=[[0,"#d62728"],[0.5,P],[1,G]])
        c3,c4 = st.columns(2)
        c3.plotly_chart(theme(f3),use_container_width=True)
        c4.plotly_chart(theme(f4),use_container_width=True)

    with t2:
        si = fi.groupby("state")[["policy_count","premium_amount"]].sum().reset_index()
        c1,c2 = st.columns(2)
        f1 = px.bar(si.nlargest(15,"policy_count"),x="state",y="policy_count",
                    title="Top 15 States - Policies",color="policy_count",color_continuous_scale=CSCALE)
        f2 = px.bar(si.nlargest(15,"premium_amount"),x="state",y="premium_amount",
                    title="Top 15 States - Premium",color="premium_amount",color_continuous_scale=CSCALE)
        f1.update_xaxes(tickangle=45); f2.update_xaxes(tickangle=45)
        c1.plotly_chart(theme(f1),use_container_width=True)
        c2.plotly_chart(theme(f2),use_container_width=True)
        f3 = px.scatter(si,x="policy_count",y="premium_amount",text="state",
                        title="Policy Count vs Premium - State Scatter",size="premium_amount",
                        color="premium_amount",color_continuous_scale=CSCALE)
        f3.update_traces(textposition="top center",textfont_size=9)
        st.plotly_chart(theme(f3,400),use_container_width=True)

    with t3:
        di = fmi.groupby("district")[["policy_count","premium_amount"]].sum().reset_index()
        f1 = px.bar(di.nlargest(15,"premium_amount"),x="premium_amount",y="district",
                    orientation="h",title="Top 15 Districts - Insurance Premium",
                    color="premium_amount",color_continuous_scale=CSCALE)
        st.plotly_chart(theme(f1,420),use_container_width=True)

# ── PAGE 5: GEO MAPS ─────────────────────────────────────────────────
elif page == "Geo Maps":
    st.markdown(f"<h1 style='color:{G}'>Geographic Distribution</h1>", unsafe_allow_html=True)
    st.info("Bubble maps - size and color represent metric magnitude per state")

    def bubble_map(df, size_col, color_col, title):
        st_coords = {
            "andaman-&-nicobar-islands":(11.7,92.6),"andhra-pradesh":(15.9,79.7),
            "arunachal-pradesh":(28.2,94.7),"assam":(26.2,92.9),"bihar":(25.1,85.3),
            "chandigarh":(30.7,76.7),"chhattisgarh":(21.3,81.9),"dadra-&-nagar-haveli-&-daman-&-diu":(20.4,72.8),
            "delhi":(28.6,77.2),"goa":(15.3,74.0),"gujarat":(22.3,71.2),
            "haryana":(29.1,76.1),"himachal-pradesh":(31.1,77.2),"jammu-&-kashmir":(33.7,75.4),
            "jharkhand":(23.6,85.3),"karnataka":(15.3,75.7),"kerala":(10.9,76.3),
            "ladakh":(34.2,77.6),"lakshadweep":(10.6,72.6),"madhya-pradesh":(22.9,78.7),
            "maharashtra":(19.7,75.7),"manipur":(24.8,93.9),"meghalaya":(25.5,91.4),
            "mizoram":(23.2,92.9),"nagaland":(26.2,94.6),"odisha":(20.9,84.2),
            "puducherry":(11.9,79.8),"punjab":(31.1,75.3),"rajasthan":(27.0,74.2),
            "sikkim":(27.5,88.5),"tamil-nadu":(11.1,78.7),"telangana":(17.4,78.5),
            "tripura":(23.7,91.7),"uttar-pradesh":(26.8,80.9),"uttarakhand":(30.1,79.0),
            "west-bengal":(22.6,88.4)
        }
        df = df.copy()
        df["lat"] = df["state"].map(lambda s: st_coords.get(s,(20,80))[0])
        df["lon"] = df["state"].map(lambda s: st_coords.get(s,(20,80))[1])
        fig = px.scatter_mapbox(df,lat="lat",lon="lon",size=size_col,color=color_col,
                                hover_name="state",title=title,
                                color_continuous_scale=CSCALE,zoom=3.8,height=520,
                                center={"lat":22,"lon":82})
        fig.update_layout(mapbox_style="carto-darkmatter",
                          paper_bgcolor="rgba(0,0,0,0)",font_color="#e8d5f5",
                          margin=dict(l=0,r=0,t=40,b=0))
        return fig

    st_t = agg_t.groupby("state")[["transaction_amount","transaction_count"]].sum().reset_index()
    st_u = map_u.groupby("state")[["registered_users","app_opens"]].sum().reset_index()
    st_i = agg_i.groupby("state")[["policy_count","premium_amount"]].sum().reset_index()

    tab1,tab2,tab3 = st.tabs(["Transaction Map","User Map","Insurance Map"])
    with tab1:
        st.plotly_chart(bubble_map(st_t,"transaction_count","transaction_amount","State-wise Transaction Amount"),use_container_width=True)
    with tab2:
        st.plotly_chart(bubble_map(st_u,"registered_users","app_opens","State-wise Registered Users"),use_container_width=True)
    with tab3:
        st.plotly_chart(bubble_map(st_i,"policy_count","premium_amount","State-wise Insurance Premium"),use_container_width=True)

# ── PAGE 6: BUSINESS CASES ───────────────────────────────────────────
elif page == "Business Cases":
    st.markdown(f"<h1 style='color:{G}'>Business Case Analysis</h1>", unsafe_allow_html=True)

    cases = {
        "Q1 - Top 10 States by Txn Amount": agg_t.groupby("state")[["transaction_count","transaction_amount"]].sum().reset_index().nlargest(10,"transaction_amount"),
        "Q2 - Brand-wise Users": agg_u.groupby("brand")["registered_users"].sum().reset_index().sort_values("registered_users",ascending=False),
        "Q3 - Insurance Growth by Quarter": agg_i.groupby(["year","quarter"])[["policy_count","premium_amount"]].sum().reset_index().sort_values(["year","quarter"]),
        "Q4 - Top 10 Districts by Users": map_u.groupby(["state","district"])["registered_users"].sum().reset_index().nlargest(10,"registered_users"),
        "Q5 - Top 5 Txn Types by Amount": agg_t.groupby("transaction_type")[["transaction_count","transaction_amount"]].sum().reset_index().nlargest(5,"transaction_amount"),
        "Q6 - Yearly Txn Trend": agg_t.groupby("year")[["transaction_count","transaction_amount"]].sum().reset_index().sort_values("year"),
        "Q7 - Top 10 Districts by Txn Amount": map_t.groupby(["state","district"])[["transaction_count","transaction_amount"]].sum().reset_index().nlargest(10,"transaction_amount"),
        "Q8 - Top 10 States by App Opens": map_u.groupby("state")["app_opens"].sum().reset_index().nlargest(10,"app_opens"),
        "Q9 - Top 10 Districts by Insurance": map_i.groupby(["state","district"])[["policy_count","premium_amount"]].sum().reset_index().nlargest(10,"premium_amount"),
    }

    for title, df in cases.items():
        with st.expander(f"{title}", expanded=False):
            c1,c2 = st.columns([1,1])
            c1.dataframe(df.style.background_gradient(cmap="Purples"), use_container_width=True)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols)>=1:
                cat = df.columns[0]
                val = num_cols[-1]
                fig = px.bar(df, x=cat, y=val, title=title,
                             color=val, color_continuous_scale=CSCALE)
                fig.update_xaxes(tickangle=45)
                c2.plotly_chart(theme(fig,320), use_container_width=True)
