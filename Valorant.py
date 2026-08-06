import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="VALORANT 数据分析与预测", layout="wide")

# ---------- 初始化 session_state ----------
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'selected_team' not in st.session_state:
    st.session_state.selected_team = None
if 'selected_team1' not in st.session_state:
    st.session_state.selected_team1 = None
if 'selected_team2' not in st.session_state:
    st.session_state.selected_team2 = None

# ---------- 数据加载与清洗 ----------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')

    required_cols = ['team1', 'team2', 'team1_score', 'team2_score',
                     'first_t_score', 'first_ct_score', 'second_t_score', 'second_ct_score',
                     'map', 'pick_team', 'win', 'first_t_side', 'first_ct_side',
                     'pistol_first', 'pistol_second', 'first_to_3', 'first_to_6', 'first_to_9']

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"缺少关键列: {missing}，请检查Excel格式")
        return None

    df_clean = df.dropna(subset=['team1', 'team2', 'team1_score', 'team2_score', 'map'])
    df_clean = df_clean.dropna(subset=['first_t_score', 'first_ct_score', 'second_t_score', 'second_ct_score'])

    numeric_cols = ['team1_score', 'team2_score', 'first_t_score', 'first_ct_score',
                    'second_t_score', 'second_ct_score']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=numeric_cols)

    if df_clean.empty:
        st.error("清洗后无有效数据")
        return None

    # 补充下半场边（上下半场互换）
    df_clean['second_t_side'] = df_clean['first_ct_side']
    df_clean['second_ct_side'] = df_clean['first_t_side']

    df_clean['total_rounds'] = df_clean['team1_score'] + df_clean['team2_score']
    df_clean['is_ot'] = df_clean['total_rounds'] > 24.5
    df_clean['first_half_diff'] = abs(df_clean['first_t_score'] - df_clean['first_ct_score'])
    df_clean['second_half_diff'] = abs(df_clean['second_t_score'] - df_clean['second_ct_score'])
    df_clean['full_diff'] = abs(df_clean['team1_score'] - df_clean['team2_score'])

    def get_first_half_leader(row):
        if row['first_t_score'] > row['first_ct_score']:
            return row['first_t_side']
        elif row['first_ct_score'] > row['first_t_score']:
            return row['first_ct_side']
        else:
            return None
    df_clean['first_half_leader'] = df_clean.apply(get_first_half_leader, axis=1)
    df_clean['is_comeback'] = (df_clean['first_half_diff'] > 4.5) & (df_clean['win'] != df_clean['first_half_leader']) & (df_clean['first_half_leader'].notna())
    df_clean['second_half_total'] = df_clean['second_t_score'] + df_clean['second_ct_score']

    return df_clean

# ---------- 工具函数 ----------
def fmt_prob(num, den):
    if den == 0:
        return "0 / 0 (0.0%)"
    return f"{int(num)} / {int(den)} ({num/den*100:.1f}%)"

# ---------- 界面1 ----------
def render_map_summary(df):
    st.header("🗺️ Map 数据汇总")
    tabs = st.tabs(["📊 All Map 综合分析", "🏆 战队Pick Map分析"])

    # ===== All Map 综合分析 =====
    with tabs[0]:
        st.subheader("全地图综合分析")
        map_stats = []
        for map_name, group in df.groupby('map'):
            if len(group) < 3:
                continue
            first_diffs = group['first_half_diff']
            second_diffs = group['second_half_diff']
            full_diffs = group['full_diff']

            second_avg = group['second_half_total'].mean()
            if second_avg > 9.5:
                w1, w2 = 0.5, 0.5
            elif second_avg > 7.5:
                w1, w2 = 0.6, 0.4
            elif second_avg > 5.5:
                w1, w2 = 0.7, 0.3
            elif second_avg > 3.5:
                w1, w2 = 0.8, 0.2
            else:
                w1, w2 = 0.9, 0.1

            t_wins_f, ct_wins_f, draws_f = 0, 0, 0
            for _, row in group.iterrows():
                if row['first_t_score'] > row['first_ct_score']:
                    t_wins_f += 1
                elif row['first_ct_score'] > row['first_t_score']:
                    ct_wins_f += 1
                else:
                    draws_f += 1
            t_wins_s, ct_wins_s, draws_s = 0, 0, 0
            for _, row in group.iterrows():
                if row['second_t_score'] > row['second_ct_score']:
                    t_wins_s += 1
                elif row['second_ct_score'] > row['second_t_score']:
                    ct_wins_s += 1
                else:
                    draws_s += 1
            total_f = len(group)
            total_s = len(group)
            t_rate = (t_wins_f/total_f)*w1 + (t_wins_s/total_s)*w2
            ct_rate = (ct_wins_f/total_f)*w1 + (ct_wins_s/total_s)*w2
            draw_rate = (draws_f/total_f)*w1 + (draws_s/total_s)*w2

            pistol_t_wins, pistol_ct_wins, pistol_total = 0, 0, 0
            for _, row in group.iterrows():
                if pd.notna(row['pistol_first']) and row['pistol_first'] != '/':
                    pistol_total += 1
                    if row['first_t_side'] == row['pistol_first']:
                        pistol_t_wins += 1
                    elif row['first_ct_side'] == row['pistol_first']:
                        pistol_ct_wins += 1
                if pd.notna(row['pistol_second']) and row['pistol_second'] != '/':
                    pistol_total += 1
                    if row['second_t_side'] == row['pistol_second']:
                        pistol_t_wins += 1
                    elif row['second_ct_side'] == row['pistol_second']:
                        pistol_ct_wins += 1
            pistol_t_rate = pistol_t_wins / pistol_total if pistol_total > 0 else 0
            pistol_ct_rate = pistol_ct_wins / pistol_total if pistol_total > 0 else 0

            team_wins = {}
            for _, row in group.iterrows():
                winner = row['win']
                if pd.notna(winner):
                    team_wins[winner] = team_wins.get(winner, 0) + 1
            top_teams = []
            for team, wins in sorted(team_wins.items(), key=lambda x: x[1], reverse=True):
                if len(group[group['win'] == team]) > 3:
                    top_teams.append(team)
                    if len(top_teams) >= 3:
                        break

            intense = {'胶着':0, '激烈':0, '一般':0, '碾压':0}
            for _, row in group.iterrows():
                tr = row['total_rounds']
                if tr > 22.5:
                    intense['胶着'] += 1
                elif tr > 21.5:
                    intense['激烈'] += 1
                elif tr > 19.5:
                    intense['一般'] += 1
                else:
                    intense['碾压'] += 1
            total_g = len(group)

            map_stats.append({
                'map': map_name,
                'T胜率(加权)': t_rate,
                'CT胜率(加权)': ct_rate,
                '平局率(加权)': draw_rate,
                '上半场最大分差': first_diffs.max(),
                '上半场最小分差': first_diffs.min(),
                '上半场平均分差': first_diffs.mean(),
                '下半场最大分差': second_diffs.max(),
                '下半场最小分差': second_diffs.min(),
                '下半场平均分差': second_diffs.mean(),
                '全场最大分差': full_diffs.max(),
                '全场最小分差': full_diffs.min(),
                '全场平均分差': full_diffs.mean(),
                '手枪局T胜率': pistol_t_rate,
                '手枪局CT胜率': pistol_ct_rate,
                '胜率前三战队': ', '.join(top_teams) if top_teams else '无',
                '胶着率': intense['胶着']/total_g,
                '激烈率': intense['激烈']/total_g,
                '一般率': intense['一般']/total_g,
                '碾压率': intense['碾压']/total_g,
                '翻盘率': group['is_comeback'].sum() / total_g,
                '样本数': total_g,
                '上半场T胜率': t_wins_f / total_f,
                '上半场CT胜率': ct_wins_f / total_f,
                '上半场平局率': draws_f / total_f
            })

        map_stats_df = pd.DataFrame(map_stats)
        if map_stats_df.empty:
            st.warning("数据量不足，需要每张地图至少3场数据")
        else:
            map_stats_df.index = range(1, len(map_stats_df)+1)
            st.dataframe(map_stats_df, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df['T胜率(加权)'], name='T胜率',
                                     marker_color='salmon',
                                     text=[f'{v:.1%}' for v in map_stats_df['T胜率(加权)']],
                                     textposition='outside'))
                fig.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df['CT胜率(加权)'], name='CT胜率',
                                     marker_color='lightskyblue',
                                     text=[f'{v:.1%}' for v in map_stats_df['CT胜率(加权)']],
                                     textposition='outside'))
                fig.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df['平局率(加权)'], name='平局率',
                                     marker_color='gold',
                                     text=[f'{v:.1%}' for v in map_stats_df['平局率(加权)']],
                                     textposition='outside'))
                fig.update_layout(yaxis_tickformat=".0%", barmode='group', height=400, title="T/CT/平局胜率（加权）")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("注：T胜率+CT胜率+平局率=1，平局指半场得分相同。")

                fig_upper = go.Figure()
                fig_upper.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df['上半场T胜率'], name='T胜率',
                                           marker_color='salmon',
                                           text=[f'{v:.1%}' for v in map_stats_df['上半场T胜率']],
                                           textposition='outside'))
                fig_upper.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df['上半场CT胜率'], name='CT胜率',
                                           marker_color='lightskyblue',
                                           text=[f'{v:.1%}' for v in map_stats_df['上半场CT胜率']],
                                           textposition='outside'))
                fig_upper.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df['上半场平局率'], name='平局率',
                                           marker_color='gold',
                                           text=[f'{v:.1%}' for v in map_stats_df['上半场平局率']],
                                           textposition='outside'))
                fig_upper.update_layout(yaxis_tickformat=".0%", barmode='group', height=400,
                                        title="上半场 T/CT/平局胜率")
                st.plotly_chart(fig_upper, use_container_width=True)

            with col2:
                color_map = {'碾压': 'lightgreen', '一般': 'gold', '激烈': 'salmon', '胶着': 'red'}
                categories = ['胶着', '激烈', '一般', '碾压']
                fig2 = go.Figure()
                for cat in categories:
                    fig2.add_trace(go.Bar(x=map_stats_df['map'], y=map_stats_df[f'{cat}率'], name=cat,
                                          marker_color=color_map[cat],
                                          text=[f'{v:.1%}' for v in map_stats_df[f'{cat}率']],
                                          textposition='auto'))
                fig2.update_layout(barmode='stack', yaxis_tickformat=".0%", height=400,
                                   title="激烈程度分布（胶着>22.5, 激烈>21.5, 一般>19.5, 碾压<19.5）")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("各地图翻盘率")
            fig3 = px.bar(map_stats_df, x='map', y='翻盘率', text=[f"{v:.1%}" for v in map_stats_df['翻盘率']])
            fig3.update_traces(textposition='outside')
            fig3.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("地图最强战队 Top 3")
            st.caption("排名规则：胜率 > 胜场 > 全场最大分差，且该地图出场次数≥3")
            team_perf_all = []
            for map_name, group in df.groupby('map'):
                if len(group) < 3:
                    continue
                team_stats = {}
                for _, row in group.iterrows():
                    t1, t2 = row['team1'], row['team2']
                    diff = abs(row['team1_score'] - row['team2_score'])
                    if t1 not in team_stats:
                        team_stats[t1] = {'total': 0, 'wins': 0, 'max_diff': 0, 'min_diff': 999}
                    team_stats[t1]['total'] += 1
                    if row['win'] == t1:
                        team_stats[t1]['wins'] += 1
                    team_stats[t1]['max_diff'] = max(team_stats[t1]['max_diff'], diff)
                    team_stats[t1]['min_diff'] = min(team_stats[t1]['min_diff'], diff)
                    if t2 not in team_stats:
                        team_stats[t2] = {'total': 0, 'wins': 0, 'max_diff': 0, 'min_diff': 999}
                    team_stats[t2]['total'] += 1
                    if row['win'] == t2:
                        team_stats[t2]['wins'] += 1
                    team_stats[t2]['max_diff'] = max(team_stats[t2]['max_diff'], diff)
                    team_stats[t2]['min_diff'] = min(team_stats[t2]['min_diff'], diff)
                perf_list = []
                for team, stat in team_stats.items():
                    if stat['total'] < 3:
                        continue
                    win_rate = stat['wins'] / stat['total'] if stat['total'] > 0 else 0
                    perf_list.append({
                        'team': team,
                        '胜率': win_rate,
                        '胜场': stat['wins'],
                        '总场次': stat['total'],
                        '最大分差': stat['max_diff'],
                        '最小分差': stat['min_diff']
                    })
                if perf_list:
                    perf_list.sort(key=lambda x: (x['胜率'], x['胜场'], x['最大分差']), reverse=True)
                    for rank, p in enumerate(perf_list[:3], start=1):
                        team_perf_all.append({
                            '地图': map_name,
                            '排名': rank,
                            '战队': p['team'],
                            '胜率_numeric': p['胜率'],
                            '胜率_display': f"{p['胜率']:.1%}",
                            '胜场': p['胜场'],
                            '总场次': p['总场次'],
                            '分差范围': f"{int(p['最小分差'])}-{int(p['最大分差'])}"
                        })
            if team_perf_all:
                top_df = pd.DataFrame(team_perf_all)
                st.dataframe(top_df[['地图', '排名', '战队', '胜率_display', '胜场', '总场次', '分差范围']],
                             use_container_width=True)
                fig_top = px.bar(top_df, x='战队', y='胜率_numeric', facet_col='地图', color='战队',
                                 text='胜率_display',
                                 hover_data={'胜场': True, '分差范围': True, '胜率_numeric': False},
                                 category_orders={'排名': [1,2,3]},
                                 title="各图最强战队 Top3 胜率对比")
                fig_top.update_traces(textposition='outside')
                fig_top.update_layout(yaxis_tickformat=".0%", showlegend=False, height=600)
                st.plotly_chart(fig_top, use_container_width=True)
            else:
                st.warning("没有足够数据计算Top3")

    # ===== 战队Pick Map分析 =====
    with tabs[1]:
        pick_df = df[df['pick_team'] != '/'].copy()
        pick_df = pick_df.dropna(subset=['pick_team'])
        if pick_df.empty:
            st.warning("无有效自选图记录。")
        else:
            results = []
            for map_name, group in pick_df.groupby('map'):
                total = len(group)
                wins = group[group['win'] == group['pick_team']].shape[0]
                half_wins, half_draws, half_losses = 0, 0, 0
                for _, row in group.iterrows():
                    if row['first_t_side'] == row['pick_team']:
                        if row['first_t_score'] > row['first_ct_score']:
                            half_wins += 1
                        elif row['first_t_score'] == row['first_ct_score']:
                            half_draws += 1
                        else:
                            half_losses += 1
                    elif row['first_ct_side'] == row['pick_team']:
                        if row['first_ct_score'] > row['first_t_score']:
                            half_wins += 1
                        elif row['first_ct_score'] == row['first_t_score']:
                            half_draws += 1
                        else:
                            half_losses += 1
                map_rate = wins / total if total > 0 else 0
                half_win_rate = half_wins / total if total > 0 else 0
                half_draw_rate = half_draws / total if total > 0 else 0
                half_loss_rate = half_losses / total if total > 0 else 0

                results.append({
                    'map': map_name,
                    'pick次数': total,
                    'pick获胜次数': wins,
                    '地图胜率': map_rate,
                    '上半场胜率': half_win_rate,
                    '上半场平率': half_draw_rate,
                    '上半场负率': half_loss_rate,
                    '上半场胜场': half_wins,
                    '上半场平场': half_draws,
                    '上半场负场': half_losses,
                })
            res_df = pd.DataFrame(results)
            res_df = res_df.sort_values('pick次数', ascending=False)
            res_df.index = range(1, len(res_df)+1)

            st.subheader("🎯 全场胜率（仅胜负）")
            col1_full, col2_full = st.columns([2, 3])
            with col1_full:
                st.dataframe(res_df[['map', 'pick次数', 'pick获胜次数', '地图胜率']], use_container_width=True)
            with col2_full:
                fig_full = px.bar(res_df, x='map', y='地图胜率',
                                  text=[f"{v:.1%}" for v in res_df['地图胜率']])
                fig_full.update_traces(textposition='outside')
                fig_full.update_layout(yaxis_tickformat=".0%", height=400, title="自选图全场胜率")
                st.plotly_chart(fig_full, use_container_width=True)

            st.subheader("⏳ 上半场胜率（含平局）")
            col1_half, col2_half = st.columns([2, 3])
            with col1_half:
                st.dataframe(res_df[['map', 'pick次数', '上半场胜率', '上半场平率', '上半场负率']],
                             use_container_width=True)
            with col2_half:
                fig_half = go.Figure()
                for cat, col in [('胜', '上半场胜率'), ('平', '上半场平率'), ('负', '上半场负率')]:
                    fig_half.add_trace(go.Bar(x=res_df['map'], y=res_df[col], name=cat,
                                             text=[f"{v:.1%}" for v in res_df[col]],
                                             textposition='outside'))
                fig_half.update_layout(barmode='group', yaxis_tickformat=".0%", height=400,
                                       title="自选图上半场胜/平/负率")
                st.plotly_chart(fig_half, use_container_width=True)

            st.subheader("自选图翻盘统计")
            comeback = pick_df[pick_df['is_comeback'] == True]
            st.write(f"翻盘场次: {len(comeback)} / {len(pick_df)} ({len(comeback)/len(pick_df)*100:.1f}%)")
            if not comeback.empty:
                display_rows = []
                for _, row in comeback.iterrows():
                    pick = row['pick_team']
                    opp = row['team1'] if row['team2'] == pick else row['team2']
                    half_leader = row['first_half_leader']
                    diff = int(row['first_half_diff'])
                    if row['first_t_side'] == pick:
                        side_pick = 'T'
                        score_pick_half = int(row['first_t_score'])
                        score_opp_half = int(row['first_ct_score'])
                    else:
                        side_pick = 'CT'
                        score_pick_half = int(row['first_ct_score'])
                        score_opp_half = int(row['first_t_score'])
                    half_str = f"{pick}({side_pick}) {score_pick_half} - {score_opp_half} {opp}"
                    final_t1 = int(row['team1_score'])
                    final_t2 = int(row['team2_score'])
                    final_str = f"{row['team1']} {final_t1} - {final_t2} {row['team2']}"
                    display_rows.append({
                        '地图': row['map'],
                        '自选队伍': pick,
                        '对手': opp,
                        '上半场比分': half_str,
                        '上半场领先方': half_leader,
                        '落后分数': diff,
                        '最终比分': final_str,
                        '翻盘胜者': row['win']
                    })
                df_display = pd.DataFrame(display_rows)
                df_display.index = range(1, len(df_display) + 1)
                st.dataframe(df_display, use_container_width=True)

# ---------- 界面2（已按需求修改）----------
def render_team_view(df):
    st.header("👤 单战队数据查看")
    all_teams = sorted(set(df['team1'].unique()).union(set(df['team2'].unique())))
    all_teams = [t for t in all_teams if pd.notna(t) and t != '/']

    team_options = [""] + all_teams
    selected = st.selectbox("选择队伍", team_options, index=0)
    if selected:
        st.session_state.selected_team = selected
    else:
        st.session_state.selected_team = None
        st.warning("请选择一支队伍")
        return

    team = st.session_state.selected_team
    team_df = df[(df['team1'] == team) | (df['team2'] == team)].copy()
    if team_df.empty:
        st.warning(f"队伍 {team} 无有效数据")
        return

    st.subheader(f"{team} 数据总览")
    st.write(f"有效比赛场次: {len(team_df)}")

    map_perf = []
    for map_name, group in team_df.groupby('map'):
        total = len(group)
        wins = group[group['win'] == team].shape[0]

        # 上半场胜/平/负
        half_wins, half_draws, half_losses = 0, 0, 0
        for _, row in group.iterrows():
            if row['first_t_side'] == team:
                if row['first_t_score'] > row['first_ct_score']:
                    half_wins += 1
                elif row['first_t_score'] == row['first_ct_score']:
                    half_draws += 1
                else:
                    half_losses += 1
            elif row['first_ct_side'] == team:
                if row['first_ct_score'] > row['first_t_score']:
                    half_wins += 1
                elif row['first_ct_score'] == row['first_t_score']:
                    half_draws += 1
                else:
                    half_losses += 1

        half_rate = half_wins / total if total > 0 else 0
        half_draw_rate = half_draws / total if total > 0 else 0
        half_loss_rate = half_losses / total if total > 0 else 0

        # 手枪局统计
        pistol_wins_total, pistol_total_games = 0, 0
        pistol_t_wins, pistol_t_total = 0, 0
        pistol_ct_wins, pistol_ct_total = 0, 0
        for _, row in group.iterrows():
            if pd.notna(row['pistol_first']) and row['pistol_first'] != '/':
                pistol_total_games += 1
                if row['pistol_first'] == team:
                    pistol_wins_total += 1
                if row['first_t_side'] == team:
                    pistol_t_total += 1
                    if row['pistol_first'] == team:
                        pistol_t_wins += 1
                elif row['first_ct_side'] == team:
                    pistol_ct_total += 1
                    if row['pistol_first'] == team:
                        pistol_ct_wins += 1
            if pd.notna(row['pistol_second']) and row['pistol_second'] != '/':
                pistol_total_games += 1
                if row['pistol_second'] == team:
                    pistol_wins_total += 1
                if row['second_t_side'] == team:
                    pistol_t_total += 1
                    if row['pistol_second'] == team:
                        pistol_t_wins += 1
                elif row['second_ct_side'] == team:
                    pistol_ct_total += 1
                    if row['pistol_second'] == team:
                        pistol_ct_wins += 1

        pistol_rate = pistol_wins_total / pistol_total_games if pistol_total_games > 0 else 0
        pistol_t_rate = pistol_t_wins / pistol_t_total if pistol_t_total > 0 else 0
        pistol_ct_rate = pistol_ct_wins / pistol_ct_total if pistol_ct_total > 0 else 0

        # 先3/6/9（含场数）
        first3_total = total
        first3_wins = group[group['first_to_3'] == team].shape[0]
        first6_wins = group[group['first_to_6'] == team].shape[0]
        first9_wins = group[group['first_to_9'] == team].shape[0]
        first3_rate = first3_wins / total if total > 0 else 0
        first6_rate = first6_wins / total if total > 0 else 0
        first9_rate = first9_wins / total if total > 0 else 0

        # 激烈程度
        intense = {'胶着': 0, '激烈': 0, '一般': 0, '碾压': 0}
        for _, row in group.iterrows():
            tr = row['total_rounds']
            if tr > 22.5:
                intense['胶着'] += 1
            elif tr > 21.5:
                intense['激烈'] += 1
            elif tr > 19.5:
                intense['一般'] += 1
            else:
                intense['碾压'] += 1

        ot_rate = group['is_ot'].sum() / total if total > 0 else 0
        comeback_rate = group['is_comeback'].sum() / total if total > 0 else 0

        map_perf.append({
            'map': map_name,
            '总场次': total,
            '胜率': wins / total if total > 0 else 0,
            '胜场': wins,
            # 上半场
            '上半场胜率': half_rate,
            '上半场平率': half_draw_rate,
            '上半场负率': half_loss_rate,
            '半场胜场': half_wins,
            '半场平场': half_draws,
            '半场负场': half_losses,
            # 手枪局
            '手枪总胜率': pistol_rate,
            '手枪T胜率': pistol_t_rate,
            '手枪CT胜率': pistol_ct_rate,
            '手枪总局数': pistol_total_games,
            '手枪总胜场': pistol_wins_total,
            '手枪T总场': pistol_t_total,
            '手枪T胜场': pistol_t_wins,
            '手枪CT总场': pistol_ct_total,
            '手枪CT胜场': pistol_ct_wins,
            # 先3/6/9
            '先3胜率': first3_rate,
            '先3胜场': first3_wins,
            '先3总场': first3_total,
            '先6胜率': first6_rate,
            '先6胜场': first6_wins,
            '先6总场': first3_total,
            '先9胜率': first9_rate,
            '先9胜场': first9_wins,
            '先9总场': first3_total,
            # 激烈等
            '胶着率': intense['胶着'] / total,
            '胶着场': intense['胶着'],
            '激烈率': intense['激烈'] / total,
            '激烈场': intense['激烈'],
            '一般率': intense['一般'] / total,
            '一般场': intense['一般'],
            '碾压率': intense['碾压'] / total,
            '碾压场': intense['碾压'],
            'OT率': ot_rate,
            'OT场': group['is_ot'].sum(),
            '翻盘率': comeback_rate,
            '翻盘场': group['is_comeback'].sum()
        })

    perf_df = pd.DataFrame(map_perf)
    if perf_df.empty:
        st.warning(f"{team} 无地图数据")
        return

    perf_df.index = range(1, len(perf_df) + 1)
    tabs = st.tabs(["📈 地图胜率", "⏳ 半场胜率", "🔫 手枪局", "🎯 先3/6/9", "⚡ 激烈/OT/翻盘"])

    # Tab 0: 地图胜率（未动）
    with tabs[0]:
        st.dataframe(perf_df[['map', '总场次', '胜率', '胜场']], use_container_width=True)
        fig = px.bar(perf_df, x='map', y='胜率',
                     text=[fmt_prob(r['胜场'], r['总场次']) for _, r in perf_df.iterrows()])
        fig.update_traces(textposition='outside')
        fig.update_layout(yaxis_tickformat=".0%", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 1: 半场胜率（需求1：增加平和负）
    with tabs[1]:
        # 表格增加平和负
        st.dataframe(perf_df[['map', '总场次', '上半场胜率', '上半场平率', '上半场负率',
                              '半场胜场', '半场平场', '半场负场']], use_container_width=True)
        # 图表：胜/平/负分组柱状图
        fig_half = go.Figure()
        fig_half.add_trace(go.Bar(x=perf_df['map'], y=perf_df['上半场胜率'], name='胜',
                                  text=[f"{v:.1%}" for v in perf_df['上半场胜率']],
                                  textposition='outside'))
        fig_half.add_trace(go.Bar(x=perf_df['map'], y=perf_df['上半场平率'], name='平',
                                  text=[f"{v:.1%}" for v in perf_df['上半场平率']],
                                  textposition='outside'))
        fig_half.add_trace(go.Bar(x=perf_df['map'], y=perf_df['上半场负率'], name='负',
                                  text=[f"{v:.1%}" for v in perf_df['上半场负率']],
                                  textposition='outside'))
        fig_half.update_layout(barmode='group', yaxis_tickformat=".0%", height=400,
                               title="上半场胜/平/负率")
        st.plotly_chart(fig_half, use_container_width=True)

    # Tab 2: 手枪局（需求2 & 3）
    with tabs[2]:
        # 表格增加场数和胜场
        st.dataframe(perf_df[['map', '手枪总胜率', '手枪总局数', '手枪总胜场',
                              '手枪T胜率', '手枪T总场', '手枪T胜场',
                              '手枪CT胜率', '手枪CT总场', '手枪CT胜场']], use_container_width=True)

        # 柱形图增加百分比，颜色调整
        fig_pistol = go.Figure()
        fig_pistol.add_trace(go.Bar(
            x=perf_df['map'],
            y=perf_df['手枪总胜率'],
            name='总胜率',
            marker_color='plum',  # 浅紫色
            text=[f"{v:.1%}" for v in perf_df['手枪总胜率']],
            textposition='outside'
        ))
        fig_pistol.add_trace(go.Bar(
            x=perf_df['map'],
            y=perf_df['手枪T胜率'],
            name='T胜率',
            marker_color='salmon',
            text=[f"{v:.1%}" for v in perf_df['手枪T胜率']],
            textposition='outside'
        ))
        fig_pistol.add_trace(go.Bar(
            x=perf_df['map'],
            y=perf_df['手枪CT胜率'],
            name='CT胜率',
            marker_color='lightskyblue',
            text=[f"{v:.1%}" for v in perf_df['手枪CT胜率']],
            textposition='outside'
        ))
        fig_pistol.update_layout(yaxis_tickformat=".0%", barmode='group', height=400,
                                title="手枪局胜率")
        st.plotly_chart(fig_pistol, use_container_width=True)

    # Tab 3: 先3/6/9（需求4）
    with tabs[3]:
        # 表格增加场数
        st.dataframe(perf_df[['map', '先3胜率', '先3胜场', '先3总场',
                              '先6胜率', '先6胜场', '先6总场',
                              '先9胜率', '先9胜场', '先9总场']], use_container_width=True)

        # 柱形图增加百分比
        fig_first = go.Figure()
        for metric, color in [('先3胜率', 'blue'), ('先6胜率', 'orange'), ('先9胜率', 'green')]:
            fig_first.add_trace(go.Bar(
                x=perf_df['map'],
                y=perf_df[metric],
                name=metric.replace('胜率', ''),
                marker_color=color,
                text=[f"{v:.1%}" for v in perf_df[metric]],
                textposition='outside'
            ))
        fig_first.update_layout(yaxis_tickformat=".0%", barmode='group', height=400,
                                title="先3/6/9胜率")
        st.plotly_chart(fig_first, use_container_width=True)

    # Tab 4: 激烈/OT/翻盘（需求5）
    with tabs[4]:
        # 表格增加场数和总场次（总场次就是总场次，可加一列）
        st.dataframe(perf_df[['map', '总场次',
                              '胶着率', '胶着场',
                              '激烈率', '激烈场',
                              '一般率', '一般场',
                              '碾压率', '碾压场',
                              'OT率', 'OT场',
                              '翻盘率', '翻盘场']], use_container_width=True)

        # 柱形图增加百分比，并使用指定颜色
        categories = ['碾压', '一般', '激烈', '胶着']
        colors = {'碾压': 'lightgreen', '一般': 'gold', '激烈': 'salmon', '胶着': 'red'}
        fig_intense = go.Figure()
        for cat in categories:
            fig_intense.add_trace(go.Bar(
                x=perf_df['map'],
                y=perf_df[f'{cat}率'],
                name=cat,
                marker_color=colors[cat],
                text=[f"{v:.1%}" for v in perf_df[f'{cat}率']],
                textposition='auto'
            ))
        fig_intense.update_layout(barmode='stack', yaxis_tickformat=".0%", height=400,
                                  title="激烈程度分布")
        st.plotly_chart(fig_intense, use_container_width=True)

        # 新增柱形图：加时率和翻盘率
        fig_ot_cb = go.Figure()
        fig_ot_cb.add_trace(go.Bar(
            x=perf_df['map'],
            y=perf_df['OT率'],
            name='加时率',
            marker_color='gray',
            text=[f"{v:.1%}" for v in perf_df['OT率']],
            textposition='outside'
        ))
        fig_ot_cb.add_trace(go.Bar(
            x=perf_df['map'],
            y=perf_df['翻盘率'],
            name='翻盘率',
            marker_color='darkred',
            text=[f"{v:.1%}" for v in perf_df['翻盘率']],
            textposition='outside'
        ))
        fig_ot_cb.update_layout(barmode='group', yaxis_tickformat=".0%", height=400,
                                title="加时与翻盘率")
        st.plotly_chart(fig_ot_cb, use_container_width=True)

# ---------- 界面3 ----------
def render_compare_predict(df):
    st.header("⚔️ 战队对阵对比与预测")
    all_teams = sorted(set(df['team1'].unique()).union(set(df['team2'].unique())))
    all_teams = [t for t in all_teams if pd.notna(t) and t != '/']

    col1, col2 = st.columns(2)
    with col1:
        team1_opts = [""] + all_teams
        team1 = st.selectbox("队伍1", team1_opts, index=0)
        if team1:
            st.session_state.selected_team1 = team1
        else:
            st.session_state.selected_team1 = None
    with col2:
        team2_opts = [""] + all_teams
        team2 = st.selectbox("队伍2", team2_opts, index=0)
        if team2:
            st.session_state.selected_team2 = team2
        else:
            st.session_state.selected_team2 = None

    team1 = st.session_state.selected_team1
    team2 = st.session_state.selected_team2
    if not team1 or not team2:
        st.warning("请选择两支队伍")
        return
    if team1 == team2:
        st.warning("请选择两支不同队伍")
        return

    team1_maps = set(df[df['team1'] == team1]['map']).union(set(df[df['team2'] == team1]['map']))
    team2_maps = set(df[df['team1'] == team2]['map']).union(set(df[df['team2'] == team2]['map']))
    common_maps = [m for m in team1_maps.intersection(team2_maps) if pd.notna(m) and m != '/']
    if not common_maps:
        st.warning(f"{team1} 和 {team2} 没有共同地图数据")
        return

    st.info(f"共同地图: {', '.join(common_maps)}")
    map_choice = st.selectbox("选择地图查看（或全览）", ["全部地图"] + common_maps, key="map_choice_compare")

    tabs = st.tabs(["📊 数据对比", "🔮 对局预测"])

    # ============================================================
    # 子界面1：数据对比
    # ============================================================
    with tabs[0]:
        st.subheader("各项指标对比")

        compare_data = []
        for map_name in common_maps:
            def get_metrics(team, map_name):
                mp = df[(df['map'] == map_name) & ((df['team1'] == team) | (df['team2'] == team))]
                if mp.empty:
                    return None
                total = len(mp)
                wins = mp[mp['win'] == team].shape[0]
                win_rate = wins / total if total > 0 else 0
                half_wins = 0
                for _, row in mp.iterrows():
                    if row['first_t_side'] == team and row['first_t_score'] > row['first_ct_score']:
                        half_wins += 1
                    elif row['first_ct_side'] == team and row['first_ct_score'] > row['first_t_score']:
                        half_wins += 1
                half_rate = half_wins / total if total > 0 else 0
                p_wins, p_total = 0, 0
                for _, row in mp.iterrows():
                    if pd.notna(row['pistol_first']) and row['pistol_first'] != '/':
                        p_total += 1
                        if row['pistol_first'] == team:
                            p_wins += 1
                    if pd.notna(row['pistol_second']) and row['pistol_second'] != '/':
                        p_total += 1
                        if row['pistol_second'] == team:
                            p_wins += 1
                pistol_rate = p_wins / p_total if p_total > 0 else 0
                first3 = mp[mp['first_to_3'] == team].shape[0] / total if total > 0 else 0
                first6 = mp[mp['first_to_6'] == team].shape[0] / total if total > 0 else 0
                first9 = mp[mp['first_to_9'] == team].shape[0] / total if total > 0 else 0
                return {
                    '胜率': win_rate,
                    '上半场胜率': half_rate,
                    '手枪胜率': pistol_rate,
                    '先3': first3,
                    '先6': first6,
                    '先9': first9
                }
            m1 = get_metrics(team1, map_name)
            m2 = get_metrics(team2, map_name)
            if m1 is None or m2 is None:
                continue
            compare_data.append({
                'map': map_name,
                f'{team1}_胜率': m1['胜率'],
                f'{team2}_胜率': m2['胜率'],
                f'{team1}_上半场胜率': m1['上半场胜率'],
                f'{team2}_上半场胜率': m2['上半场胜率'],
                f'{team1}_手枪胜率': m1['手枪胜率'],
                f'{team2}_手枪胜率': m2['手枪胜率'],
                f'{team1}_先3': m1['先3'],
                f'{team2}_先3': m2['先3'],
                f'{team1}_先6': m1['先6'],
                f'{team2}_先6': m2['先6'],
                f'{team1}_先9': m1['先9'],
                f'{team2}_先9': m2['先9'],
            })

        if not compare_data:
            st.warning("无法获取对比数据")
            return

        comp_df = pd.DataFrame(compare_data)

        if map_choice != "全部地图":
            comp_df_filtered = comp_df[comp_df['map'] == map_choice]
        else:
            comp_df_filtered = comp_df

        if comp_df_filtered.empty:
            st.warning(f"没有 {map_choice} 的数据")
            return

        comp_df_filtered_display = comp_df_filtered.reset_index(drop=True)
        comp_df_filtered_display.index = range(1, len(comp_df_filtered_display) + 1)
        st.dataframe(comp_df_filtered_display, use_container_width=True)

        metrics_to_plot = ['胜率', '上半场胜率', '手枪胜率', '先3', '先6', '先9']
        for metric in metrics_to_plot:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=comp_df_filtered['map'],
                y=comp_df_filtered[f'{team1}_{metric}'],
                name=team1,
                text=[f"{v:.1%}" for v in comp_df_filtered[f'{team1}_{metric}']],
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                x=comp_df_filtered['map'],
                y=comp_df_filtered[f'{team2}_{metric}'],
                name=team2,
                text=[f"{v:.1%}" for v in comp_df_filtered[f'{team2}_{metric}']],
                textposition='outside'
            ))
            fig.update_layout(
                yaxis_tickformat=".0%",
                barmode='group',
                title=f"{metric}对比" + (f" ({map_choice})" if map_choice != "全部地图" else ""),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("雷达图对比")
        maps_to_show = [map_choice] if map_choice != "全部地图" else common_maps
        for map_name in maps_to_show:
            row = comp_df[comp_df['map'] == map_name].iloc[0]
            fig = go.Figure()
            categories = ['胜率', '上半场胜率', '手枪胜率', '先3', '先6', '先9']
            r1 = [row[f'{team1}_{c}'] for c in categories]
            r2 = [row[f'{team2}_{c}'] for c in categories]
            fig.add_trace(go.Scatterpolar(r=r1, theta=categories, fill='toself', name=team1))
            fig.add_trace(go.Scatterpolar(r=r2, theta=categories, fill='toself', name=team2))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"{map_name}",
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 子界面2：对局预测
    # ============================================================
    with tabs[1]:
        st.subheader(f"{team1} vs {team2} 逐地图预测")
        st.caption("预测模型: 全局胜率40% + 近5场状态30% + H2H交手记录30% (上限5场)")

        st.info(f"当前查看地图: {map_choice}")

        t_side = st.radio("本场谁先担任进攻方(T)?", [team1, team2], horizontal=True)

        maps_to_predict = [map_choice] if map_choice != "全部地图" else common_maps

        for map_name in maps_to_predict:
            st.markdown(f"### 🗺️ {map_name}")

            def get_global_rate(team, map_name):
                mp = df[(df['map'] == map_name) & ((df['team1'] == team) | (df['team2'] == team))]
                if mp.empty:
                    return 0.5
                return mp[mp['win'] == team].shape[0] / len(mp)

            def get_recent_rate(team, map_name):
                mp = df[(df['map'] == map_name) & ((df['team1'] == team) | (df['team2'] == team))]
                if mp.empty:
                    return 0.5
                if 'date' in mp.columns:
                    mp = mp.sort_values('date', ascending=False)
                recent = mp.head(5)
                return recent[recent['win'] == team].shape[0] / len(recent) if len(recent) > 0 else 0.5

            def get_h2h_rate(team, opp, map_name):
                mp = df[(df['map'] == map_name) &
                        (((df['team1'] == team) & (df['team2'] == opp)) |
                         ((df['team1'] == opp) & (df['team2'] == team)))]
                if mp.empty:
                    return 0.5
                total = min(len(mp), 5)
                mp = mp.head(total)
                wins = mp[mp['win'] == team].shape[0]
                return wins / total if total > 0 else 0.5

            g1 = get_global_rate(team1, map_name)
            g2 = get_global_rate(team2, map_name)
            r1 = get_recent_rate(team1, map_name)
            r2 = get_recent_rate(team2, map_name)
            h1 = get_h2h_rate(team1, team2, map_name)
            h2 = get_h2h_rate(team2, team1, map_name)

            p1 = 0.4 * g1 + 0.3 * r1 + 0.3 * h1
            p2 = 0.4 * g2 + 0.3 * r2 + 0.3 * h2
            total_p = p1 + p2
            if total_p == 0:
                p1, p2 = 0.5, 0.5
            else:
                p1, p2 = p1 / total_p, p2 / total_p

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{team1} 胜率", f"{p1 * 100:.1f}%")
            col2.metric(f"{team2} 胜率", f"{p2 * 100:.1f}%")
            col3.metric("预测胜者", team1 if p1 > p2 else team2)
            st.progress(p1, text=f"{team1} 胜率 {p1 * 100:.1f}%")

            def get_half_rate(team, map_name, side):
                mp = df[(df['map'] == map_name) & ((df['team1'] == team) | (df['team2'] == team))]
                if mp.empty:
                    return 0.33, 0.33, 0.34
                wins, draws, losses = 0, 0, 0
                for _, row in mp.iterrows():
                    if side == 'T':
                        if row['first_t_side'] == team:
                            if row['first_t_score'] > row['first_ct_score']:
                                wins += 1
                            elif row['first_t_score'] == row['first_ct_score']:
                                draws += 1
                            else:
                                losses += 1
                    else:
                        if row['first_ct_side'] == team:
                            if row['first_ct_score'] > row['first_t_score']:
                                wins += 1
                            elif row['first_ct_score'] == row['first_t_score']:
                                draws += 1
                            else:
                                losses += 1
                total = wins + draws + losses
                if total == 0:
                    return 0.33, 0.33, 0.34
                return wins / total, draws / total, losses / total

            if t_side == team1:
                side1, side2 = 'T', 'CT'
            else:
                side1, side2 = 'CT', 'T'

            w1, d1, l1 = get_half_rate(team1, map_name, side1)
            w2, d2, l2 = get_half_rate(team2, map_name, side2)

            st.write("**上半场预测 (胜/平/负)**")
            c1, c2 = st.columns(2)
            c1.write(f"{team1} ({side1}开局): 胜{w1:.1%}, 平{d1:.1%}, 负{l1:.1%}")
            c2.write(f"{team2} ({side2}开局): 胜{w2:.1%}, 平{d2:.1%}, 负{l2:.1%}")

            def get_avg_score(team, map_name, side):
                mp = df[(df['map'] == map_name) & ((df['team1'] == team) | (df['team2'] == team))]
                scores = []
                for _, row in mp.iterrows():
                    if side == 'T':
                        if row['first_t_side'] == team:
                            scores.append(row['first_t_score'])
                        if row['second_t_side'] == team:
                            scores.append(row['second_t_score'])
                    else:
                        if row['first_ct_side'] == team:
                            scores.append(row['first_ct_score'])
                        if row['second_ct_side'] == team:
                            scores.append(row['second_ct_score'])
                if not scores:
                    return 6.0
                return np.mean(scores)

            avg1 = get_avg_score(team1, map_name, side1)
            avg2 = get_avg_score(team2, map_name, side2)
            st.write(f"**预测半场比分**: {team1} {avg1:.1f} - {avg2:.1f} {team2}")

            def get_first_to(team, map_name, target):
                mp = df[(df['map'] == map_name) & ((df['team1'] == team) | (df['team2'] == team))]
                if mp.empty:
                    return 0.5
                col = f'first_to_{target}'
                if col not in mp.columns:
                    return 0.5
                return mp[mp[col] == team].shape[0] / len(mp) if len(mp) > 0 else 0.5

            f3_1 = get_first_to(team1, map_name, 3)
            f6_1 = get_first_to(team1, map_name, 6)
            f9_1 = get_first_to(team1, map_name, 9)
            f3_2 = get_first_to(team2, map_name, 3)
            f6_2 = get_first_to(team2, map_name, 6)
            f9_2 = get_first_to(team2, map_name, 9)

            st.write("**先3/6/9局概率**")
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{team1} 先3", f"{f3_1:.1%}")
            col2.metric(f"{team1} 先6", f"{f6_1:.1%}")
            col3.metric(f"{team1} 先9", f"{f9_1:.1%}")
            col1.metric(f"{team2} 先3", f"{f3_2:.1%}")
            col2.metric(f"{team2} 先6", f"{f6_2:.1%}")
            col3.metric(f"{team2} 先9", f"{f9_2:.1%}")

            def get_avg_total(t1, t2, map_name):
                mp = df[(df['map'] == map_name) &
                        (((df['team1'] == t1) & (df['team2'] == t2)) |
                         ((df['team1'] == t2) & (df['team2'] == t1)))]
                if mp.empty:
                    return 20.0
                return mp['total_rounds'].mean()

            avg_total = get_avg_total(team1, team2, map_name)
            if avg_total > 22.5:
                intense = "胶着"
            elif avg_total > 21.5:
                intense = "激烈"
            elif avg_total > 19.5:
                intense = "一般"
            else:
                intense = "碾压"

            score1 = avg_total * p1
            score2 = avg_total * p2
            st.write(f"**预测全场**: {intense} (总回合 {avg_total:.1f}), 比分 {team1} {score1:.1f} - {score2:.1f} {team2}")

            def get_ot_rate(t1, t2, map_name):
                mp = df[(df['map'] == map_name) &
                        (((df['team1'] == t1) & (df['team2'] == t2)) |
                         ((df['team1'] == t2) & (df['team2'] == t1)))]
                if mp.empty:
                    return 0.1
                return mp['is_ot'].sum() / len(mp) if len(mp) > 0 else 0.1

            ot_rate = get_ot_rate(team1, team2, map_name)
            st.metric("加时(OT)概率", f"{ot_rate:.1%}")
            st.progress(ot_rate, text="OT概率")
            st.divider()

# ---------- 主函数 ----------
def main():
    st.title("🎯 VALORANT 数据分析与预测系统")
    uploaded_file = st.file_uploader("上传 Excel 数据文件", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        with st.spinner("加载数据..."):
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.uploaded_data = df
                st.success(f"加载成功！共 {len(df)} 场有效记录。")
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        main_tabs = st.tabs(["🗺️ Map数据汇总", "👤 单战队数据查看", "⚔️ 对比与预测"])
        with main_tabs[0]:
            render_map_summary(df)
        with main_tabs[1]:
            render_team_view(df)
        with main_tabs[2]:
            render_compare_predict(df)
    else:
        st.info("👆 请上传Excel文件开始分析")

if __name__ == "__main__":
    main()
