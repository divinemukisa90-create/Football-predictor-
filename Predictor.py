
import streamlit as st
import requests
import pandas as pd
import numpy as np
from math import factorial, exp
from datetime import datetime

st.set_page_config(page_title="Football Poisson Predictor", layout="wide")
st.title("⚽ Poisson Predictor + Real Backtesting")
st.caption("1X2, Over/Under 2.5, BTTS using API-Football")

try:
    API_KEY = st.secrets["API_KEY"]
except:
    st.error("API key not found. Add it in Manage App > Secrets as API_KEY")
    st.stop()

SEASON = st.sidebar.selectbox("Season", [2024, 2023, 2022], index=0)

LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61
}
league_id = LEAGUES[st.sidebar.selectbox("League", list(LEAGUES.keys()))]
BASE_URL = "https://v3.football.api-sports.io"
headers = {"x-apisports-key": API_KEY}

tab1, tab2 = st.tabs(["Live Predictions", "Backtesting"])

def poisson_prob(mean, k):
    return (mean**k * np.exp(-mean)) / factorial(k)

def calculate_strengths(matches):
    goals_for, goals_against, games_played = {}, {}
    for m in matches:
        goals = m['goals']
        home, away = m['teams']['home']['name'], m['teams']['away']['name']
        hg, ag = goals['home'], goals['away']
        if hg is None: continue
        for team in [home, away]:
            goals_for.setdefault(team, 0)
            goals_against.setdefault(team, 0)
            games_played.setdefault(team, 0)
        goals_for[home] += hg
        goals_against[home] += ag
        goals_for[away] += ag
        goals_against[away] += hg
        games_played[home] += 1
        games_played[away] += 1

    if not games_played:
        return {}, 1.5
    avg_goals = np.mean([goals_for[t]/games_played[t] for t in games_played])
    attack = {t: (goals_for[t]/games_played[t]) / avg_goals for t in games_played}
    defense = {t: (goals_against[t]/games_played[t]) / avg_goals for t in games_played}
    return attack, defense, avg_goals

def predict_match(home, away, attack, defense, avg_goals, home_adv=0.25):
    if home not in attack or away not in attack:
        return None

    home_goals = attack[home] * defense[away] * avg_goals * (1 + home_adv)
    away_goals = attack[away] * defense[home] * avg_goals

    h, d, a = 0, 0, 0
    over25, under25 = 0, 0
    btts_yes, btts_no = 0, 0

    for i in range(7):
        for j in range(7):
            p = poisson_prob(home_goals, i) * poisson_prob(away_goals, j)
            total_goals = i + j

            if i > j: h += p
            elif i == j: d += p
            else: a += p

            if total_goals >= 3:
                over25 += p
            else:
                under25 += p

            if i >= 1 and j >= 1:
                btts_yes += p
            else:
                btts_no += p

    return {
        'H%': round(h*100, 1),
        'D%': round(d*100, 1),
        'A%': round(a*100, 1),
        'Over 2.5%': round(over25*100, 1),
        'Under 2.5%': round(under25*100, 1),
        'BTTS Yes%': round(btts_yes*100, 1),
        'BTTS No%': round(btts_no*100, 1)
    }

def odds_to_implied(odds):
    return round((1/odds)*100, 1) if odds and odds > 1 else 0

@st.cache_data(ttl=3600)
def get_fixtures(league, season, status=None):
    url = f"{BASE_URL}/fixtures"
    params = {"league": league, "season": season}
    if status: params["status"] = status
    r = requests.get(url, headers=headers, params=params, timeout=20)
    return r.json().get('response', [])

@st.cache_data(ttl=3600)
def get_odds(league, season, fixture_id=None, bet_type=1):
    url = f"{BASE_URL}/odds"
    params = {"league": league, "season": season, "bet": bet_type}
    if fixture_id: params["fixture"] = fixture_id
    r = requests.get(url, headers=headers, params=params, timeout=20)
    return r.json().get('response', [])

--- Tab 1: Live Predictions ---
with tab1:
    market = st.radio("Market", ["1X2", "Over/Under 2.5", "BTTS"], horizontal=True)
    try:
        today = datetime.utcnow().strftime('%Y-%m-%d')
        fixtures = get_fixtures(league_id, SEASON, status="NS")
        fixtures = [f for f in fixtures if f['fixture']['date'][:10]==today]

        if not fixtures:
            st.info("No fixtures today")
        else:
            hist = get_fixtures(league_id, SEASON, status="FT")
            attack, defense, avg = calculate_strengths(hist)

            results = []
            for f in fixtures[:10]:
                fixture_id = f['fixture']['id']
                home = f['teams']['home']['name']
                away = f['teams']['away']['name']

                pred = predict_match(home, away, attack, defense, avg)
                if not pred: continue

                bet_type = 1 if market=="1X2" else 5 if market=="Over/Under 2.5" else 8
                odds_data = get_odds(league_id, SEASON, fixture_id, bet_type)
                odds = {}

                if odds_data:
                    for b in odds_data[0].get('bookmakers', []):
                        if b['name']=='Pinnacle':
                            for bet in b['bets']:
                                if market=="1X2" and bet['name']=='Match Winner':
                                    for val in bet['values']:
                                        odds[val['value']+' Odds'] = float(val['odd'])
                                elif market=="Over/Under 2.5" and bet['name']=='Over/Under' and bet['values'][0]['handicap']=='2.5':
                                    for val in bet['values']:
                                        odds[val['value']+' Odds'] = float(val['odd'])
                                elif market=="BTTS" and bet['name']=='Both Teams To Score':
                                    for val in bet['values']:
                                        odds[val['value']+' Odds'] = float(val['odd'])
                                break

                row = {'Match': f"{home} vs {away}"}
                if market=="1X2":
                    row.update({'Model H%': pred['H%'], 'Model D%': pred['D%'], 'Model A%': pred['A%'],
                                'Home Odds': odds.get('Home Odds'), 'Draw Odds': odds.get('Draw Odds'), 'Away Odds': odds.get('Away Odds')})
                elif market=="Over/Under 2.5":
                    row.update({'Model Over 2.5%': pred['Over 2.5%'], 'Model Under 2.5%': pred['Under 2.5%'],
                                'Over Odds': odds.get('Over Odds'), 'Under Odds': odds.get('Under Odds')})
                else:
                    row.update({'Model BTTS Yes%': pred['BTTS Yes%'], 'Model BTTS No%': pred['BTTS No%'],
                                'Yes Odds': odds.get('Yes Odds'), 'No Odds': odds.get('No Odds')})

                results.append(row)

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            st.caption("Green edge >3% appears when you sort. Check the raw numbers for now.")
    except Exception as e:
        st.error(f"Error: {e}")

--- Tab 2: Backtesting ---
with tab2:
    market = st.radio("Market to Backtest", ["1X2", "Over/Under 2.5", "BTTS"], horizontal=True, key="bt_market")
    min_edge = st.sidebar.slider("Min Edge %", 1.0, 10.0, 3.0, 0.5)

    if st.button("Run Backtest"):
        try:
            with st.spinner("Loading historical data..."):
                finished = get_fixtures(league_id, SEASON, status="FT")
                if len(finished) < 50:
                    st.warning("Not enough finished matches")
                    st.stop()

                split = int(len(finished)*0.8)
                train_data = finished[:split]
                test_data = finished[split:split+50]

                attack, defense, avg = calculate_strengths(train_data)

                bets, profit, wins = 0, 0, 0
                equity_curve = []

                for m in test_data:
                    fixture_id = m['fixture']['id']
                    home = m['teams']['home']['name']
                    away = m['teams']['away']['name']
                    hg, ag = m['goals']['home'], m['goals']['away']
                    total_goals = hg + ag

                    pred = predict_match(home, away, attack, defense, avg)
                    if not pred: continue

                    bet_type = 1 if market=="1X2" else 5 if market=="Over/Under 2.5" else 8
                    odds_data = get_odds(league_id, SEASON, fixture_id, bet_type)
                    if not odds_data: continue

                    odds_list, model_list = [], []
                    if market=="1X2":
                        for b in odds_data[0].get('bookmakers', []):
                            if b['name']=='Pinnacle':
                                for bet in b['bets']:
                                    if bet['name']=='Match Winner':
                                        for val in bet['values']:
                                            if val['value']=='Home': odds_list.append(float(val['odd']))
                                            if val['value']=='Draw': odds_list.append(float(val['odd']))
                                            if val['value']=='Away': odds_list.append(float(val['odd']))
                                break
                        model_list = [pred['H%'], pred['D%'], pred['A%']]
                        actual_idx = 0 if hg>ag else 1 if hg==ag else 2

                    elif market=="Over/Under 2.5":
                        for b in odds_data[0].get('bookmakers', []):
                            if b['name']=='Pinnacle':
                                for bet in b['bets']:
                                    if bet['name']=='Over/Under' and bet['values'][0]['handicap']=='2.5':
                                        for val in bet['values']:
                                            if val['value']=='Over': odds_list.append(float(val['odd']))
                                            if val['value']=='Under': odds_list.append(float(val['odd']))
                                break
                        model_list = [pred['Over 2.5%'], pred['Under 2.5%']]
                        actual_idx = 0 if total_goals >= 3 else 1

                    else: # BTTS
                        for b in odds_data[0].get('bookmakers', []):
                            if b['name']=='Pinnacle':
                                for bet in b['bets']:
                                    if bet['name']=='Both Teams To Score':
                                        for val in bet['values']:
                                            if val['value']=='Yes': odds_list.append(float(val['odd']))
                                            if val['value']=='No': odds_list.append(float(val['odd']))
                                break
                        model_list = [pred['BTTS Yes%'], pred['BTTS No%']]
                        actual_idx = 0 if hg>=1 and ag>=1 else 1

                    if len(odds_list) < 2: continue

                    imp = [odds_to_implied(o) for o in odds_list]        edges = [m-i for m,i in zip(model_list, imp)]
                    max_edge = max(edges)
                    pick_idx = edges.index(max_edge)

                    if max_edge >= min_edge:
                        bets += 1
                        if pick_idx == actual_idx:
                            wins += 1
                            profit += 100 * (odds_list[pick_idx]-1)
                        else:
                            profit -= 100
                        equity_curve.append(profit)

                roi = round(profit/(bets*100)*100, 2) if bets>0 else 0
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Bets", bets)
                col2.metric("Wins", wins)
                col3.metric("Profit", f"${round(profit,2)}")
                col4.metric("ROI", f"{roi}%")

                if equity_curve:
                    st.subheader("Cumulative Profit")
                    chart_df = pd.DataFrame({
                        'Bet #': range(1, len(equity_curve)+1),
                        'Profit $': equity_curve
                     })
                    st.line_chart(chart_df.set_index('Bet #'))

                st.caption("Limited to 50 matches. API-Football free tier: 100 req/day")
        except Exception as e:
            st.error(f"Backtest error: {e}")
 
