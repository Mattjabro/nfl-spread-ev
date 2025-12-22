with tab3:
    st.subheader("Historical Predictions (Model Bets)")

    week = st.selectbox(
        "Select Week",
        options=list(range(1, 16)),
        index=0,
    )

    hist = load_historical_week(SEASON, week)
    actuals = load_actual_results(SEASON)

    if hist is None:
        st.warning("No data found for this week.")
        st.stop()

    # actual_margin in your file is HOME - AWAY
    actuals = actuals[actuals["season"] == SEASON]

    hist = hist.merge(
        actuals,
        on=["season", "week", "home_team", "away_team"],
        how="left"
    )

    rows = []

    for _, g in hist.iterrows():
        away = g["away_team"]
        home = g["home_team"]

        mu_home = float(g["model_spread_home"])
        sigma = max(float(g["sigma"]), MIN_SIGMA)

        # keep this, since you said bet column is correct with it
        spread_home = -float(g["vegas_spread_home"])

        # actual_margin is HOME - AWAY
        m_home = g["actual_margin"]
        if pd.isna(m_home):
            continue
        m_home = float(m_home)
        m_away = -m_home

        # ---- probabilities (same math as Tab 1) ----
        skew_adj = 0.15 * np.sign(mu_home)
        z_home = (mu_home + spread_home + skew_adj) / sigma
        prob_home = float(student_t_cdf(z_home / temperature, df=6))
        prob_away = 1 - prob_home

        # ---- EXACT SAME BET CONSTRUCTION AS TAB 1 ----
        if prob_home >= prob_away:
            bet_team = home
            bet_line = spread_home
            prob = prob_home
        else:
            bet_team = away
            bet_line = -spread_home
            prob = prob_away

        bet = f"{bet_team} {bet_line:+.1f}"
        ev = ev_from_prob(prob, odds_price)

        # ---- grade using bet_team/bet_line + HOME-AWAY margin ----
        margin_for_bet = m_home if bet_team == home else m_away
        cover_val = margin_for_bet + bet_line

        if abs(cover_val) < 1e-9:
            result = "➖ Push"
        elif cover_val > 0:
            result = "✅ Win"
        else:
            result = "❌ Loss"

        # ---- display actual margin (AWAY - HOME, to match your expected UI) ----
        margin_away_minus_home = -m_home
        if abs(margin_away_minus_home - round(margin_away_minus_home)) < 1e-9:
            actual_margin_display = f"{away} {int(round(margin_away_minus_home)):+d}"
        else:
            actual_margin_display = f"{away} {margin_away_minus_home:+.1f}"

        rows.append({
            "matchup": f"{away} @ {home}",
            "bet": bet,
            "model_mu": format_matchup_spread(away, home, mu_home),
            "cover_prob": prob,
            "bet_ev": ev,
            "actual_margin": actual_margin_display,
            "result": result,
        })

    table = pd.DataFrame(rows).sort_values("bet_ev", ascending=False)

    st.dataframe(
        table.style
        .format({
            "cover_prob": "{:.3f}",
            "bet_ev": "{:.3f}",
        })
        .applymap(color_results, subset=["result"]),
        use_container_width=True,
    )