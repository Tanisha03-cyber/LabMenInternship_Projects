from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

# ── CONNECTION ───────────────────────────────────────────────────────
password = quote_plus("Pr@santh001")
ENGINE = create_engine(f"mysql+mysqlconnector://root:{password}@localhost/phonepe_pulse")

def run_query(sql):
    with ENGINE.connect() as conn:
        return conn.execute(text(sql))

# ══════════════════════════════════════════════════════════════════════
# CASE 1 - Transaction Dynamics
# ══════════════════════════════════════════════════════════════════════

def q1_quarterly_trend():
    return run_query("""
        SELECT year, quarter,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM agg_transaction
        GROUP BY year, quarter
        ORDER BY year, quarter
    """)

def q1_category_split():
    return run_query("""
        SELECT transaction_type,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM agg_transaction
        GROUP BY transaction_type
        ORDER BY total_amount DESC
    """)

def q1_top10_states():
    return run_query("""
        SELECT state,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM agg_transaction
        GROUP BY state
        ORDER BY total_amount DESC
        LIMIT 10
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 2 - Device Dominance & User Engagement
# ══════════════════════════════════════════════════════════════════════

def q2_brand_users():
    return run_query("""
        SELECT brand,
               SUM(registered_users) AS total_users,
               ROUND(SUM(percentage), 2) AS avg_share
        FROM agg_user
        GROUP BY brand
        ORDER BY total_users DESC
    """)

def q2_engagement_ratio():
    return run_query("""
        SELECT state,
               SUM(registered_users) AS total_users,
               SUM(app_opens) AS total_opens,
               ROUND(SUM(app_opens) / NULLIF(SUM(registered_users), 0) * 100, 2) AS engagement_pct
        FROM map_user
        GROUP BY state
        ORDER BY engagement_pct DESC
    """)

def q2_low_engagement_districts():
    return run_query("""
        SELECT state, district,
               SUM(registered_users) AS users,
               SUM(app_opens) AS opens,
               ROUND(SUM(app_opens) / NULLIF(SUM(registered_users), 0) * 100, 2) AS engagement_pct
        FROM map_user
        GROUP BY state, district
        HAVING engagement_pct < 50
        ORDER BY users DESC
        LIMIT 15
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 3 - Insurance Penetration & Growth
# ══════════════════════════════════════════════════════════════════════

def q3_insurance_growth():
    return run_query("""
        SELECT year, quarter,
               SUM(insurance_count) AS total_policies,
               ROUND(SUM(insurance_amount), 2) AS total_premium
        FROM agg_insurance
        GROUP BY year, quarter
        ORDER BY year, quarter
    """)

def q3_state_insurance():
    return run_query("""
        SELECT state,
               SUM(insurance_count) AS total_policies,
               ROUND(SUM(insurance_amount), 2) AS total_premium
        FROM map_insurance
        GROUP BY state
        ORDER BY total_policies DESC
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 4 - Transaction Market Expansion
# ══════════════════════════════════════════════════════════════════════

def q4_state_market():
    return run_query("""
        SELECT state,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount,
               COUNT(DISTINCT year) AS years_active
        FROM map_transaction
        GROUP BY state
        ORDER BY total_amount DESC
    """)

def q4_district_growth():
    return run_query("""
        SELECT state, district,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM map_transaction
        GROUP BY state, district
        ORDER BY total_count DESC
        LIMIT 20
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 5 - User Engagement & Growth Strategy
# ══════════════════════════════════════════════════════════════════════

def q5_state_user_growth():
    return run_query("""
        SELECT state, year,
               SUM(registered_users) AS total_users,
               SUM(app_opens) AS total_opens
        FROM map_user
        GROUP BY state, year
        ORDER BY total_users DESC
    """)

def q5_retention_risk():
    return run_query("""
        SELECT state,
               SUM(registered_users) AS users,
               SUM(app_opens) AS opens,
               ROUND(SUM(app_opens) / NULLIF(SUM(registered_users), 0) * 100, 2) AS open_rate
        FROM map_user
        GROUP BY state
        HAVING open_rate < 100
        ORDER BY users DESC
        LIMIT 10
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 6 - Insurance Engagement
# ══════════════════════════════════════════════════════════════════════

def q6_district_insurance():
    return run_query("""
        SELECT state, district,
               SUM(insurance_count) AS total_policies,
               ROUND(SUM(insurance_amount), 2) AS total_premium
        FROM map_insurance
        GROUP BY state, district
        ORDER BY total_policies DESC
        LIMIT 20
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 7 - Top Transaction States, Districts, Pincodes
# ══════════════════════════════════════════════════════════════════════

def q7_top_states():
    return run_query("""
        SELECT state,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM top_transaction
        WHERE entity_level = 'districts'
        GROUP BY state
        ORDER BY total_count DESC
        LIMIT 10
    """)

def q7_top_districts():
    return run_query("""
        SELECT state, entity_name AS district,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM top_transaction
        WHERE entity_level = 'districts'
        GROUP BY state, entity_name
        ORDER BY total_count DESC
        LIMIT 10
    """)

def q7_top_pincodes():
    return run_query("""
        SELECT state, entity_name AS pincode,
               SUM(transaction_count) AS total_count,
               ROUND(SUM(transaction_amount), 2) AS total_amount
        FROM top_transaction
        WHERE entity_level = 'pincodes'
        GROUP BY state, entity_name
        ORDER BY total_count DESC
        LIMIT 10
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 8 - User Registration Analysis
# ══════════════════════════════════════════════════════════════════════

def q8_top_states_users():
    return run_query("""
        SELECT state,
               SUM(registered_users) AS total_users
        FROM top_user
        WHERE entity_level = 'districts'
        GROUP BY state
        ORDER BY total_users DESC
        LIMIT 10
    """)

def q8_top_districts_users():
    return run_query("""
        SELECT state, entity_name AS district,
               SUM(registered_users) AS total_users
        FROM top_user
        WHERE entity_level = 'districts'
        GROUP BY state, entity_name
        ORDER BY total_users DESC
        LIMIT 10
    """)

def q8_top_pincodes_users():
    return run_query("""
        SELECT state, entity_name AS pincode,
               SUM(registered_users) AS total_users
        FROM top_user
        WHERE entity_level = 'pincodes'
        GROUP BY state, entity_name
        ORDER BY total_users DESC
        LIMIT 10
    """)

# ══════════════════════════════════════════════════════════════════════
# CASE 9 - Insurance Transactions Analysis
# ══════════════════════════════════════════════════════════════════════

def q9_top_states_insurance():
    return run_query("""
        SELECT state,
               SUM(insurance_count) AS total_policies,
               ROUND(SUM(insurance_amount), 2) AS total_premium
        FROM top_insurance
        WHERE entity_level = 'districts'
        GROUP BY state
        ORDER BY total_policies DESC
        LIMIT 10
    """)

def q9_top_districts_insurance():
    return run_query("""
        SELECT state, entity_name AS district,
               SUM(insurance_count) AS total_policies,
               ROUND(SUM(insurance_amount), 2) AS total_premium
        FROM top_insurance
        WHERE entity_level = 'districts'
        GROUP BY state, entity_name
        ORDER BY total_policies DESC
        LIMIT 10
    """)

def q9_top_pincodes_insurance():
    return run_query("""
        SELECT state, entity_name AS pincode,
               SUM(insurance_count) AS total_policies,
               ROUND(SUM(insurance_amount), 2) AS total_premium
        FROM top_insurance
        WHERE entity_level = 'pincodes'
        GROUP BY state, entity_name
        ORDER BY total_policies DESC
        LIMIT 10
    """)
