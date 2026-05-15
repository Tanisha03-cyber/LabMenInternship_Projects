import os
import json
import pandas as pd

BASE_PATH = "data/pulse/data"

# ── 1. AGG TRANSACTION ──────────────────────────────────────────────
def extract_agg_transaction():
    rows = []
    path = os.path.join(BASE_PATH, "aggregated/transaction/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for txn in data["data"]["transactionData"]:
                    rows.append([state, int(year), quarter,
                                  txn["name"],
                                  txn["paymentInstruments"][0]["count"],
                                  txn["paymentInstruments"][0]["amount"]])
    return pd.DataFrame(rows, columns=["state","year","quarter","transaction_type","transaction_count","transaction_amount"])

# ── 2. AGG USER ─────────────────────────────────────────────────────
def extract_agg_user():
    rows = []
    path = os.path.join(BASE_PATH, "aggregated/user/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                try:
                    for brand in data["data"]["usersByDevice"]:
                        rows.append([state, int(year), quarter,
                                      brand["brand"], brand["count"], brand["percentage"]])
                except (TypeError, KeyError):
                    pass
    return pd.DataFrame(rows, columns=["state","year","quarter","brand","registered_users","percentage"])

# ── 3. AGG INSURANCE ────────────────────────────────────────────────
def extract_agg_insurance():
    rows = []
    path = os.path.join(BASE_PATH, "aggregated/insurance/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for txn in data["data"]["transactionData"]:
                    rows.append([state, int(year), quarter,
                                  txn["name"],
                                  txn["paymentInstruments"][0]["count"],
                                  txn["paymentInstruments"][0]["amount"]])
    return pd.DataFrame(rows, columns=["state","year","quarter","insurance_type","insurance_count","insurance_amount"])

# ── 4. MAP TRANSACTION ──────────────────────────────────────────────
def extract_map_transaction():
    rows = []
    path = os.path.join(BASE_PATH, "map/transaction/hover/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for district in data["data"]["hoverDataList"]:
                    rows.append([state, int(year), quarter,
                                  district["name"],
                                  district["metric"][0]["count"],
                                  district["metric"][0]["amount"]])
    return pd.DataFrame(rows, columns=["state","year","quarter","district","transaction_count","transaction_amount"])

# ── 5. MAP USER ─────────────────────────────────────────────────────
def extract_map_user():
    rows = []
    path = os.path.join(BASE_PATH, "map/user/hover/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for district in data["data"]["hoverData"].items():
                    rows.append([state, int(year), quarter,
                                  district[0],
                                  district[1]["registeredUsers"],
                                  district[1]["appOpens"]])
    return pd.DataFrame(rows, columns=["state","year","quarter","district","registered_users","app_opens"])

# ── 6. MAP INSURANCE ────────────────────────────────────────────────
def extract_map_insurance():
    rows = []
    path = os.path.join(BASE_PATH, "map/insurance/hover/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for district in data["data"]["hoverDataList"]:
                    rows.append([state, int(year), quarter,
                                  district["name"],
                                  district["metric"][0]["count"],
                                  district["metric"][0]["amount"]])
    return pd.DataFrame(rows, columns=["state","year","quarter","district","insurance_count","insurance_amount"])

# ── 7. TOP TRANSACTION ──────────────────────────────────────────────
def extract_top_transaction():
    rows = []
    path = os.path.join(BASE_PATH, "top/transaction/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for level in ["districts", "pincodes"]:
                    try:
                        for item in data["data"][level]:
                            rows.append([state, int(year), quarter, level,
                                          item["entityName"],
                                          item["metric"]["count"],
                                          item["metric"]["amount"]])
                    except (TypeError, KeyError):
                        pass
    return pd.DataFrame(rows, columns=["state","year","quarter","entity_level","entity_name","transaction_count","transaction_amount"])

# ── 8. TOP USER ─────────────────────────────────────────────────────
def extract_top_user():
    rows = []
    path = os.path.join(BASE_PATH, "top/user/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for level in ["districts", "pincodes"]:
                    try:
                        for item in data["data"][level]:
                            rows.append([state, int(year), quarter, level,
                                          item["name"],
                                          item["registeredUsers"]])
                    except (TypeError, KeyError):
                        pass
    return pd.DataFrame(rows, columns=["state","year","quarter","entity_level","entity_name","registered_users"])

# ── 9. TOP INSURANCE ────────────────────────────────────────────────
def extract_top_insurance():
    rows = []
    path = os.path.join(BASE_PATH, "top/insurance/country/india/state")
    for state in os.listdir(path):
        for year in os.listdir(os.path.join(path, state)):
            for file in os.listdir(os.path.join(path, state, year)):
                quarter = int(file.replace(".json", ""))
                with open(os.path.join(path, state, year, file)) as f:
                    data = json.load(f)
                for level in ["districts", "pincodes"]:
                    try:
                        for item in data["data"][level]:
                            rows.append([state, int(year), quarter, level,
                                          item["entityName"],
                                          item["metric"]["count"],
                                          item["metric"]["amount"]])
                    except (TypeError, KeyError):
                        pass
    return pd.DataFrame(rows, columns=["state","year","quarter","entity_level","entity_name","insurance_count","insurance_amount"])
