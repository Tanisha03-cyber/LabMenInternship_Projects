from sqlalchemy import create_engine
from etl.extract import (
    extract_agg_transaction, extract_agg_user, extract_agg_insurance,
    extract_map_transaction, extract_map_user, extract_map_insurance,
    extract_top_transaction, extract_top_user, extract_top_insurance
)

# ── DB CONNECTION ────────────────────────────────────────────────────
from urllib.parse import quote_plus
password = quote_plus("Pr@santh001")
ENGINE = create_engine(f"mysql+mysqlconnector://root:{password}@localhost/phonepe_pulse")

def load_all():
    tables = {
        "agg_transaction"  : extract_agg_transaction(),
        "agg_user"         : extract_agg_user(),
        "agg_insurance"    : extract_agg_insurance(),
        "map_transaction"  : extract_map_transaction(),
        "map_user"         : extract_map_user(),
        "map_insurance"    : extract_map_insurance(),
        "top_transaction"  : extract_top_transaction(),
        "top_user"         : extract_top_user(),
        "top_insurance"    : extract_top_insurance()
    }

    for table_name, df in tables.items():
        df.to_sql(table_name, con=ENGINE, if_exists="replace", index=False)
        print(f"Loaded {len(df)} rows into '{table_name}'")

    print("\n All 9 tables loaded into MySQL successfully!")

if __name__ == "__main__":
    load_all()
