-- PhonePe Transaction Insights — Database Schema
-- PostgreSQL 15+

CREATE TABLE IF NOT EXISTS agg_transaction (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    transaction_type    VARCHAR(100),
    transaction_count   BIGINT,
    transaction_amount  FLOAT
);

CREATE TABLE IF NOT EXISTS agg_users (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    brand               VARCHAR(100),
    registered_users    BIGINT,
    percentage          FLOAT
);

CREATE TABLE IF NOT EXISTS agg_insurance (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    policy_count        BIGINT,
    premium_amount      FLOAT
);

CREATE TABLE IF NOT EXISTS map_transaction (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    district            VARCHAR(100),
    transaction_count   BIGINT,
    transaction_amount  FLOAT
);

CREATE TABLE IF NOT EXISTS map_users (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    district            VARCHAR(100),
    registered_users    BIGINT,
    app_opens           BIGINT
);

CREATE TABLE IF NOT EXISTS map_insurance (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    district            VARCHAR(100),
    policy_count        BIGINT,
    premium_amount      FLOAT
);

CREATE TABLE IF NOT EXISTS top_transaction (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    entity_name         VARCHAR(100),
    entity_type         VARCHAR(50),
    transaction_count   BIGINT,
    transaction_amount  FLOAT
);

CREATE TABLE IF NOT EXISTS top_users (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    entity_name         VARCHAR(100),
    entity_type         VARCHAR(50),
    registered_users    BIGINT
);

CREATE TABLE IF NOT EXISTS top_insurance (
    state               VARCHAR(100),
    year                INTEGER,
    quarter             INTEGER,
    entity_name         VARCHAR(100),
    entity_type         VARCHAR(50),
    policy_count        BIGINT,
    premium_amount      FLOAT
);
