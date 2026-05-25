-- ─────────────────────────────────────────────────────────────
-- Migration 004: Tier-0 admin-verified nutrition table
--
-- Purpose: Admin/nutritionist-curated ground-truth entries that
--   override every automated lookup tier (pgvector, USDA, fallback).
--   Only humans write here; the service only reads.
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS food_nutrition_verified (
    food_label      TEXT PRIMARY KEY,
    calories_kcal   FLOAT NOT NULL,
    protein_g       FLOAT NOT NULL,
    carbs_g         FLOAT NOT NULL,
    fat_g           FLOAT NOT NULL,
    fiber_g         FLOAT,
    verified_by     TEXT,
    verified_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes           TEXT
);

-- Service role can insert/update; anon role can only read.
ALTER TABLE food_nutrition_verified ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_read_verified"
    ON food_nutrition_verified FOR SELECT
    USING (true);

CREATE POLICY "service_write_verified"
    ON food_nutrition_verified FOR ALL
    USING (auth.role() = 'service_role');

-- Index for fast exact-label lookup (already PK, but explicit for docs)
-- PK already creates a btree index; no extra index needed.

-- ── RPC: exact label lookup ────────────────────────────────
CREATE OR REPLACE FUNCTION get_verified_food(p_label TEXT)
RETURNS TABLE (
    calories_kcal   FLOAT,
    protein_g       FLOAT,
    carbs_g         FLOAT,
    fat_g           FLOAT,
    fiber_g         FLOAT
)
LANGUAGE sql STABLE SECURITY DEFINER
AS $$
    SELECT calories_kcal, protein_g, carbs_g, fat_g, fiber_g
    FROM food_nutrition_verified
    WHERE food_label = p_label
    LIMIT 1;
$$;

-- ── RPC: upsert a verified entry (called by admin tooling) ─
CREATE OR REPLACE FUNCTION upsert_verified_food(
    p_label         TEXT,
    p_calories_kcal FLOAT,
    p_protein_g     FLOAT,
    p_carbs_g       FLOAT,
    p_fat_g         FLOAT,
    p_fiber_g       FLOAT DEFAULT NULL,
    p_verified_by   TEXT DEFAULT NULL,
    p_notes         TEXT DEFAULT NULL
)
RETURNS VOID
LANGUAGE sql SECURITY DEFINER
AS $$
    INSERT INTO food_nutrition_verified
        (food_label, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, verified_by, notes, verified_at)
    VALUES
        (p_label, p_calories_kcal, p_protein_g, p_carbs_g, p_fat_g, p_fiber_g, p_verified_by, p_notes, now())
    ON CONFLICT (food_label) DO UPDATE SET
        calories_kcal = EXCLUDED.calories_kcal,
        protein_g     = EXCLUDED.protein_g,
        carbs_g       = EXCLUDED.carbs_g,
        fat_g         = EXCLUDED.fat_g,
        fiber_g       = EXCLUDED.fiber_g,
        verified_by   = EXCLUDED.verified_by,
        notes         = EXCLUDED.notes,
        verified_at   = now();
$$;
