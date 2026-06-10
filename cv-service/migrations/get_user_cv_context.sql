-- MenuGreen backend: expose via PostgREST as RPC get_user_cv_context(p_user_id uuid)
-- Called by cv-service to load allergies, health profile, and dietary restrictions.

CREATE OR REPLACE FUNCTION get_user_cv_context(p_user_id uuid)
RETURNS jsonb
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT jsonb_build_object(
    'allergies', COALESCE((
      SELECT jsonb_agg(jsonb_build_object(
        'allergen_key', a.allergen_key,
        'name', a.name,
        'severity', ua.severity
      ) ORDER BY a.name)
      FROM user_allergies ua
      JOIN allergies a ON a.id = ua.allergy_id
      WHERE ua.user_id = p_user_id
    ), '[]'::jsonb),
    'health_conditions', COALESCE((
      SELECT jsonb_agg(jsonb_build_object(
        'condition_key', hp.condition_key,
        'name', hp.condition_name,
        'notes', COALESCE(hp.notes, '')
      ))
      FROM health_profiles hp
      WHERE hp.user_id = p_user_id
        AND hp.condition_key IS NOT NULL
    ), '[]'::jsonb),
    'dietary_goal', (
      SELECT hp.dietary_goal FROM health_profiles hp
      WHERE hp.user_id = p_user_id LIMIT 1
    ),
    'avoid_ingredient_keys', COALESCE((
      SELECT hp.avoid_ingredient_keys FROM health_profiles hp
      WHERE hp.user_id = p_user_id LIMIT 1
    ), '[]'::jsonb),
    'daily_calorie_limit', (
      SELECT hp.daily_calorie_limit FROM health_profiles hp
      WHERE hp.user_id = p_user_id LIMIT 1
    ),
    'daily_protein_limit', (
      SELECT hp.daily_protein_limit FROM health_profiles hp
      WHERE hp.user_id = p_user_id LIMIT 1
    ),
    'daily_carbs_limit', (
      SELECT hp.daily_carbs_limit FROM health_profiles hp
      WHERE hp.user_id = p_user_id LIMIT 1
    ),
    'daily_fat_limit', (
      SELECT hp.daily_fat_limit FROM health_profiles hp
      WHERE hp.user_id = p_user_id LIMIT 1
    )
  );
$$;

-- Grant execute to the PostgREST service role used by cv-service.
-- GRANT EXECUTE ON FUNCTION get_user_cv_context(uuid) TO service_role;
