-- Meal history table for Option 3: RAG-based meal history query
-- Stores AnalysisResult with 384-dim sentence-transformer embedding
-- RLS: each user sees only their own meals

create table if not exists public.meal_history (
    id            uuid        primary key default gen_random_uuid(),
    user_id       uuid        not null,
    request_id    text        not null unique,
    foods_json    jsonb       not null,
    total_macros  jsonb       not null,
    meal_text     text        not null,
    embedding     vector(384) not null,
    analyzed_at   timestamptz not null default now()
);

create index if not exists meal_history_embedding_hnsw_idx
    on public.meal_history
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

create index if not exists meal_history_user_time_idx
    on public.meal_history (user_id, analyzed_at desc);

-- RLS: user sees only their own meals
alter table public.meal_history enable row level security;

create policy "owner_only" on public.meal_history
    using (user_id = auth.uid());

-- RPC: semantic query meal history for a user (called with service role)
create or replace function public.query_meal_history(
    p_user_id            uuid,
    query_embedding      vector(384),
    similarity_threshold float default 0.3,
    match_count          int   default 10
)
returns table (
    id           uuid,
    request_id   text,
    foods_json   jsonb,
    total_macros jsonb,
    similarity   float,
    analyzed_at  timestamptz
)
language sql stable
as $$
    select
        mh.id,
        mh.request_id,
        mh.foods_json,
        mh.total_macros,
        1 - (mh.embedding <=> query_embedding) as similarity,
        mh.analyzed_at
    from public.meal_history mh
    where mh.user_id = p_user_id
      and 1 - (mh.embedding <=> query_embedding) > similarity_threshold
    order by mh.embedding <=> query_embedding
    limit match_count;
$$;

-- RPC: recent meals (no vector search)
create or replace function public.get_recent_meals(
    p_user_id   uuid,
    match_count int default 10
)
returns table (
    id           uuid,
    request_id   text,
    foods_json   jsonb,
    total_macros jsonb,
    analyzed_at  timestamptz
)
language sql stable
as $$
    select id, request_id, foods_json, total_macros, analyzed_at
    from public.meal_history
    where user_id = p_user_id
    order by analyzed_at desc
    limit match_count;
$$;

-- RPC: store meal (called with service role key — bypasses RLS)
create or replace function public.store_meal(
    p_user_id      uuid,
    p_request_id   text,
    p_foods_json   jsonb,
    p_total_macros jsonb,
    p_meal_text    text,
    p_embedding    vector(384)
)
returns void
language sql
as $$
    insert into public.meal_history
        (user_id, request_id, foods_json, total_macros, meal_text, embedding)
    values
        (p_user_id, p_request_id, p_foods_json, p_total_macros, p_meal_text, p_embedding)
    on conflict (request_id) do nothing;
$$;

grant execute on function public.query_meal_history  to service_role;
grant execute on function public.get_recent_meals    to service_role;
grant execute on function public.store_meal          to service_role;
grant execute on function public.query_meal_history  to authenticated;
grant execute on function public.get_recent_meals    to authenticated;
