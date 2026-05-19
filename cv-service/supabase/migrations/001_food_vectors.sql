-- Enable pgvector extension (run once per Supabase project)
create extension if not exists vector with schema extensions;

-- Food nutrition table with vector embeddings
create table if not exists public.food_nutrition (
    id             bigserial primary key,
    label          text        not null unique,
    display_name   text        not null,
    search_text    text        not null,
    embedding      vector(384) not null,
    calories_kcal  real        not null,
    protein_g      real        not null,
    carbs_g        real        not null,
    fat_g          real        not null,
    fiber_g        real,
    fdc_id         text,
    source         text        not null default 'fallback',
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now()
);

-- HNSW index (better than IVFFlat for ~300 rows, no training needed)
create index if not exists food_nutrition_embedding_hnsw_idx
    on public.food_nutrition
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

create index if not exists food_nutrition_label_idx
    on public.food_nutrition (label);

-- Auto-update updated_at
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create trigger food_nutrition_updated_at
    before update on public.food_nutrition
    for each row execute function public.set_updated_at();

-- RPC: semantic similarity search
create or replace function public.match_food(
    query_embedding  vector(384),
    similarity_threshold float,
    match_count      int default 1
)
returns table (
    label          text,
    display_name   text,
    calories_kcal  real,
    protein_g      real,
    carbs_g        real,
    fat_g          real,
    fiber_g        real,
    fdc_id         text,
    source         text,
    similarity     float
)
language sql stable
as $$
    select
        fn.label,
        fn.display_name,
        fn.calories_kcal,
        fn.protein_g,
        fn.carbs_g,
        fn.fat_g,
        fn.fiber_g,
        fn.fdc_id,
        fn.source,
        1 - (fn.embedding <=> query_embedding) as similarity
    from public.food_nutrition fn
    where 1 - (fn.embedding <=> query_embedding) > similarity_threshold
    order by fn.embedding <=> query_embedding
    limit match_count;
$$;

-- RPC: upsert a food entry (insert or update by label)
create or replace function public.upsert_food(
    p_label          text,
    p_display_name   text,
    p_search_text    text,
    p_embedding      vector(384),
    p_calories_kcal  real,
    p_protein_g      real,
    p_carbs_g        real,
    p_fat_g          real,
    p_fiber_g        real,
    p_fdc_id         text,
    p_source         text default 'usda'
)
returns void
language sql
as $$
    insert into public.food_nutrition
        (label, display_name, search_text, embedding,
         calories_kcal, protein_g, carbs_g, fat_g, fiber_g, fdc_id, source)
    values
        (p_label, p_display_name, p_search_text, p_embedding,
         p_calories_kcal, p_protein_g, p_carbs_g, p_fat_g, p_fiber_g, p_fdc_id, p_source)
    on conflict (label) do update set
        display_name  = excluded.display_name,
        search_text   = excluded.search_text,
        embedding     = excluded.embedding,
        calories_kcal = excluded.calories_kcal,
        protein_g     = excluded.protein_g,
        carbs_g       = excluded.carbs_g,
        fat_g         = excluded.fat_g,
        fiber_g       = excluded.fiber_g,
        fdc_id        = coalesce(excluded.fdc_id, food_nutrition.fdc_id),
        source        = excluded.source,
        updated_at    = now();
$$;

-- RLS policies
alter table public.food_nutrition enable row level security;

create policy "anon_select" on public.food_nutrition
    for select using (true);

grant execute on function public.match_food to anon;
grant execute on function public.upsert_food to anon;
