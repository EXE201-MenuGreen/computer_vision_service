-- Visual embeddings table for CLIP image features (Option 2)
-- Stores CLIP ViT-B/32 embeddings (512-dim) from each food crop inference.
-- Enables future visual similarity search (Option 3).

create table if not exists public.food_visual_embeddings (
    id           bigserial primary key,
    request_id   text        not null,
    food_label   text        not null,
    confidence   real        not null,
    embedding    vector(512) not null,
    created_at   timestamptz not null default now()
);

-- HNSW index for approximate nearest-neighbour visual search
create index if not exists food_visual_embeddings_hnsw_idx
    on public.food_visual_embeddings
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

create index if not exists food_visual_embeddings_label_idx
    on public.food_visual_embeddings (food_label);

create index if not exists food_visual_embeddings_request_idx
    on public.food_visual_embeddings (request_id);

-- RPC: find visually similar food crops
create or replace function public.match_visual_food(
    query_embedding      vector(512),
    similarity_threshold float,
    match_count          int default 5
)
returns table (
    id           bigint,
    request_id   text,
    food_label   text,
    confidence   real,
    similarity   float,
    created_at   timestamptz
)
language sql stable
as $$
    select
        fve.id,
        fve.request_id,
        fve.food_label,
        fve.confidence,
        1 - (fve.embedding <=> query_embedding) as similarity,
        fve.created_at
    from public.food_visual_embeddings fve
    where 1 - (fve.embedding <=> query_embedding) > similarity_threshold
    order by fve.embedding <=> query_embedding
    limit match_count;
$$;

-- RLS policies
alter table public.food_visual_embeddings enable row level security;

create policy "anon_select" on public.food_visual_embeddings
    for select using (true);

create policy "anon_insert" on public.food_visual_embeddings
    for insert with check (true);

grant execute on function public.match_visual_food to anon;
