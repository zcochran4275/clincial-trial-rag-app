-- How common is each adverse event across trial arms? Pooled across serious + other tables.
-- Phase 3 adds a drug column per arm (from the agent's drug-normalization output) so this
-- can be grouped by drug and joined to FAERS PRR results.
with by_arm as (
    select
        nct_id,
        group_id,
        term_key,
        max(organ_system)  as organ_system,
        max(num_affected)  as num_affected,   -- serious and other tables overlap; take max, don't double count
        max(num_at_risk)   as num_at_risk
    from {{ ref('stg_arm_adverse_events') }}
    group by nct_id, group_id, term_key
)
select
    term_key,
    organ_system,
    count(distinct nct_id)                     as n_trials,
    count(*)                                   as n_arms,
    sum(num_affected)                          as total_affected,
    sum(num_at_risk)                           as total_at_risk,
    sum(num_affected) / nullif(sum(num_at_risk), 0) as pooled_rate
from by_arm
group by term_key, organ_system
