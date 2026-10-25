-- One row per trial x arm x MedDRA term x seriousness, with a normalized term key
-- so trial terms ("Diarrhoea") join to FAERS terms ("DIARRHOEA").
select
    nct_id,
    group_id,
    group_title,
    seriousness,
    term,
    upper(trim(term))                          as term_key,
    organ_system,
    num_affected,
    num_at_risk,
    case when num_at_risk > 0 then num_affected / num_at_risk end as arm_rate
from {{ source('silver', 'silver_arm_adverse_events') }}
