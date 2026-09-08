"""
generate_dataset.py
--------------------
Generates a simulated support-ticket dataset for the
"Support Ticket Analytics: What Drives Escalations and SLA Breaches?" project.

The data is synthetic but is built so that realistic causal relationships hold,
mirroring a real software support queue:

    channel  -->  first response time  -->  resolution time  -->  escalation  -->  SLA breach
    priority -->  SLA target
    customer tier --> handling speed
    agent experience --> resolution time & escalation risk
    category --> baseline complexity

Run:
    python generate_dataset.py

Output:
    support_tickets.csv  (one row per ticket)

Everything is seeded for reproducibility.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEED = 42
N_TICKETS = 6000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31, 23, 59)

rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# REFERENCE / LOOKUP DATA
# ----------------------------------------------------------------------------

CHANNELS = ["Phone", "Chat", "Email", "Portal"]
CHANNEL_WEIGHTS = [0.20, 0.30, 0.35, 0.15]

# Baseline first-response time in minutes, per channel (lognormal params: mean, sigma)
CHANNEL_RESPONSE_PARAMS = {
    "Phone":  (2.0, 0.35),   # fast, low variance (live conversation)
    "Chat":   (2.6, 0.45),
    "Email":  (4.0, 0.55),
    "Portal": (4.4, 0.65),   # slowest, most variance
}

CATEGORIES = [
    "Billing", "Technical Issue", "Account Access",
    "Bug Report", "Feature Request", "Onboarding",
]
CATEGORY_WEIGHTS = [0.18, 0.30, 0.16, 0.17, 0.09, 0.10]

# Baseline resolution complexity multiplier per category (relative to Billing = 1.0)
CATEGORY_COMPLEXITY = {
    "Billing": 1.0,
    "Technical Issue": 1.6,
    "Account Access": 0.8,
    "Bug Report": 2.3,
    "Feature Request": 2.6,
    "Onboarding": 1.1,
}

PRODUCT_AREAS = ["Mobile App", "Web Dashboard", "API", "Integrations", "Billing System"]
PRODUCT_AREA_WEIGHTS = [0.27, 0.28, 0.14, 0.13, 0.18]

PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_WEIGHTS = [0.32, 0.40, 0.21, 0.07]

# SLA targets in hours, by (priority, tier)
SLA_TARGET_HOURS = {
    "Free":       {"Low": 72, "Medium": 48, "High": 24, "Critical": 12},
    "Pro":        {"Low": 48, "Medium": 24, "High": 12, "Critical": 6},
    "Enterprise": {"Low": 24, "Medium": 12, "High": 6,  "Critical": 2},
}

TIERS = ["Free", "Pro", "Enterprise"]
TIER_WEIGHTS = [0.45, 0.40, 0.15]

# Resolution-time multiplier by tier (Enterprise gets prioritized handling)
TIER_SPEED_MULTIPLIER = {"Free": 1.15, "Pro": 1.0, "Enterprise": 0.65}

N_AGENTS = 24
agent_ids = [f"AG{str(i).zfill(3)}" for i in range(1, N_AGENTS + 1)]
# Agent tenure in months (skewed toward newer hires, a few veterans)
agent_experience = rng.gamma(shape=2.0, scale=9.0, size=N_AGENTS).clip(1, 60).round(0)
agent_experience_map = dict(zip(agent_ids, agent_experience))
# Each agent has a small individual "quality" offset (systematic fast/slow, in log-hours)
agent_quality_offset = dict(zip(agent_ids, rng.normal(0, 0.15, size=N_AGENTS)))

ESCALATION_REASONS = [
    "Customer requested supervisor",
    "SLA at risk",
    "Repeated contact / unresolved",
    "Technical complexity beyond tier-1 scope",
    "Billing dispute above agent authority",
    "Negative sentiment / churn risk",
]

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def random_business_weighted_timestamp(rng, start, end, n):
    """
    Sample n timestamps between start/end, weighted so that:
      - weekdays get far more volume than weekends
      - 9am-6pm gets more volume than off-hours (but not zero, support runs extended hours)
    """
    total_days = (end - start).days
    days_offset = rng.integers(0, total_days, size=n)

    # weekday/weekend weighting via rejection-free approach:
    # oversample days, then bias hour distribution
    dates = np.array([start + timedelta(days=int(d)) for d in days_offset])
    dow = np.array([d.weekday() for d in dates])  # 0=Mon ... 6=Sun

    # Re-roll weekend dates with 70% probability to push them toward weekdays
    weekend_mask = dow >= 5
    reroll_mask = weekend_mask & (rng.random(n) < 0.70)
    while reroll_mask.any():
        new_days_offset = rng.integers(0, total_days, size=reroll_mask.sum())
        new_dates = np.array([start + timedelta(days=int(d)) for d in new_days_offset])
        dates[reroll_mask] = new_dates
        dow = np.array([d.weekday() for d in dates])
        weekend_mask = dow >= 5
        reroll_mask = weekend_mask & (rng.random(n) < 0.70)

    # Hour of day: mixture of business-hours-heavy normal + small uniform tail (24/7 portal/email)
    business_hours = rng.normal(loc=13, scale=3.2, size=n).clip(0, 23.99)
    uniform_hours = rng.uniform(0, 24, size=n)
    use_uniform = rng.random(n) < 0.12
    hours = np.where(use_uniform, uniform_hours, business_hours)

    timestamps = [
        d + timedelta(hours=float(h))
        for d, h in zip(dates, hours)
    ]
    return pd.Series(pd.to_datetime(timestamps)).sort_values().reset_index(drop=True)


def weighted_choice(rng, options, weights, n):
    return rng.choice(options, size=n, p=weights)


# ----------------------------------------------------------------------------
# GENERATE CORE FIELDS
# ----------------------------------------------------------------------------

n = N_TICKETS
ticket_ids = [f"TCK-{100000 + i}" for i in range(n)]

created_at = random_business_weighted_timestamp(rng, START_DATE, END_DATE, n)

channel = weighted_choice(rng, CHANNELS, CHANNEL_WEIGHTS, n)
category = weighted_choice(rng, CATEGORIES, CATEGORY_WEIGHTS, n)
product_area = weighted_choice(rng, PRODUCT_AREAS, PRODUCT_AREA_WEIGHTS, n)
tier = weighted_choice(rng, TIERS, TIER_WEIGHTS, n)

# Priority correlated loosely with category (bugs/technical skew higher priority)
priority = np.empty(n, dtype=object)
for i in range(n):
    w = np.array(PRIORITY_WEIGHTS, dtype=float)
    if category[i] in ("Bug Report", "Technical Issue"):
        w = w * np.array([0.6, 0.9, 1.4, 1.8])
    elif category[i] == "Feature Request":
        w = w * np.array([1.6, 1.1, 0.4, 0.1])
    w = w / w.sum()
    priority[i] = rng.choice(PRIORITIES, p=w)

customer_id = [f"CUST-{rng.integers(1, 2400):05d}" for _ in range(n)]
agent_id = rng.choice(agent_ids, size=n)
agent_exp_months = np.array([agent_experience_map[a] for a in agent_id])
agent_offset = np.array([agent_quality_offset[a] for a in agent_id])

# ----------------------------------------------------------------------------
# FIRST RESPONSE TIME (minutes) — driven by channel, priority, time of day
# ----------------------------------------------------------------------------

response_minutes = np.empty(n)
for i in range(n):
    mu, sigma = CHANNEL_RESPONSE_PARAMS[channel[i]]
    base = rng.lognormal(mean=mu, sigma=sigma)

    # High/critical priority tickets get faster first response (routing/triage)
    priority_factor = {"Low": 1.25, "Medium": 1.0, "High": 0.7, "Critical": 0.45}[priority[i]]

    # Off-hours tickets (created outside 8am-8pm) wait longer for first response
    hour = created_at[i].hour
    off_hours_factor = 1.6 if (hour < 8 or hour > 20) else 1.0

    response_minutes[i] = base * priority_factor * off_hours_factor

response_minutes = np.round(response_minutes, 1)

# ----------------------------------------------------------------------------
# RESOLUTION TIME (hours) — driven by category complexity, tier, agent, response time
# ----------------------------------------------------------------------------

resolution_hours_base = np.empty(n)
for i in range(n):
    complexity = CATEGORY_COMPLEXITY[category[i]]
    tier_mult = TIER_SPEED_MULTIPLIER[tier[i]]

    # Agent experience reduces resolution time (diminishing returns), plus individual offset
    exp_factor = 1.0 / (1.0 + np.log1p(agent_exp_months[i]) * 0.12)

    # A slow first response is itself a signal of a harder/queue-backed-up ticket
    response_signal = 1.0 + (response_minutes[i] / 600.0)  # small nudge upward for slow responses

    log_mean = np.log(1.8 * complexity * tier_mult * exp_factor * response_signal) + agent_offset[i]
    resolution_hours_base[i] = rng.lognormal(mean=log_mean, sigma=0.75)

# Backlog spikes: a share of tickets get stuck (waiting on engineering, customer
# non-response, agent out of office, holiday backlog, etc.) causing a long tail.
# More likely for complex categories and lower-tier customers (less white-glove follow-up).
spike_prob = np.array([
    0.10 + (0.05 if CATEGORY_COMPLEXITY[category[i]] > 1.5 else 0.0)
         + (0.04 if tier[i] == "Free" else (0.0 if tier[i] == "Pro" else -0.05))
    for i in range(n)
])
spike_hit = rng.random(n) < spike_prob
spike_multiplier = np.where(spike_hit, rng.uniform(2.5, 9.0, size=n), 1.0)
resolution_hours_base = resolution_hours_base * spike_multiplier

resolution_hours_base = resolution_hours_base.clip(0.1, 400)

# ----------------------------------------------------------------------------
# ESCALATION — probability increases with: slow response, high priority,
# complex category, low agent experience, Free tier (less white-glove handling)
# ----------------------------------------------------------------------------

escalation_logit = (
    -2.6
    + 0.9 * (response_minutes > 30).astype(float)
    + 0.6 * (response_minutes > 90).astype(float)
    + np.select(
        [priority == "Low", priority == "Medium", priority == "High", priority == "Critical"],
        [-0.5, 0.0, 0.7, 1.3],
    )
    + np.select(
        [category == cat for cat in CATEGORIES],
        [0.5 if CATEGORY_COMPLEXITY[cat] > 1.5 else -0.2 for cat in CATEGORIES],
    )
    + 0.5 * (agent_exp_months < 6).astype(float)
    + 0.3 * (tier == "Free").astype(float)
    - 0.35 * (tier == "Enterprise").astype(float)
)
escalation_prob = 1 / (1 + np.exp(-escalation_logit))
escalated = rng.random(n) < escalation_prob

# Escalation adds handling time (hand-off, supervisor review, context transfer)
escalation_delay_hours = np.where(
    escalated,
    rng.gamma(shape=2.0, scale=2.2, size=n),
    0.0,
)
resolution_hours = resolution_hours_base + escalation_delay_hours
resolution_hours = np.round(resolution_hours, 2)

escalation_reason = np.array([
    rng.choice(ESCALATION_REASONS) if esc else "" for esc in escalated
])

# ----------------------------------------------------------------------------
# SLA TARGET & BREACH
# ----------------------------------------------------------------------------

sla_target_hours = np.array([
    SLA_TARGET_HOURS[tier[i]][priority[i]] for i in range(n)
])
sla_breached = resolution_hours > sla_target_hours

# ----------------------------------------------------------------------------
# REOPENED — more likely for escalated / breached / rushed resolutions
# ----------------------------------------------------------------------------

reopen_logit = (
    -3.0
    + 0.8 * escalated.astype(float)
    + 0.6 * sla_breached.astype(float)
    + 0.4 * (category == "Bug Report").astype(float)
)
reopen_prob = 1 / (1 + np.exp(-reopen_logit))
reopened = rng.random(n) < reopen_prob

# ----------------------------------------------------------------------------
# STATUS
# ----------------------------------------------------------------------------

# A small share of tickets are still open at "data pull" time (near the end of the year)
days_since_created = (END_DATE - created_at).dt.total_seconds() / 3600
still_open_prob = np.clip(1.2 - (days_since_created / (sla_target_hours * 3)), 0.0, 0.85)
is_open = rng.random(n) < still_open_prob

status = np.select(
    [is_open, escalated & ~is_open, ~is_open],
    ["Open", "Resolved (Escalated)", "Resolved"],
    default="Resolved",
)
# Where open, resolution metrics don't apply yet
resolution_hours = np.where(is_open, np.nan, resolution_hours)
sla_breached_final = np.where(is_open, False, sla_breached)  # breach flag only meaningful once resolved (open-and-overdue handled separately below)

# Tickets that are open AND already past SLA target are "at risk / breaching in progress"
sla_at_risk = is_open & (days_since_created > sla_target_hours)

# ----------------------------------------------------------------------------
# CSAT — only for resolved tickets; inversely related to resolution time & escalation
# ----------------------------------------------------------------------------

csat = np.full(n, np.nan)
for i in range(n):
    if is_open[i]:
        continue
    base_score = 4.3
    penalty = 0.0
    if escalated[i]:
        penalty += 0.6
    if sla_breached[i]:
        penalty += 0.9
    if reopened[i]:
        penalty += 0.5
    penalty += min(resolution_hours[i] / 100.0, 1.0) if not np.isnan(resolution_hours[i]) else 0
    score = base_score - penalty + rng.normal(0, 0.6)
    csat[i] = int(np.clip(round(score), 1, 5))

# CSAT surveys aren't always completed — simulate ~55% response rate
csat_responded = rng.random(n) < 0.55
csat = np.where((~is_open) & csat_responded, csat, np.nan)

# ----------------------------------------------------------------------------
# ASSEMBLE DATAFRAME
# ----------------------------------------------------------------------------

df = pd.DataFrame({
    "ticket_id": ticket_ids,
    "created_at": created_at,
    "channel": channel,
    "category": category,
    "product_area": product_area,
    "priority": priority,
    "customer_id": customer_id,
    "customer_tier": tier,
    "agent_id": agent_id,
    "agent_tenure_months": agent_exp_months.astype(int),
    "first_response_minutes": response_minutes,
    "resolution_hours": resolution_hours,
    "sla_target_hours": sla_target_hours,
    "sla_breached": np.where(is_open, sla_at_risk, sla_breached),
    "escalated": escalated,
    "escalation_reason": escalation_reason,
    "reopened": np.where(is_open, False, reopened),
    "status": status,
    "csat_score": csat,
})

df = df.sort_values("created_at").reset_index(drop=True)

# Light realistic messiness: a few missing agent tenure / product area values,
# a handful of duplicate-looking (but distinct) customer contacts, minor typos in reason field
missing_idx = rng.choice(df.index, size=int(n * 0.01), replace=False)
df.loc[missing_idx, "product_area"] = np.nan

df.to_csv("support_tickets.csv", index=False)

print(f"Generated {len(df)} tickets -> support_tickets.csv")
print(df.head(10).to_string())
print("\n--- Sanity checks ---")
print("Escalation rate:", round(df["escalated"].mean(), 3))
print("SLA breach rate:", round(df["sla_breached"].mean(), 3))
print("Escalation rate | SLA breached:", round(df.loc[df["sla_breached"], "escalated"].mean(), 3))
print("Escalation rate | not breached:", round(df.loc[~df["sla_breached"], "escalated"].mean(), 3))
print("Mean resolution hrs by channel:\n", df.groupby("channel")["resolution_hours"].mean())
print("Status counts:\n", df["status"].value_counts())
