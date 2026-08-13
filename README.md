[README.md](https://github.com/user-attachments/files/31021606/README.md)
# Support Ticket Analytics: What Drives Escalations and SLA Breaches?

A support-operations analytics project answering three questions a real helpdesk team
cares about: what does the ticket queue look like, where do escalations and SLA breaches
concentrate, and can we flag at-risk tickets early enough to intervene.

## Why this project

I currently work as a technical support analyst and wanted a portfolio project grounded
in a domain I actually understand, rather than a generic public dataset. To protect
employer confidentiality, the dataset here is **simulated** — but it's built to mirror the
structure and relationships (channel → response time → resolution time → escalation) of a
real software support queue.

## What's inside

| File | Description |
|---|---|
| `support_ticket_analysis.ipynb` | Full analysis: EDA, SLA/escalation breakdowns, logistic regression risk model, business recommendations |
| `generate_data.py` | Script that simulates the 4,000-ticket dataset (fully reproducible, seeded) |
| `support_tickets.csv` | The generated dataset |
| `chart_*.png` | Exported chart images used in the write-up |

## Approach

1. **Descriptive analysis** — ticket volume trends, category mix, and where SLA breaches
   and escalations concentrate by category, tier, and priority.
2. **Hypothesis check** — does faster first-response time actually reduce escalations?
   (Checked against the data rather than assumed.)
3. **Predictive model** — a logistic regression trained only on information available at
   ticket intake (no leakage from resolution time), used to flag high-risk tickets early.
   Deliberately kept interpretable over more complex models, since a support team needs to
   trust *why* a ticket was flagged.
4. **Business write-up** — findings translated into concrete, actionable recommendations,
   not just charts.

## Key result

The escalation-risk model reaches an **ROC-AUC of ~0.69** using only category, channel,
customer tier, priority, and first-response time — with `Bug/Error`, `Integration/API`,
and `Performance/Slowness` categories, high priority, and slow first response emerging as
the strongest positive predictors of escalation, and Enterprise tier / fast channels as
protective factors.

## Tools

Python, pandas, scikit-learn, matplotlib.

## Limitations

This uses simulated, not production, data — see the notebook's final section for a fuller
discussion of what would need validating against real ticket data.
