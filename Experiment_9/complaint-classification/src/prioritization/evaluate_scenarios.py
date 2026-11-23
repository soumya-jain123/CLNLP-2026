"""
Synthetic Scenario Evaluation Pipeline for Phase 6.

Evaluates category routing correctness, priority tier consistency, escalation
consistency, indicator precision/recall/FPR/FNR, and explainability coverage
across 28 diverse synthetic scenarios spanning all 7 complaint categories.

Note: Synthetic scenario evaluation measures rule/integration behavior on controlled
scenarios and does NOT substitute for real-world clinical/operational validation.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prioritization.complaint_analyzer import ComplaintAnalyzer

sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300


SYNTHETIC_SCENARIOS: List[Dict[str, Any]] = [
    # --- bank_account ---
    {
        "id": "S01",
        "text": "My checking account at the bank was frozen without notice and I cannot access my funds of $4,200. I called multiple times with no response and will hire an attorney.",
        "expected_category": "bank_account",
        "expected_department": "Banking Operations",
        "expected_escalation": True,
        "expected_tiers": ["P1", "P2"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": True, "legal_action_mentioned": True, "repeated_contact": True, "monetary_amount_mentioned": True},
    },
    {
        "id": "S02",
        "text": "I would like to update the mailing address on my savings account and order new checks for my checking account.",
        "expected_category": "bank_account",
        "expected_department": "Banking Operations",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S03",
        "text": "Someone made an unauthorized debit card withdrawal of $850 from my checking account. This is fraud and I filed a complaint with the CFPB.",
        "expected_category": "bank_account",
        "expected_department": "Banking Operations",
        "expected_escalation": True,
        "expected_tiers": ["P2", "P3"],
        "gt_indicators": {"unauthorized_transaction": True, "fraud_related": True, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": True},
    },
    {
        "id": "S04",
        "text": "There was no fraud and no unauthorized transaction on my checking account, I simply want a copy of my bank statement.",
        "expected_category": "bank_account",
        "expected_department": "Banking Operations",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    # --- credit_card ---
    {
        "id": "S05",
        "text": "I discovered $2,300 in unauthorized charges on my credit card from a stolen card. I contacted customer service several times with no reply and am facing collections notice.",
        "expected_category": "credit_card",
        "expected_department": "Credit Card Services",
        "expected_escalation": True,
        "expected_tiers": ["P1", "P2"],
        "gt_indicators": {"unauthorized_transaction": True, "fraud_related": True, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": True, "monetary_amount_mentioned": True},
    },
    {
        "id": "S06",
        "text": "My credit card annual fee of $95 posted this month and I want to ask if rewards points can cover it.",
        "expected_category": "credit_card",
        "expected_department": "Credit Card Services",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": True},
    },
    {
        "id": "S07",
        "text": "Credit card company ignored my formal dispute letter for months without resolution and charged fraudulent interest.",
        "expected_category": "credit_card",
        "expected_department": "Credit Card Services",
        "expected_escalation": True,
        "expected_tiers": ["P2", "P3"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": True, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S08",
        "text": "I did not experience any fraud on my credit card; I just need to change my billing due date.",
        "expected_category": "credit_card",
        "expected_department": "Credit Card Services",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    # --- credit_reporting ---
    {
        "id": "S09",
        "text": "Experian and Equifax are reporting an inaccurate account on my credit report from identity theft. I sent a formal dispute under FCRA and will sue in court.",
        "expected_category": "credit_reporting",
        "expected_department": "Credit Reporting & Disputes",
        "expected_escalation": True,
        "expected_tiers": ["P2", "P3"],
        "gt_indicators": {"unauthorized_transaction": True, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": True, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S10",
        "text": "Please remove an old paid inquiry from my TransUnion credit report so my score updates cleanly.",
        "expected_category": "credit_reporting",
        "expected_department": "Credit Reporting & Disputes",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S11",
        "text": "XXXX XXXX XXXX XXXX XXXX XXXX inaccurate credit report inquiry on XX/XX/2023.",
        "expected_category": "credit_reporting",
        "expected_department": "Credit Reporting & Disputes",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S12",
        "text": "Credit bureau failed to fix my credit report after repeated complaints and my CFPB regulatory complaint.",
        "expected_category": "credit_reporting",
        "expected_department": "Credit Reporting & Disputes",
        "expected_escalation": True,
        "expected_tiers": ["P3", "P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": True, "monetary_amount_mentioned": False},
    },
    # --- debt_collection ---
    {
        "id": "S13",
        "text": "A debt collector is threatening wage garnishment for $1,900 I do not owe and refused to provide validation of debt. I am contacting an attorney.",
        "expected_category": "debt_collection",
        "expected_department": "Collections Resolution",
        "expected_escalation": True,
        "expected_tiers": ["P2", "P3"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": True, "repeated_contact": False, "monetary_amount_mentioned": True},
    },
    {
        "id": "S14",
        "text": "A collection agency called me once regarding a medical bill and I requested a written statement.",
        "expected_category": "debt_collection",
        "expected_department": "Collections Resolution",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S15",
        "text": "Debt collector keeps calling repeatedly about a settled debt of $600 and ignored my cease letter. I filed an FTC complaint.",
        "expected_category": "debt_collection",
        "expected_department": "Collections Resolution",
        "expected_escalation": True,
        "expected_tiers": ["P3", "P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": True, "monetary_amount_mentioned": True},
    },
    {
        "id": "S16",
        "text": "The debt collector did not threaten a lawsuit and I did not hire a lawyer, just checking balance.",
        "expected_category": "debt_collection",
        "expected_department": "Collections Resolution",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    # --- money_transfer ---
    {
        "id": "S17",
        "text": "My PayPal account is locked and $5,000 in funds are frozen after a wire transfer failed. This is urgent because I cannot pay rent and face eviction.",
        "expected_category": "money_transfer",
        "expected_department": "Payments & Transfers",
        "expected_escalation": True,
        "expected_tiers": ["P1", "P2"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": True, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": True},
    },
    {
        "id": "S18",
        "text": "I was scammed on Zelle for $1,400 through an unauthorized transfer without my permission. I called multiple times with no response.",
        "expected_category": "money_transfer",
        "expected_department": "Payments & Transfers",
        "expected_escalation": True,
        "expected_tiers": ["P2", "P3"],
        "gt_indicators": {"unauthorized_transaction": True, "fraud_related": True, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": True, "monetary_amount_mentioned": True},
    },
    {
        "id": "S19",
        "text": "Can you tell me how long a standard Venmo transfer takes to settle into a wallet?",
        "expected_category": "money_transfer",
        "expected_department": "Payments & Transfers",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S20",
        "text": "Money transfer stuck.",
        "expected_category": "money_transfer",
        "expected_department": "Payments & Transfers",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    # --- mortgage ---
    {
        "id": "S21",
        "text": "My mortgage servicer misapplied $3,100 of my escrow payment and issued a foreclosure notice due tomorrow. I contacted them numerous times with no response and retained a lawyer.",
        "expected_category": "mortgage",
        "expected_department": "Mortgage Services",
        "expected_escalation": True,
        "expected_tiers": ["P1", "P2"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": True, "repeated_contact": True, "monetary_amount_mentioned": True},
    },
    {
        "id": "S22",
        "text": "I would like an updated escrow analysis and amortization schedule for my mortgage loan.",
        "expected_category": "mortgage",
        "expected_department": "Mortgage Services",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S23",
        "text": "My mortgage modification has been unresolved for months without reply and I filed a complaint with the state regulator.",
        "expected_category": "mortgage",
        "expected_department": "Mortgage Services",
        "expected_escalation": True,
        "expected_tiers": ["P3", "P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": True, "monetary_amount_mentioned": False},
    },
    {
        "id": "S24",
        "text": "Mortgage loan modification was approved without foreclosure or legal action.",
        "expected_category": "mortgage",
        "expected_department": "Mortgage Services",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    # --- student_loan ---
    {
        "id": "S25",
        "text": "MOHELA double charged my student loan payment of $1,100 and threatened wage garnishment despite my repeated calls. I am filing a CFPB complaint.",
        "expected_category": "student_loan",
        "expected_department": "Student Loan Servicing",
        "expected_escalation": True,
        "expected_tiers": ["P2", "P3"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": True, "monetary_amount_mentioned": True},
    },
    {
        "id": "S26",
        "text": "I need help checking my income driven repayment recertification date with Nelnet for my student loans.",
        "expected_category": "student_loan",
        "expected_department": "Student Loan Servicing",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S27",
        "text": "My student loan servicer delayed resolution for weeks without response after my credit card and checking account autopay failed.",
        "expected_category": "student_loan",
        "expected_department": "Student Loan Servicing",
        "expected_escalation": False,
        "expected_tiers": ["P3", "P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
    {
        "id": "S28",
        "text": "My student loan balance is zero and I just need a payoff confirmation letter.",
        "expected_category": "student_loan",
        "expected_department": "Student Loan Servicing",
        "expected_escalation": False,
        "expected_tiers": ["P4"],
        "gt_indicators": {"unauthorized_transaction": False, "fraud_related": False, "account_access_issue": False, "legal_action_mentioned": False, "repeated_contact": False, "monetary_amount_mentioned": False},
    },
]


def run_synthetic_evaluation(project_root: Path = _PROJECT_ROOT) -> Dict[str, Any]:
    reports_dir = project_root / "reports" / "prioritization"
    figures_dir = project_root / "reports" / "figures" / "prioritization"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ComplaintAnalyzer(
        project_root=project_root,
        audit_log_path=reports_dir / "synthetic_audit_log.jsonl",
    )

    records = []
    ind_keys = [
        "unauthorized_transaction",
        "fraud_related",
        "account_access_issue",
        "legal_action_mentioned",
        "repeated_contact",
        "monetary_amount_mentioned",
    ]
    ind_counts = {k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for k in ind_keys}

    for sc in SYNTHETIC_SCENARIOS:
        res = analyzer.analyze(sc["text"], complaint_id=sc["id"], use_transformer=False)
        res_ind = res.indicators.model_dump()

        for k in ind_keys:
            gt = sc["gt_indicators"][k]
            pr = bool(res_ind[k])
            if gt and pr:
                ind_counts[k]["tp"] += 1
            elif not gt and pr:
                ind_counts[k]["fp"] += 1
            elif gt and not pr:
                ind_counts[k]["fn"] += 1
            else:
                ind_counts[k]["tn"] += 1

        records.append({
            "scenario_id": sc["id"],
            "expected_category": sc["expected_category"],
            "predicted_category": res.predicted_category,
            "category_correct": res.predicted_category == sc["expected_category"],
            "expected_department": sc["expected_department"],
            "predicted_department": res.department,
            "routing_correct": res.department == sc["expected_department"],
            "secondary_department": res.secondary_department or "None",
            "classification_confidence": res.classification_confidence,
            "priority_score": res.priority_score,
            "priority_tier": res.priority_tier,
            "expected_tiers": "|".join(sc["expected_tiers"]),
            "priority_consistent": res.priority_tier in sc["expected_tiers"],
            "expected_escalation": sc["expected_escalation"],
            "escalation_recommended": res.escalation_recommended,
            "escalation_consistent": res.escalation_recommended == sc["expected_escalation"],
            "escalation_level": res.escalation_level,
            "matched_rule_ids": "|".join(res.matched_rule_ids) if res.matched_rule_ids else "None",
            "sla_target": res.sla_target,
            "review_required": res.review_required,
            "uncertainty_flags": "|".join(res.uncertainty_flags) if res.uncertainty_flags else "None",
            "num_reasons": len(res.reasons),
            "processing_time_ms": res.processing_time_ms,
        })

    df = pd.DataFrame(records)
    df.to_csv(reports_dir / "synthetic_scenario_results.csv", index=False)

    # Indicator Evaluation CSV
    ind_rows = []
    for k, c in ind_counts.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        ind_rows.append({
            "indicator": k, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4),
            "FPR": round(fpr, 4), "FNR": round(fnr, 4),
        })
    ind_df = pd.DataFrame(ind_rows)
    ind_df.to_csv(reports_dir / "indicator_evaluation.csv", index=False)

    # Routing, Priority, Escalation Evaluation CSVs
    routing_eval = df.groupby("expected_category").agg(
        Total=("scenario_id", "count"),
        Correct_Routing=("routing_correct", "sum"),
    ).reset_index()
    routing_eval["Routing_Accuracy"] = (routing_eval["Correct_Routing"] / routing_eval["Total"]).round(4)
    routing_eval.to_csv(reports_dir / "routing_evaluation.csv", index=False)

    priority_eval = df.groupby("priority_tier").agg(
        Count=("scenario_id", "count"),
        Mean_Score=("priority_score", "mean"),
        Consistent_Count=("priority_consistent", "sum"),
    ).reset_index()
    priority_eval.to_csv(reports_dir / "priority_evaluation.csv", index=False)

    esc_eval = pd.DataFrame([{
        "Total_Scenarios": len(df),
        "Expected_Escalation_True": int(df["expected_escalation"].sum()),
        "Predicted_Escalation_True": int(df["escalation_recommended"].sum()),
        "Escalation_Consistency_Rate": round(float(df["escalation_consistent"].mean()), 4),
    }])
    esc_eval.to_csv(reports_dir / "escalation_evaluation.csv", index=False)

    # --- Generate 7 Figures ---
    # 1. Priority tier distribution
    plt.figure(figsize=(7, 4.2))
    tier_order = ["P1", "P2", "P3", "P4"]
    sns.countplot(data=df, x="priority_tier", order=tier_order, palette="Set2")
    plt.title("Synthetic Scenarios: Priority Tier Distribution")
    plt.xlabel("Priority Tier")
    plt.ylabel("Number of Complaints")
    plt.bar_label(plt.gca().containers[0], padding=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "01_priority_tier_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Indicator distribution
    plt.figure(figsize=(8.5, 4.5))
    ind_counts_plot = pd.DataFrame([
        {"Indicator": r["indicator"], "Detected Count": r["TP"] + r["FP"]} for r in ind_rows
    ]).sort_values("Detected Count", ascending=False)
    sns.barplot(data=ind_counts_plot, y="Indicator", x="Detected Count", palette="Set2")
    plt.title("Synthetic Scenarios: Extracted Indicator Counts")
    plt.bar_label(plt.gca().containers[0], padding=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "02_indicator_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Category-to-department routing
    plt.figure(figsize=(9, 4.8))
    ct = pd.crosstab(df["predicted_category"], df["predicted_department"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Synthetic Scenarios: Category-to-Department Routing Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "03_category_to_department_routing.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Synthetic scenario evaluation summary
    plt.figure(figsize=(8, 4.2))
    eval_bars = pd.DataFrame({
        "Dimension": ["Category Routing", "Priority Consistency", "Escalation Consistency", "Explainability Coverage"],
        "Accuracy / Coverage %": [
            df["routing_correct"].mean() * 100,
            df["priority_consistent"].mean() * 100,
            df["escalation_consistent"].mean() * 100,
            (df["num_reasons"] >= 3).mean() * 100,
        ],
    })
    sns.barplot(data=eval_bars, y="Dimension", x="Accuracy / Coverage %", palette="Paired")
    plt.xlim(0, 110)
    plt.title("Phase 6 Synthetic Scenario Evaluation Metrics (%)")
    plt.bar_label(plt.gca().containers[0], fmt="%.1f%%", padding=3)
    plt.tight_layout()
    plt.savefig(figures_dir / "04_synthetic_scenario_evaluation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Escalation rule coverage
    rule_list = [r for rules in df["matched_rule_ids"] for r in rules.split("|") if r != "None"]
    plt.figure(figsize=(8.5, 4.2))
    if rule_list:
        rc_df = pd.Series(rule_list).value_counts().reset_index()
        rc_df.columns = ["Rule ID", "Triggered Count"]
        sns.barplot(data=rc_df, y="Rule ID", x="Triggered Count", palette="Set2")
        plt.bar_label(plt.gca().containers[0], padding=2)
    plt.title("Synthetic Scenarios: Escalation Rule Coverage")
    plt.tight_layout()
    plt.savefig(figures_dir / "05_escalation_rule_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Priority score distribution
    plt.figure(figsize=(8, 4.2))
    sns.histplot(df["priority_score"], bins=10, kde=True, color="#4C72B0")
    plt.axvline(40, color="green", linestyle="--", label="P3 (40)")
    plt.axvline(60, color="orange", linestyle="--", label="P2 (60)")
    plt.axvline(80, color="red", linestyle="--", label="P1 (80)")
    plt.title("Synthetic Scenarios: Priority Score Distribution (0–100)")
    plt.xlabel("Priority Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "06_priority_score_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Explainability coverage (reasons count per scenario)
    plt.figure(figsize=(7.5, 4.2))
    sns.countplot(data=df, x="num_reasons", palette="Set2")
    plt.title("Explainability Coverage: Evidence-Backed Reasons per Scenario")
    plt.xlabel("Number of Generated Explanation Reasons")
    plt.ylabel("Scenarios")
    plt.bar_label(plt.gca().containers[0], padding=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "07_explainability_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary = {
        "total_scenarios": int(len(df)),
        "category_accuracy": round(float(df["category_correct"].mean()), 4),
        "routing_accuracy": round(float(df["routing_correct"].mean()), 4),
        "priority_consistency": round(float(df["priority_consistent"].mean()), 4),
        "escalation_consistency": round(float(df["escalation_consistent"].mean()), 4),
        "explainability_coverage": round(float((df["num_reasons"] >= 3).mean()), 4),
        "mean_processing_ms": round(float(df["processing_time_ms"].mean()), 2),
    }
    return summary


if __name__ == "__main__":
    summary = run_synthetic_evaluation()
    print("Synthetic Evaluation Summary:", json.dumps(summary, indent=2))
