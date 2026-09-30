import os
import numpy as np
import pandas as pd


def generate_intern_telemetry(n_samples: int = 600, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic intern operational telemetry and saves to data/intern_data.csv.
    Features:
      - task_completion_rate: % of assigned tasks completed (Beta distributed)
      - avg_turnaround_days: Average time taken per task in days (Gamma distributed)
      - attendance_rate: % of meeting/portal attendance (Beta distributed)
      - mentor_feedback_rating: Score from 1.0 to 5.0 (Normally distributed)
      - peer_collaboration_score: Score from 1.0 to 5.0 (Normally distributed)
    Target:
      - performance_score: Continuous score (0 - 100) reflecting weighted performance.
    """
    np.random.seed(random_seed)

    intern_ids = [f"INT-{1000 + i}" for i in range(n_samples)]

    # Feature distributions
    task_completion_rate = np.random.beta(a=5, b=2, size=n_samples) * 100
    task_completion_rate = np.clip(task_completion_rate, 25.0, 100.0)

    avg_turnaround_days = np.random.gamma(shape=3.0, scale=1.2, size=n_samples)
    avg_turnaround_days = np.clip(avg_turnaround_days, 1.0, 14.0)

    attendance_rate = np.random.beta(a=6, b=2, size=n_samples) * 100
    attendance_rate = np.clip(attendance_rate, 30.0, 100.0)

    mentor_feedback_rating = np.random.normal(loc=3.8, scale=0.8, size=n_samples)
    mentor_feedback_rating = np.clip(mentor_feedback_rating, 1.0, 5.0)

    peer_collaboration_score = np.random.normal(loc=3.5, scale=0.9, size=n_samples)
    peer_collaboration_score = np.clip(peer_collaboration_score, 1.0, 5.0)

    # Domain ground-truth equation with random Gaussian noise
    noise = np.random.normal(0, 3.0, n_samples)
    performance_score = (
        0.35 * task_completion_rate
        + 0.25 * attendance_rate
        + 0.20 * (mentor_feedback_rating * 20.0)
        + 0.10 * (peer_collaboration_score * 20.0)
        - 1.50 * avg_turnaround_days
        + noise
    )
    performance_score = np.clip(performance_score, 0.0, 100.0)

    df = pd.DataFrame(
        {
            "intern_id": intern_ids,
            "task_completion_rate": np.round(task_completion_rate, 2),
            "avg_turnaround_days": np.round(avg_turnaround_days, 2),
            "attendance_rate": np.round(attendance_rate, 2),
            "mentor_feedback_rating": np.round(mentor_feedback_rating, 2),
            "peer_collaboration_score": np.round(peer_collaboration_score, 2),
            "performance_score": np.round(performance_score, 2),
        }
    )

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "intern_data.csv")
    df.to_csv(csv_path, index=False)

    print(f"[OK] Generated {n_samples} rows -> {csv_path}")
    return df


if __name__ == "__main__":
    generate_intern_telemetry()