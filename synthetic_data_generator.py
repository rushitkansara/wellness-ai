"""
wellness_data_generator.py

Generate synthetic wellness datasets with profile-based daily simulations.
Features include demographics, activity, environment, vitals, mood & stress
computed from continuous equations and a discrete-time coupling.

Author: ChatGPT (professional template)
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import math
import os

# ----------------------------
# Configuration / Defaults
# ----------------------------
DEFAULTS = {
    "n_profiles": 10,
    "days_per_profile": 14,
    "seed": 42,
    "smoker_pct": 0.2,
    "sedentary_pct": 0.4,   # fraction of profiles that will have high sedentary behavior
    "sedentary_hours_threshold": 10.0,  # hours sitting considered high
    "sex_filter": "even",   # 'even'|'male'|'female'|'custom'
    "age_min": 20,
    "age_max": 60,
    "output_csv": "synthetic_wellness.csv",
    "missing_rate": 0.03
}

# ----------------------------
# Utility functions
# ----------------------------
def clip01(x):
    return max(0.0, min(1.0, x))

def normalize_log(x, scale=1.0):
    # log normalization for positive quantities
    return math.log1p(x) / math.log1p(scale) if scale > 0 else 0.0

# ----------------------------
# Profile generation
# ----------------------------
def create_profiles(n_profiles,
                    sex_filter="even",
                    age_min=20, age_max=60,
                    smoker_pct=0.2,
                    sedentary_pct=0.4,
                    seed=42):
    """
    Returns a DataFrame with profile-level attributes:
      user_id, age, sex, height_cm, weight_kg, bmi, activity_level(1-3),
      smoker(bool), sedentary(bool), chronic_factor (0.8-1.3)
    sex_filter: 'even'|'male'|'female'|'custom' (custom not implemented here)
    """
    rng = np.random.default_rng(seed)
    profiles = []
    sexes = []
    if sex_filter == "even":
        sexes = ["male","female"] * (n_profiles//2 + 1)
    elif sex_filter == "male":
        sexes = ["male"] * n_profiles
    elif sex_filter == "female":
        sexes = ["female"] * n_profiles
    else:
        sexes = ["male","female"] * (n_profiles//2 + 1)
    sexes = sexes[:n_profiles]

    # sample ages uniformly in range but allow optional clustering later
    ages = rng.integers(age_min, age_max+1, size=n_profiles)

    # heights and weights by sex (rough realistic distributions)
    heights = np.zeros(n_profiles, dtype=float)
    weights = np.zeros(n_profiles, dtype=float)
    for i, s in enumerate(sexes):
        if s == "male":
            heights[i] = rng.normal(175, 7)  # cm
            weights[i] = rng.normal(78, 12)  # kg
        else:
            heights[i] = rng.normal(162, 7)
            weights[i] = rng.normal(65, 10)
    heights = np.clip(heights, 140, 210)
    weights = np.clip(weights, 40, 150)
    bmi = weights / ((heights/100.0)**2)

    # activity_level: 1 sedentary, 2 moderate, 3 active (profile baseline)
    activity_level = rng.choice([1,2,3], size=n_profiles, p=[0.3,0.5,0.2])

    # smokers selection (choose indices)
    n_smokers = int(round(smoker_pct * n_profiles))
    smoker_idx = rng.choice(n_profiles, size=n_smokers, replace=False) if n_smokers>0 else []

    # sedentary selection
    n_sedentary = int(round(sedentary_pct * n_profiles))
    sedentary_idx = rng.choice(n_profiles, size=n_sedentary, replace=False) if n_sedentary>0 else []

    # chronic predisposition factor: multiplicative factor for baseline stress susceptibility
    chronic_factor = rng.normal(1.0, 0.12, size=n_profiles)
    chronic_factor = np.clip(chronic_factor, 0.7, 1.4)

    for i in range(n_profiles):
        profiles.append({
            "user_id": f"user_{i}",
            "age": int(ages[i]),
            "sex": sexes[i],
            "height_cm": float(round(heights[i],1)),
            "weight_kg": float(round(weights[i],1)),
            "bmi": float(round(bmi[i],2)),
            "activity_level": int(activity_level[i]),
            "smoker": bool(i in smoker_idx),
            "sedentary_profile": bool(i in sedentary_idx),
            "chronic_factor": float(round(chronic_factor[i],3))
        })
    return pd.DataFrame(profiles)


# ----------------------------
# Core equations (math & assumptions)
# ----------------------------
# We'll implement discrete-time equations. Time-step = 1 day.
# stress(t) is base_stress(profile) + acute_env_effects(daily features)
# mood(t) is base_mood(profile) + benefits(activity, sleep, social) - coupling*stress(t) + noise
#
# Base profile modifiers:
#   - Older age and higher BMI increase baseline stress susceptibility
#   - Active profile reduces baseline stress
# Equations use log/exponential transforms so extreme values have diminishing returns.

def baseline_modifiers(profile_row):
    """Compute baseline mood/stress offset for a profile (constant)."""
    age = profile_row['age']
    bmi = profile_row['bmi']
    activity_lvl = profile_row['activity_level']
    chronic = profile_row['chronic_factor']
    # baseline stress offset (higher for older/high BMI)
    stress_base = 3.0 + 0.02*(age-30) + 0.4 * ((bmi-22)/5.0)  # center at age 30 and BMI 22
    # activity reduces baseline stress
    stress_base -= 0.5*(activity_lvl-2)   # -0.5 for active, +0.5 for sedentary
    stress_base *= chronic
    # baseline mood (higher with activity, lower with age/BMI)
    mood_base = 6.5 + 0.1*(activity_lvl-2) - 0.03*(age-30) - 0.15*((bmi-22)/5.0)
    mood_base = float(np.clip(mood_base, 1.0, 9.0))
    return stress_base, mood_base

def daily_env_effects(day_features, profile_row):
    """
    day_features: dict with keys:
      - steps, exercise_min, exercise_intensity (0-1), sleep_hours,
      - water_liters, calories, caffeine_mg, sunlight_minutes,
      - aqi, noise_db, lighting_lux, time_outdoors, time_in_office,
      - social_minutes, sedentary_hours
    returns: stress_delta, mood_delta, also derived vitals modifications
    """
    # unpack with defaults
    steps = float(day_features.get('steps', 0.0))
    exercise_min = float(day_features.get('exercise_min', 0.0))
    exercise_intensity = float(day_features.get('exercise_intensity', 0.0))
    sleep_hours = float(day_features.get('sleep_hours', 7.0))
    water_l = float(day_features.get('water_liters', 2.0))
    calories = float(day_features.get('calories', 2200))
    caffeine = float(day_features.get('caffeine_mg', 0))
    sunlight = float(day_features.get('sunlight_minutes', 60.0))
    aqi = float(day_features.get('aqi', 100))
    noise = float(day_features.get('noise_db', 40))
    lighting = float(day_features.get('lighting_lux', 300))
    time_outdoors = float(day_features.get('time_outdoors', 60))
    time_in_office = float(day_features.get('time_in_office', 8.0))
    social_minutes = float(day_features.get('social_minutes', 30))
    sedentary_hours = float(day_features.get('sedentary_hours', 8.0))
    smoker = bool(profile_row.get('smoker', False))

    # --- Transformations / Normalizations (nonlinear) ---
    # Use log1p to reflect diminishing returns for activity/sunlight
    steps_score = math.log1p(steps) / math.log1p(10000)  # 0..~1
    exercise_score = math.log1p(exercise_min*exercise_intensity) / math.log1p(120*1.0)
    sleep_score = clip01((sleep_hours - 4.0) / 6.0)   # map 4-10h to 0..1
    water_score = clip01((water_l - 0.5) / 3.5)       # map 0.5-4.0L
    sunlight_score = math.log1p(sunlight) / math.log1p(180)
    social_score = clip01(social_minutes/120.0)       # up to 2 hours -> 1
    sedentary_penalty = clip01((sedentary_hours - 4.0) / 12.0)  # >4 hours starts penalty
    aqi_penalty = clip01(max(0.0, (aqi - 50.0))/200.0)  # 50-250 -> 0..1
    noise_penalty = clip01((noise - 30.0)/70.0)
    caffeine_effect = clip01(caffeine / 400.0)
    # smoker factor adds baseline vulnerability and acute penalty
    smoker_penalty = 0.15 if smoker else 0.0

    # --- Stress delta: positive increases stress ---
    # Lowered by activity, sleep, water, social. Increased by aqi, noise, caffeine, sedentary, office time
    stress_delta = 0.0
    stress_delta += -1.2 * steps_score
    stress_delta += -0.9 * exercise_score
    stress_delta += -1.5 * sleep_score
    stress_delta += -0.8 * water_score
    stress_delta += -0.6 * social_score
    stress_delta += +1.6 * aqi_penalty
    stress_delta += +1.1 * noise_penalty
    stress_delta += +0.9 * caffeine_effect
    stress_delta += +0.6 * sedentary_penalty
    stress_delta += +0.6 * clip01((time_in_office - 6.0)/12.0)
    stress_delta += +0.9 * smoker_penalty
    # mild calorie effect: extreme high or low increases stress
    if calories < 1400:
        stress_delta += 0.6
    elif calories > 3200:
        stress_delta += 0.4

    # --- Mood delta: positive increases mood ---
    mood_delta = 0.0
    mood_delta += +1.1 * steps_score
    mood_delta += +1.4 * exercise_score
    mood_delta += +1.5 * sleep_score
    mood_delta += +0.9 * sunlight_score
    mood_delta += +0.8 * water_score
    mood_delta += +0.8 * social_score
    mood_delta += -1.2 * aqi_penalty
    mood_delta += -0.9 * noise_penalty
    mood_delta += -0.8 * caffeine_effect  # assume caffeine has slight negative on mood net
    mood_delta += -1.0 * sedentary_penalty
    mood_delta += -0.7 * clip01((time_in_office - 6.0)/12.0)
    mood_delta += -0.6 * smoker_penalty
    # calorie effect: moderate calories positive, extremes negative
    if 1800 <= calories <= 2800:
        mood_delta += 0.4
    else:
        mood_delta -= 0.3

    # lighting: very low lighting reduces mood slightly
    if lighting < 150:
        mood_delta -= 0.3

    # return deltas
    return float(stress_delta), float(mood_delta), {
        "steps_score": steps_score,
        "exercise_score": exercise_score,
        "sleep_score": sleep_score,
        "water_score": water_score,
        "sunlight_score": sunlight_score,
        "aqi_penalty": aqi_penalty,
        "noise_penalty": noise_penalty,
        "caffeine_effect": caffeine_effect
    }

# ----------------------------
# Simulation engine: discrete-time update for each day
# ----------------------------
def simulate_profile_days(profile_row, days=14, rng=None, missing_rate=0.03):
    """
    Simulates 'days' of data for a given profile (one user).
    Returns a list of daily dicts with features + mood/stress/vitals/confidence.
    """
    if rng is None:
        rng = np.random.default_rng()

    stress_base, mood_base = baseline_modifiers(profile_row)
    user_id = profile_row['user_id']
    rows = []
    # initialize previous values for simple temporal coupling
    prev_stress = stress_base
    prev_mood = mood_base

    for day in range(days):
        # SAMPLE DAILY INPUTS (realistic distributions conditioned on profile)
        # Steps depend on activity_level & sedentary_profile
        act = profile_row['activity_level']
        sedentary = profile_row['sedentary_profile']
        steps_mean = 3500 + (act-1)*2500  # 1 -> 3500, 2 -> 6000, 3 -> 8500
        if sedentary:
            steps_mean *= 0.55

        steps = float(max(0, rng.normal(steps_mean, steps_mean*0.35)))
        exercise_min = float(max(0.0, rng.normal(30 if act>=2 else 10, 20)))
        exercise_intensity = float(clip01(rng.normal(0.4 if act==2 else 0.2 if act==1 else 0.7, 0.2)))
        sleep_hours = float(rng.normal(7.0 - 0.2*(profile_row['age']-30)/30.0, 1.2))
        water_l = float(max(0.2, rng.normal(2.0, 0.6)))
        calories = float(max(1200, rng.normal(2200, 450)))
        caffeine_mg = float(0 if rng.random() < 0.3 else rng.normal(95,20) * max(0, rng.poisson(1)))
        sunlight_minutes = float(max(0.0, rng.normal(60, 40)))
        aqi = float(max(10, rng.normal(100, 45)))
        noise_db = float(max(20, rng.normal(45, 12)))
        lighting_lux = float(max(10, rng.normal(300, 180)))
        time_outdoors = float(clip01(sunlight_minutes/300.0) * (rng.uniform(20,180)))
        time_in_office = float(max(0.0, rng.normal(8 if act>=2 else 4, 3)))
        social_minutes = float(max(0.0, rng.normal(40, 45)))
        sedentary_hours = float(max(0.0, rng.normal(9 if sedentary else 7, 2)))

        # build day_features
        day_features = {
            "steps": steps,
            "exercise_min": exercise_min,
            "exercise_intensity": exercise_intensity,
            "sleep_hours": sleep_hours,
            "water_liters": water_l,
            "calories": calories,
            "caffeine_mg": caffeine_mg,
            "sunlight_minutes": sunlight_minutes,
            "aqi": aqi,
            "noise_db": noise_db,
            "lighting_lux": lighting_lux,
            "time_outdoors": time_outdoors,
            "time_in_office": time_in_office,
            "social_minutes": social_minutes,
            "sedentary_hours": sedentary_hours
        }

        # Compute daily deltas
        stress_delta, mood_delta, debug_scores = daily_env_effects(day_features, profile_row)

        # Temporal coupling (simple discrete dynamics)
        # We use Euler update: new_stress = prev_stress + lambda * (stress_base + daily_delta - prev_stress)
        lambda_s = 0.35   # inertia for stress (0..1)
        lambda_m = 0.25   # inertia for mood

        # instantaneous raw (profile base + delta)
        raw_stress_inst = baseline_modifiers(profile_row)[0] + stress_delta
        # update
        new_stress = prev_stress + lambda_s * (raw_stress_inst - prev_stress) + rng.normal(0, 0.35)
        # clamp
        new_stress = float(np.clip(new_stress, 1.0, 10.0))

        # mood influenced negatively by stress (coupling coefficient)
        stress_influence = 0.6
        raw_mood_inst = baseline_modifiers(profile_row)[1] + mood_delta - stress_influence * (new_stress - 4.0)
        new_mood = prev_mood + lambda_m * (raw_mood_inst - prev_mood) + rng.normal(0, 0.4)
        new_mood = float(np.clip(new_mood, 1.0, 10.0))

        # derived vitals (approx)
        heart_rate = float(np.clip(60 + 0.002*steps + 0.8*(new_stress-4.0) + (exercise_min/30.0)*4 + rng.normal(0,3), 40, 160))
        bp_sys = float(np.clip(110 + 2.0*(new_stress-4.0) + 0.01*caffeine_mg + rng.normal(0,6), 90, 200))
        bp_dia = float(np.clip(70 + 1.0*(new_stress-4.0) + rng.normal(0,4), 50, 120))
        spo2 = float(np.clip(98 - max(0,(aqi-120))/180 + rng.normal(0,0.3), 80, 100))

        # confidence metric: how many strong daily features present
        present_features_count = sum(1 for v in day_features.values() if not pd.isna(v))
        total_features = len(day_features)
        confidence = present_features_count / total_features

        # assemble row
        row = {
            "user_id": profile_row['user_id'],
            "day": day,
            **day_features,
            "heart_rate": round(heart_rate,1),
            "bp_sys": round(bp_sys,1),
            "bp_dia": round(bp_dia,1),
            "spo2": round(spo2,1),
            "stress": round(new_stress,2),
            "mood": round(new_mood,2),
            "confidence": round(confidence,3),
            "smoker": profile_row['smoker'],
            "sedentary_profile": profile_row['sedentary_profile']
        }

        # Introduce random missingness
        if rng.random() < missing_rate:
            # choose k features to drop randomly
            k = rng.integers(1, max(2, int(0.06*len(row))))
            keys = [k for k in list(day_features.keys())]
            drop_keys = rng.choice(keys, size=k, replace=False)
            for dk in drop_keys:
                row[dk] = np.nan

        rows.append(row)

        # update previous states
        prev_stress = new_stress
        prev_mood = new_mood

    return rows

# ----------------------------
# Top-level generator
# ----------------------------
def generate_dataset(n_profiles=10, days_per_profile=14, seed=42,
                     smoker_pct=0.2, sedentary_pct=0.4, sex_filter='even',
                     age_min=20, age_max=60, missing_rate=0.03, output_csv=None):
    rng = np.random.default_rng(seed)
    profiles = create_profiles(n_profiles, sex_filter=sex_filter, age_min=age_min,
                               age_max=age_max, smoker_pct=smoker_pct,
                               sedentary_pct=sedentary_pct, seed=seed)
    all_rows = []
    for idx, prow in profiles.iterrows():
        prow_dict = prow.to_dict()
        rows = simulate_profile_days(prow_dict, days=days_per_profile, rng=rng, missing_rate=missing_rate)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    # reorder columns to place identifiers first
    cols = ['user_id','day','age','sex','smoker','sedentary_profile'] if 'age' in profiles.columns else ['user_id','day']
    # merge age/sex/smoker into df
    df = df.merge(profiles[['user_id','age','sex','smoker','sedentary_profile','height_cm','weight_kg','bmi','activity_level']], on='user_id', how='left')
    # optional: drop duplicates
    df = df.sort_values(['user_id','day']).reset_index(drop=True)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved dataset to {output_csv} ({len(df)} rows).")

    return df, profiles

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic wellness dataset.")
    p.add_argument("--n_profiles", type=int, default=DEFAULTS["n_profiles"])
    p.add_argument("--days", type=int, default=DEFAULTS["days_per_profile"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--smoker_pct", type=float, default=DEFAULTS["smoker_pct"])
    p.add_argument("--sedentary_pct", type=float, default=DEFAULTS["sedentary_pct"])
    p.add_argument("--sex", type=str, default=DEFAULTS["sex_filter"])
    p.add_argument("--age_min", type=int, default=DEFAULTS["age_min"])
    p.add_argument("--age_max", type=int, default=DEFAULTS["age_max"])
    p.add_argument("--missing_rate", type=float, default=DEFAULTS["missing_rate"])
    p.add_argument("--out", type=str, default=DEFAULTS["output_csv"])
    return p.parse_args()

def main_cli():
    args = parse_args()
    df, profiles = generate_dataset(n_profiles=args.n_profiles,
                                    days_per_profile=args.days,
                                    seed=args.seed,
                                    smoker_pct=args.smoker_pct,
                                    sedentary_pct=args.sedentary_pct,
                                    sex_filter=args.sex,
                                    age_min=args.age_min,
                                    age_max=args.age_max,
                                    missing_rate=args.missing_rate,
                                    output_csv=args.out)
    print("Profiles sample:")
    print(profiles.head())
    print("Data sample:")
    print(df.head())

if __name__ == "__main__":
    main_cli()
