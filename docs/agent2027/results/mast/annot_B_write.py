import json, os
D = os.path.dirname(os.path.abspath(__file__))
MODES = ["FM-1.1","FM-1.2","FM-1.3","FM-1.4","FM-1.5","FM-2.1","FM-2.2","FM-2.3","FM-2.4","FM-2.5","FM-2.6","FM-3.1","FM-3.2","FM-3.3"]

U = {}
def add(uid, ev, ms, msq, hf, notes):
    U[uid] = dict(unit_id=uid, labels={m: (1 if m in ev else 0) for m in MODES}, evidence=ev,
                  missed_signal=ms, missed_signal_quote=msq, handoff_fidelity=hf, notes=notes)

add("U24", {
 "FM-1.5": "This completes the set of higher‑order engineered attributes for the current focus. [followed by 5 further append_new_attribute calls]",
 "FM-3.3": "`f05_f13_inter` (MI = 0.47, pfi ≈ 0.0031, mean |SHAP| ≈ 0.016) remains the single most important engineered feature. [vs f05_f13_div_f07 pfi 1.27, SHAP 0.58]"},
 0, "", 2, "Planner's focus item 1 is phi itself; coder built it, evaluator mis-ranks it below f05_f13_inter.")

add("U23", {
 "FM-1.3": "Attribute f01_sq was pruned — it has been restored to Active status.",
 "FM-1.5": "completing the requested set of transformations and centered interaction products. [followed by ~10 more appends]"},
 1, "High MI & importance: f13_f05_product (MI 0.46, pfi 0.146, SHAP 0.244, VIF 1e12); f05_f07_ratio (MI 0.418, pfi 0.223, SHAP 0.248", 2,
 "Product f13*f05 and ratio f05/f07 both top-ranked; combining them (phi) not pursued.")

add("U22", {
 "FM-1.3": "temp_humidity_corr_5 resubmitted with essentially the same code and missing argumentation twice (Field required [type=missing ...])",
 "FM-1.5": "completing the planned higher‑order temporal and cyclical feature set. [then temp_autocorr_5, humidity_pressure_corr_5, temp_pressure_corr_5 ...]",
 "FM-3.2": "result = corr_series.reindex(df_features.index).fillna(0).astype(float).values  [all corr/IQR features later show MI/PFI/SHAP = 0, never checked]"},
 0, "", 2, "phi (hour_sin) was proposed, built and identified as strongest; no missed signal.")

add("U21", {
 "FM-1.3": "append_new_attribute `sensor_age_power_factor_interaction` submitted identically twice, both failing: args.argumentation Field required",
 "FM-1.5": "completing the requested set of interaction, ratio, and quantile‑binned features. [then ambient_temp_power_factor_interaction, sensor_age_linepressure_interaction, ...]",
 "FM-3.3": "Highest PFI: *supply_voltage_v* (0.0035) [own table: vibration_rms_mm_s PFI 0.004]"},
 1, "| **hours_since_start** | 0.0235 | 4.55 | **0.0005** | **0.116** | 0.0089 | **Medium** | Decent SHAP/MI, VIF acceptable. |", 2,
 "hours_since_start has highest MI in the table but no temporal/cyclic follow-up.")

add("U20", {
 "FM-1.5": "This completes the set of target‑encoded interactions with the key numeric predictors. [followed by ~8 more appends]",
 "FM-2.6": "Added target‑encoded interaction features ... `op_shift_x_stress` – operator shift ordinal × axial stress",
 "FM-3.3": "The engineered target‑encoded interactions (`*_target_enc_x_stress` and `*_target_enc_x_vib`) dominate predictive signal, achieving the highest mutual information (≈ 0.41–0.44)"},
 1, "| **member_length_m** | 0.121 | 8.87 | 0.448 | 0.308 | ★ | Very high PFI, low VIF – solid baseline. |", 1,
 "Coder used ordinals and mean axial stress as 'target encoding'. Evaluator also pruned vendor_x_stress it called reasonable.")

add("U19", {
 "FM-1.5": "completing the requested set of engineered variables for the next assessment step. [then log_axial_stress, buckling_factor, inverse_slenderness, ...]",
 "FM-2.4": "\\(\\delta = \\dfrac{P\\,L}{A\\,E}\\) [in planner search result; focus splits it into axial_stress and load_length_interaction]"},
 1, "\\(\\delta = \\dfrac{P\\,L}{A\\,E}\\)", 2,
 "Coder's extra stress_slenderness_product approximates phi and dominates; P*L/A itself never built.")

add("U18", {
 "FM-1.3": "TOOL CALL `generic_pandas_tool`: command: result = df[['f12','f17','f21']].head() [issued twice identically, both failing]",
 "FM-1.5": "completing the set of categorical encodings based on the mean of `f5` per category. [then 7 more three-way/square features]",
 "FM-2.6": "Any other three‑way interaction with PFI < 0.001 and VIF > 40 ... – keep for now ... *No pruning actions are taken in this step* [then prunes them]",
 "FM-3.3": "| **f05_f13_prod_f08** | 0.3205 | 20.0 | 0.00227 | 0.01115 | Among the better 3‑way terms. | [pruned]"},
 0, "", 1,
 "phi = f05_f13_prod_div_f07 built and recognised as strongest. Target encoding replaced by f05-mean proxy.")

add("U17", {
 "FM-1.1": "repeated append_new_attribute calls with 'explanation:' instead of required argumentation after repeated 'args.argumentation Field required' errors",
 "FM-1.3": "result = df_features['site_region_freq_enc'].values  [same failing call repeated identically]; Attribute site_region_target_enc was pruned — it has been restored"},
 1, "| **hours_since_start** | **0.036** | 36.24 | 0 | 0 | 0.0068 | 0.080 | – | ★ (only MI >0.03, VIF moderate) |", 2,
 "Most of the coder section is truncated; visible part shows many format-error retries.")

add("U16", {}, 1,
 "| **member_length_m** | 0.155 | 14.59 | 0.131 | 0.00301 | 0.151 | Redundant with `length_per_cross_section` but still informative. |", 2,
 "member_length_m PFI exceeds axial_vibration_stress yet dismissed; k fixed at 4 without elbow selection.")

add("U15", {
 "FM-3.1": "The `f23_bin_1` one‑hot column has added **​​**​  —  ​  —  added  ... [f23_bin_2..4 never created]"},
 1, "| **f05_f13** | **0.484** | **0.4016** | **0.3917** | 391.8 | / | **ratio_f05_f07** | 0.429 | 0.0876 | 0.1541 |", 1,
 "Planner moved to binning despite product f05*f13 and ratio f05/f07 topping the results.")

add("U14", {
 "FM-1.3": "append_new_attribute `f07_sq` submitted identically twice, both failing; Attribute log_f07 was pruned — it has been restored to Active status.",
 "FM-1.5": "All of the requested non‑linear transformations, difference features, ratio‑×‑raw interactions, and quantile‑binned ordinal columns have now been created [then log_f07, log_f05, log_f13 ...]",
 "FM-3.3": "| **ratio_f07_f05** | **0.413** | 432.5 | **0.0160** | **0.0610** | **Very strong** – top predictor, keep. | [ratio_f05_f07_x_f13 has PFI 1.184]"},
 0, "", 2, "ratio_f05_f07_x_f13 equals phi; built per focus.")

add("U13", {
 "FM-1.3": "Attribute ratio_sq was pruned — it has been restored to Active status. [planner re-proposed pruned ratio_sq, ratio_cubed, log_ratio]",
 "FM-1.5": "This completes the set of quadratic geometry transformations requested. [then log_slenderness]",
 "FM-2.5": "argumentation: Box-Cox like log transform to reduce skewness [focus asked for Box-Cox/Yeo-Johnson with optimised lambda; slenderness_sq not built]",
 "FM-3.3": "**A – Strong contributors** | MI ≥ 0.4, **PFI** ≥ 0.1, **mean |SHAP|** ≥ 0.15 ... | ... `axial_load_kn`, `load_vib_temp` [load_vib_temp PFI 0.000915, SHAP 0.0046]"},
 1, "| **member_length_m** | 0.155 | 112.5 | 0.0573 | 0.00186 | 0.088 | B | Good MI & SHAP; VIF high (collinear with slenderness). |", 1,
 "axial_load_per_cross_section and member_length_m both strong; their product (phi) not tried.")

add("U12", {
 "FM-1.3": "command: result = df_test_results[['id','status','cv_RMSE','cv_accuracy','feature_assessment']] [near-identical planner queries x4]; Attribute stress_humidity was pruned — it has been restored",
 "FM-1.5": "This completes the suite of physics‑inspired fatigue‑damage proxies requested. [followed by 6 more appends]",
 "FM-3.3": "the strongest predictors are the stress‑strain‑temperature interaction family (... `damage_rate`, `damage_proxy`, etc.) [damage_rate PFI 0.00084, damage_proxy PFI 0.00175]"},
 1, "| `member_length_m` | 0.158 | **42.3** | 0.0463 | 0.0862 | 0.00189 | Strong predictor", 2,
 "Planner focus on fatigue proxies; length x stress never combined.")

add("U11", {
 "FM-1.3": "command: result = df_test_results[['id','status','cv_RMSE']].sort_values('id') [issued 3 times with identical result]",
 "FM-1.5": "This completes the current exploratory focus. [followed by 6 more appends]",
 "FM-3.3": "keep **one** well‑conditioned representation ... (e.g., `log_abs_norm_load_vibration_per_area`) and **prune** ... `load_vibration_per_area` [PFI 1.012 vs 0.0055]"},
 1, "| **member_length_m** | 0.176 | 34.5 | 0.033 | 0.040 | Useful geometric predictor, acceptable VIF. |", 2,
 "Evaluator claims log_abs version 'keeps most of the predictive signal' against its own numbers.")

add("U10", {
 "FM-3.3": "Any interaction where **pfi_mean = 0** and **mean |SHAP| ≤ 0.02** (e.g., `dust_sq_x_pca4`, `ambient_temperature_c_sq_x_pca1`, `vibration_rms_mm_s_sq_x_pca1`) [own SHAP 0.064/0.037/0.071]"},
 1, "a suite of interactions with the two strongest numeric predictors (`dust_concentration_mg_m3` and `hours_since_start`)", 2,
 "Planner squared hours_since_start; no periodic encoding.")

add("U09", {
 "FM-1.3": "Attribute log_ambient_temperature_c was pruned — it has been restored to Active status.",
 "FM-1.5": "completing the set of physics‑based derived attributes outlined in the focus. [followed by ~8 more appends]",
 "FM-3.3": "The top five features dominate the predictive signal (high MI, high SHAP magnitude, and sizeable PFI). [stress PFI 0.00228, load_squared_per_area PFI 0.0012]"},
 0, "", 2, "Coder's extra bending_moment_per_area equals phi and is found dominant.")

add("U08", {
 "FM-1.5": "This completes the set of orthogonal PCA components and their vendor‑specific interaction features [followed by ~6 more appends]"},
 1, "| **hours_since_start** | **0.0354** (highest) | 55.3 (moderate) | 0.00225 | 0.0091 | **0.087** | Strong temporal trend", 2,
 "hours_since_start strongest by MI; used only multiplicatively.")

add("U07", {
 "FM-1.1": "generic_pandas_tool called ~17 times without the required explanation field despite repeated 'args.explanation Field required' errors",
 "FM-1.3": "TOOL CALL `generic_pandas_tool`: command: result = pd.Series({col: df[col].unique().tolist() for col in ['operator_shift','site_region','sensor_vendor']}) [identical, 10x]",
 "FM-1.5": "same failing generic_pandas_tool call repeated 10 times then 7 more variants before succeeding"},
 1, "| **member_length_m** | 0.185 | 0.171 | Influences slenderness and overall deformation magnitude. |", 1,
 "Target-mean encoding replaced by axial-stress mean proxy; evaluator also pruned 3 non-recommended features.")

add("U06", {
 "FM-1.5": "The requested transformations of the top‑ranked predictor `ratio_f05_f07` have been added [followed by ~10 more appends]",
 "FM-3.3": "even if the current booster does not exploit them due to multicollinearity. [int_ratio_f05_f07_f13_f07 PFI 0.665, SHAP 0.461]"},
 1, "Strong predictive signals in interaction/ratio features involving f05, f07, f13 (e.g., int_ratio_f05_f07_f13_f07, prod_f05_f13, prod_log_f05_f13", 2,
 "Focus extended ratio f05/f07 with diffs/logs instead of prod_f05_f13 / f07.")

add("U05", {
 "FM-1.3": "vibration_band_energy_low: vib_energy = df.groupby('id')['vibration_rms_mm_s'].apply(lambda x: np.mean(x**2)) [identical to vibration_energy]",
 "FM-3.1": "to support the planned cross‑modal interaction with vibration kurtosis. [interaction, spectral entropy, spectral centroid never built]"},
 1, "| hours_since_start | 0.0488", 1,
 "Focus requested FFT/spectral features not computable from the table; coder used proxies.")

add("U04", {
 "FM-1.5": "completing the exploitative focus. [then std_pc1..5, pc*_sq, log_std_freq ...]"},
 1, "`f05`, `f07`, `f13` still dominate (high MI, high PFI).", 2,
 "Frequencies computed on full data, not training folds (evaluator flagged).")

add("U03", {
 "FM-1.3": "Attribute site_region_te_x_hours_since_start already exists and is active.",
 "FM-1.5": "This completes the exploitative focus implementation. [repeated, followed by further appends]",
 "FM-3.3": "**High‑VIF continuous columns** (hours_since_start, ambient_temperature_c, sensor_age_months, supply_voltage_v, vibration_rms_mm_s) | 0 – 0.03 | 1e12 [same features earlier listed with VIF ≤ 20, SHAP 0.07–0.09]"},
 1, "Core continuous predictors (hours_since_start, ambient_temperature_c, vibration_rms_mm_s, dust_concentration_mg_m3, humidity_dust_ratio, dew_point_to_temp_ratio) show highest mean_abs_shap", 1,
 "Target encodings replaced with frequency/feature-mean surrogates (flagged by coder).")

add("U02", {
 "FM-1.3": "Attribute f05_log was pruned — it has been restored to Active status. [plus 3 near-identical planner queries]",
 "FM-3.2": "result = df_features['id'].map(cat_enc).fillna(global_mean).values [maps id onto category means -> constant; evaluator: MI≈0, PFI≈0, SHAP≈0]"},
 1, "The **core numeric predictors** (f05, f13, f07) remain the strongest contributors (high MI, high PFI, large mean‑abs‑SHAP).", 1,
 "Target encoding done with f05 as proxy target; raw *_target_enc columns are constant.")

add("U01", {
 "FM-1.5": "completing the requested dust‑ and time‑focused feature set. [followed by 7 more dust_* appends]"},
 1, "| **hours_since_start** | 0.049 | 23.17 | 0 | 0.0065 | 0.072 | Moderate SHAP, modest VIF. |", 2,
 "hours_since_start has highest MI; planner focused on dust.")

out_dir = os.path.join(D, "annot_B")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "U01-U24.jsonl"), "w", encoding="utf-8") as f:
    for i in range(1, 25):
        f.write(json.dumps(U[f"U{i:02d}"], ensure_ascii=False) + "\n")
print(len(U))
