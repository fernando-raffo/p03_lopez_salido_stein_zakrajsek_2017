# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Case Study: Can the LSZ Sentiment Framework Speak to 2020-2022?
# ### An honest out-of-sample probe — and why the decisive test can't be run
#
# **Motivation.** The 2020-2021 period looked, on the surface, like the
# configuration Lopez-Salido, Stein & Zakrajsek (2017) call *elevated
# credit-market sentiment*: after the Fed's March-2020 backstop, the
# Baa-Treasury spread compressed sharply and high-yield issuance surged. LSZ's
# mechanism says elevated sentiment predicts a subsequent slowdown, as frothy
# credit conditions mean-revert. The natural question: **did the framework see
# the 2022 slowdown coming?**
#
# **The finding, stated up front.** We *cannot* run the decisive test, and that
# limitation is the main result. LSZ's sentiment signal is a two-part first
# stage — a high-yield issuance share and the spread level, both lagged two
# years, jointly forecasting the change in the spread. **The high-yield
# issuance share, the variable that actually captures "froth," is unavailable
# after 2008** in our data (it comes from Mergent FISD via WRDS, which we lack;
# the manually-transcribed series ends in 2008). So for the COVID window we can
# only use the *spread-mean-reversion* half of the signal — and that half,
# tested honestly, is too weak to call the episode: it points the right way in
# 2022 but the wrong way in 2021 and 2023.
#
# The interesting result, then, is not "the framework called it." It is that
# **the piece of the mechanism that would matter most for 2020-21 — issuance
# froth — is precisely the piece we cannot observe.**

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
import replicate_table_2 as t2

# %% [markdown]
# ## 1. The LSZ first stage, replicated on 1929-2015
#
# Table II's first stage forecasts the change in the credit spread from
# twice-lagged sentiment:
#
# $$\Delta s_t = a_0 + a_1 \ln(\mathrm{HYS})_{t-2} + a_2\, s_{t-2}$$
#
# We fit it on the replication sample and read off the coefficients. The signs
# are the LSZ mechanism: a high past issuance share predicts the spread
# *widening* ($a_1 > 0$), and a wide spread mean-reverts ($a_2 < 0$).

# %%
df = t2.build_panel()
res = t2.run_table_2(df)          # fit on 1929-2015
aux = res["aux_spread"]
c = aux.params
print("First-stage coefficients (fit 1929-2015):")
for k, v in c.items():
    print(f"  {k:>12}: {v:+.4f}")
print(f"\n  a1 (ln HYS_t-2) = {c['ln_hys_lag2']:+.3f}  -> high issuance share predicts widening")
print(f"  a2 (s_t-2)      = {c['spread_lag2']:+.3f}  -> wide spread mean-reverts")

# %% [markdown]
# Both signs replicate the paper's logic. This is a genuine reproduction of the
# sentiment engine on the historical sample — the machinery works.

# %% [markdown]
# ## 2. The data wall: the froth variable ends in 2008
#
# To apply this first stage to the COVID window, we need $\ln(\mathrm{HYS})$
# for 2018-2022. It isn't there.

# %%
last_hys = int(df["ln_hys"].dropna().index.max())
recent = df.loc[2018:2024, ["ln_hys", "spread_lag2", "d_spread", "dy"]].round(3)
print(f"Last year with a high-yield share observation: {last_hys}")
print(recent)

# %% [markdown]
# Every `ln_hys` value from 2009 on is missing. So the sentiment-driven
# prediction — the whole point of the two-step — cannot be formed for any COVID
# year. We are left with only the spread-mean-reversion term $a_0 + a_2 s_{t-2}$.

# %% [markdown]
# ## 3. What the spread-only fragment predicts — and how it does
#
# Dropping the (missing) issuance term, we form the partial prediction
# $\widehat{\Delta s_t} = a_0 + a_2\, s_{t-2}$ for 2021-2023 and compare it to
# the realized change in the spread. This is *not* the LSZ sentiment test — it
# is the mean-reversion half alone, and we report it as such.

# %%
rows = []
for yr in [2021, 2022, 2023]:
    s_lag2 = df.loc[yr, "spread_lag2"]
    pred = c["const"] + c["spread_lag2"] * s_lag2
    actual = df.loc[yr, "d_spread"]
    hit = "correct sign" if np.sign(pred) == np.sign(actual) else "WRONG sign"
    rows.append((yr, round(s_lag2, 2), round(pred, 3), round(actual, 3), hit))
partial = pd.DataFrame(rows, columns=["year", "s_(t-2)", "pred_dS_spread_leg",
                                      "actual_dS", "direction"])
print(partial.to_string(index=False))

# %% [markdown]
# The scorecard is **one-for-three**:
#
# - **2021:** predicts widening, spread actually *compressed* hard — wrong.
# - **2022:** predicts widening, spread *widened* — right, and close in size.
# - **2023:** predicts widening, spread *compressed* — wrong.
#
# The reason is unflattering: because spreads sat below their long-run mean
# through this period, the mean-reversion term predicts "widening toward
# average" almost every year. It happened to be right in 2022 — the year of the
# slowdown everyone remembers — but calling that a successful forecast would be
# cherry-picking the one hit. On its own, the spread-reversion signal does not
# reliably call the COVID cycle.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
x = partial["year"].astype(str)
ax.bar(np.arange(len(x)) - 0.2, partial["pred_dS_spread_leg"], 0.4,
       label="Predicted Δs (spread leg only)", color="#8faadc")
ax.bar(np.arange(len(x)) + 0.2, partial["actual_dS"], 0.4,
       label="Actual Δs", color="#1f4e79")
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(np.arange(len(x)))
ax.set_xticklabels(x)
ax.set_ylabel("Change in Baa-Treasury spread (pp)")
ax.set_title("Spread-reversion prediction vs. reality, 2021-2023 (1-for-3)")
ax.legend()
plt.show()

# %% [markdown]
# ## 4. Why this is the finding
#
# The honest conclusion is about *observability*, not about vindicating or
# refuting LSZ:
#
# 1. **The sentiment two-step replicates on history** (Section 1) — the
#    mechanism is real and the signs are right on 1929-2015.
# 2. **It cannot be tested on COVID**, because the high-yield issuance share —
#    the variable that would register the 2020-21 froth — is unavailable after
#    2008 without WRDS/Mergent access.
# 3. **The remaining spread-only fragment is too weak to substitute** (1-for-3),
#    and its one success is the mean-reversion term getting lucky in 2022.
#
# So the framework's most COVID-relevant component is exactly the one we are
# blind to. That is a real limitation worth stating plainly: a proper
# out-of-sample test of LSZ on the 2020-2022 episode is not possible with
# publicly transcribable data, and would require the modern (post-2008)
# high-yield issuance series from Mergent FISD. Extending that series is the
# natural next step — and until it exists, claims that credit-market sentiment
# "called" the post-COVID slowdown should be treated with caution.

# %% [markdown]
# ## Summary
#
# We replicate the LSZ sentiment first stage on 1929-2015 (correct signs), then
# show that the decisive out-of-sample test for 2020-2022 cannot be run: the
# high-yield issuance share ends in 2008, and the spread-reversion fragment
# that remains calls only one of three years. The contribution is the honest
# mapping of what *can* and *cannot* be concluded — and a concrete data need
# (post-2008 HY issuance) for anyone who wants to finish the test.
