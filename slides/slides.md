---
theme: default
title: "Генерация синтетических данных для оценки алгоритмов поиска в векторных базах данных"
highlighter: shiki
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Генерация синтетических данных для оценки алгоритмов поиска в векторных базах данных

Архипов Прохор, группа J4141

<div class="pt-12 text-gray-400">
Курсовая работа по Машинному Обучению, ИТМО, Осень 2026
</div>

---
layout: default
class: subheading
---

# Контекст
Задача векторного поиска

В множестве векторов $X \subset \R^d$ найти $k$ ближайших соседей вектора-запроса $q$: $\underset{u \in X}{\operatorname{arg\,min}} \ \delta(q, u).$

Здесь и далее мы будем использовать расстояние $L_2$: $\delta(x, y) = \|x - y\|_2$ и говорить про задачу приближённого поиска ближайших соседей (ANN).

Где встречается задача векторного поиска:

- Semantic search
- RAG
- RecSys
- Multimodal search

---

# Контекст

Рассмотрим модификацию задачи: векторный поиск в распределённой базе данных:

<div class="flex flex-col items-center justify-center mt-2">
  <img src="/03-distrib-search.svg" class="h-75" />
</div>

---
layout: default
class: subheading
---

# Контекст
SOTA-алгоритмы

- RaBitQ -- алгоритм квантизации (сжатия) векторной базы данных: использует *n_bits битов* на координату вектора 
- HNSW -- алгоритм быстрого векторного поиска, использует *графовый индекс*

<div class="grid grid-cols-2 gap-6 mt-4 items-center">
  <div class="flex flex-col items-center">
    <img src="/04-hnsw.svg" class="h-72 w-auto" />
  </div>
  <div class="flex flex-col items-center">
    <img src="/04-rabitq.svg" class="h-72 w-auto" />
  </div>
</div>

---

# Постановка задачи

Здесь ставим задачу: научиться генерировать синтетические данные так, чтобы они повторяли какие-то характеристики датасета + были достаточно точными (не завышали метрики для наших алгоритмов)

---

# Обзор решений

Здесь кратко про классы алгоритмов и их ярких представителей. Отдельно про класс, в котором находится cuVS. Сказать про то, почему не рассматриваем другие классы (нет доступа к данным заказчика, нужно повторение распределения датасет). Выделяем три алгоритма, которые будем сравнивать: cuVS, DisturbGen, GMM.

---

# Датасет

Здесь кратко рассказать про то, как устроен LAION. Одним предложением про то, почему выбрали именно его: по распределению больше всего похож на данные заказчика

---

# Метрики

Throwback к контексту: напоминаем про распределённость, шарды, роутер.
Далее подробно говорим про метрики: rho(q), recall от количества бит квантизации.
Дополнительные статистические метрики: хабы (скошенность N_10), доля дубликатов, C2ST, спектр ковариации
Ещё что-то?

---

# Алгоритмы

Три алгоритма: DisturbGen baseline, GMM (sklearn), cuVS (базовый)

Дальше можно ещё сказать про вариацию cuVS + blur, но в основных сравнениях сравниваем базовый cuVS

---

# Гипотезы

**H1 (over-separation):**  
Perturbation baseline over-estimates ρ(q) by > 0.02 — clean Gaussian clusters mirror unblurred cuVS behaviour.

**H2 (GMM captures variance):**  
Diagonal GMM reproduces per-bit recall@100 more accurately than the baseline because per-dimension variance is a first-order predictor of quantisation difficulty.

**H3 (all fail at low bits):**  
At 1–3 bits, all methods underestimate LAION difficulty because none captures its continuous intrinsic dimensionality (~24).

---

# Data

<div class="grid grid-cols-2 gap-4">
<div>

**Real data (ground truth)**

| Property | Value |
|----------|-------|
| Dataset | LAION-10M |
| Embedding model | CLIP ViT-L/14 |
| Dimensionality | 768 |
| Scale | ~10M vectors |
| Intrinsic dim (LID₁₀₀) | 24.3 |
| Hub skewness N₁₀ | 3.5 |

</div>
<div>

**Generator inputs**

- 100k-row sample for fitting (same sample for all three)
- Synthetic output: 1M and 10M vectors
- Ground truth computed with Faiss brute-force (k = 200)
- Two query sets: *heldout* (points not in base) and *jitter* (base point + noise)

</div>
</div>

---

# Evaluation: Routing Recall ρ(q)

ρ(q) measures what fraction of a query's k nearest neighbours land in the shard routed to.

**Setup:** balanced k-means, N = 16 shards, r = 100 neighbours

<div class="text-center my-6 text-gray-400 italic">
  [ ρ(q) curves — centroid router and oracle — will appear here ]
  <br/>
  x-axis: k (1 → 100), y-axis: ρ̄, one line per method
</div>

A systematic shift > 0.02 means benchmark routing scores don't transfer to real data.

---

# Evaluation: Quantisation (RaBitQ)

Minimum bits for recall@100 ≥ 0.99 in a flat protocol with top-L re-ranking.

<div class="text-center my-6 text-gray-400 italic">
  [ Recall@100 vs bits curves — 1–8 bits — will appear here ]
  <br/>
  One line per method + LAION ground truth
</div>

**Key numbers from prior work (cuVS + blur vs LAION-10M):**

| Bits | LAION recall | cuVS+blur recall | Δ |
|------|-------------|-----------------|---|
| 4 | — | — | ≤ 0.005 |
| 7 | 0.990 | 0.991 | 0.001 |
| 1 | 0.719 | 0.754 | 0.035 |

Baseline and GMM results: TBD.

---

# Statistical Properties

<div class="text-center my-6 text-gray-400 italic">
  [ Table: LAION vs all three generators — LID, loc_d90, hub skewness, C2ST ]
</div>

**Reference numbers (LAION-10M vs cuVS variants):**

| | LAION 10M | cuVS 10M | cuVS+blur 10M |
|--|--|--|--|
| C2ST | — | 0.887 | 0.865 |
| LID₁₀₀ | 24.3 | 20.4 | 21.9 |
| loc_d90 | 51 | 23 | 25 |
| hub skewness | 3.5 | 2.3 | 23.2 |

---

# Results Summary

<div class="text-center my-8 text-gray-400 italic">
  [ Summary table — all three methods × all metrics — will appear here ]
</div>

**Expected outcome:**  
- cuVS + blur meets both ρ and quantisation thresholds at 4+ bits  
- Perturbation baseline likely fails ρ threshold (H1)  
- GMM likely improves quantisation accuracy over baseline (H2)  
- All three likely fail at 1–3 bits (H4)

---

# Conclusions

> **Which method should you use to generate synthetic data for ANN benchmarks?**

<div class="mt-6">

- **DisturbGen perturbation** is the simplest but its over-separated clusters likely bias routing metrics
- **Diagonal GMM** better captures per-dimension variance and is likely more accurate for quantisation
- **cuVS + blur** is the most carefully validated option: meets ρ and quantisation thresholds at ≥ 4 bits, but introduces hub artefacts
- None of the three reliably reproduces LAION at 1–3 bits

</div>

---

# Limitations & Future Work

**Limitations**
- Results are LAION/CLIP-specific; other embedding models may differ
- 1M and 10M scales only; behaviour may change at 100M+
- GMM fitting on 100k sample with 1000 components is underdetermined

**Future directions**
- Flow-based generators (normalising flows) for better distributional fidelity
- Conditional generation matching query difficulty distribution
- Validation on graph indexes (HNSW, NSG), not just flat protocols

---
layout: two-cols-header
---

# Спасибо за внимание

::left::

<div class="flex flex-col items-center justify-center h-full">
  <img src="/qr-code.png" class="w-56" />
  <a href="https://github.com/roscherk/itmo-masters-ml-coursework" class="mt-4 text-sm text-gray-400 no-underline">ссылка на репозиторий</a>
</div>

::right::

<div class="flex flex-col justify-center h-full">

**Источники**

<div class="text-sm leading-relaxed">

- Gottesbüren et al., *Scaling Graph-Based ANNS*, VLDB 2025
- Gao & Long, *RaBitQ*, SIGMOD 2024
- NVIDIA cuVS Benchmarking Guide
- Lopez-Paz & Oquab, *Classifier Two-Sample Tests*, ICLR 2017
- Facco et al., *TwoNN intrinsic dimensionality*, Scientific Reports 2017
- Radovanović et al., *Hubs in Space*, JMLR 2010

</div>

</div>
