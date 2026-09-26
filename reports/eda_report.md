# Exploratory Data Analysis Report

This report presents strictly descriptive findings computed directly from the executed exploratory data analysis notebook (`notebooks/01_eda.ipynb`) across training and test splits.

---

## 1. Actual Dataset Dimensions & Row Counts

| Dataset | Rows | Columns | In-RAM Memory | Description |
|---|---|---|---|---|
| `train_source1` | 2,206,821 | 4 | 633.8 MB | Primary training entity table |
| `train_source2` | 5,034,616 | 4 | 1,487.7 MB | Secondary training entity table |
| `train_source3` | 5,285,603 | 4 | 1,550.3 MB | Tertiary training entity table |
| `train_ground_truth` | 2,206,821 | 2 | 336.8 MB | Ground-truth match mapping table |
| `test_source1` | 1,732,544 | 4 | 509.5 MB | Primary test entity table |
| `test_source2` | 4,887,273 | 4 | 1,488.7 MB | Secondary test entity table |
| `test_source3` | 5,082,316 | 4 | 1,523.0 MB | Tertiary test entity table |

---

## 2. Actual Missing-Value Statistics

Across all six entity datasets (24,193,173 total entity records):
- **`entity_id`**: 0 missing (0.000%) across all sources.
- **`country`**: 0 missing (0.000%) across all sources.
- **`business_name`**:
  - `train_source1`: 0 missing (0.0000%), 0 whitespace-only (0.000%)
  - `train_source2`: 2 missing (0.00004%), 0 whitespace-only (0.000%)
  - `train_source3`: 13 missing (0.00025%), 0 whitespace-only (0.000%)
  - `test_source1`: 0 missing (0.0000%), 0 whitespace-only (0.000%)
  - `test_source2`: 46 missing (0.00094%), 0 whitespace-only (0.000%)
  - `test_source3`: 59 missing (0.00116%), 0 whitespace-only (0.000%)
- **`business_address`**:
  - `train_source1`: 0 missing (0.0000%), 0 whitespace-only (0.000%)
  - `train_source2`: 168,967 missing (3.3561%), 0 whitespace-only (0.000%)
  - `train_source3`: 175,916 missing (3.3282%), 0 whitespace-only (0.000%)
  - `test_source1`: 0 missing (0.0000%), 0 whitespace-only (0.000%)
  - `test_source2`: 129,408 missing (2.6479%), 0 whitespace-only (0.000%)
  - `test_source3`: 136,098 missing (2.6779%), 0 whitespace-only (0.000%)

---

## 3. Actual Duplicate Statistics

- **`entity_id` Uniqueness**:
  - `train_source1`: 2,206,821 / 2,206,821 unique (0 duplicates, 0.0%)
  - `train_source2`: 5,034,616 / 5,034,616 unique (0 duplicates, 0.0%)
  - `train_source3`: 5,285,603 / 5,285,603 unique (0 duplicates, 0.0%)
  - Zero duplicate entity IDs were observed within any source file.
- **`business_name` Multiplicity**:
  - `train_source1`: 1,539,229 distinct names; 177,793 names occur >1 time.
    - Top occurrences: "Primary Care Group" (253x), "Ear Nose & Throat Group" (251x), "Pediatric Group" (222x), "Womens Health Group" (220x), "Physical Therapy Group" (218x).
  - `train_source2`: 4,402,008 distinct names; 239,778 names occur >1 time.
    - Top occurrences: "Primary Care" (320x), "Physical Therapy" (307x), "Urgent Care" (297x), "Womens Health" (297x), "Behavioral Health" (296x).
  - `train_source3`: 4,651,608 distinct names; 258,275 names occur >1 time.
    - Top occurrences: "Primary Care" (421x), "Physical Therapy" (399x), "Pediatric Dental" (393x), "Womens Health" (379x), "Urgent Care" (377x).
- **`business_address` Multiplicity**:
  - `train_source1`: 2,130,606 distinct addresses; 40,089 occur >1 time. Top: "108 Norle Street, College Twp, PA" (14x).
  - `train_source2`: 4,337,261 distinct addresses; 421,472 occur >1 time. Top: "842 38, CALEDONAI TOWNSHIP, WI" (18x) and "1948 WEAVER FOREST WAY, MORRISVILLE, NC" (18x).
  - `train_source3`: 4,632,764 distinct addresses; 382,890 occur >1 time. Top: "Ground Floor, Bangalore, KA" (29x).

---

## 4. Actual ID Prefix Validation Results

Full chunked scan across 24,193,173 entity rows:

| Dataset | Total Rows | Expected Prefix | Observed Prefixes | Anomalies | Validation Status |
|---|---|---|---|---|---|
| `train_source1` | 2,206,821 | `S1-` | `[('S1-', 2206821)]` | 0 | PASS |
| `train_source2` | 5,034,616 | `S2-` | `[('S2-', 5034616)]` | 0 | PASS |
| `train_source3` | 5,285,603 | `S3-` | `[('S3-', 5285603)]` | 0 | PASS |
| `test_source1` | 1,732,544 | `S1-` | `[('S1-', 1732544)]` | 0 | PASS |
| `test_source2` | 4,887,273 | `S2-` | `[('S2-', 4887273)]` | 0 | PASS |
| `test_source3` | 5,082,316 | `S3-` | `[('S3-', 5082316)]` | 0 | PASS |

---

## 5. Actual Country Distributions

| Dataset | US Count (%) | India Count (%) | France Count (%) | Total Rows |
|---|---|---|---|---|
| `train_source1` | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | 2,206,821 |
| `train_source2` | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | 5,034,616 |
| `train_source3` | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.00%) | 5,285,603 |
| `test_source1` | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 1,732,544 |
| `test_source2` | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 4,887,273 |
| `test_source3` | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) | 5,082,316 |

---

## 6. Actual Business-Name Statistics

Character length distribution across non-null records:

| Source | Non-Null Count | Min | Max | Mean | Median | p25 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1` | 2,206,821 | 3 | 105 | 24.1 | 24.0 | 18.0 | 30.0 | 37.0 | 42.0 |
| `train_source2` | 5,034,614 | 2 | 104 | 25.1 | 25.0 | 18.0 | 31.0 | 40.0 | 48.0 |
| `train_source3` | 5,285,590 | 2 | 123 | 25.2 | 25.0 | 18.0 | 31.0 | 42.0 | 50.0 |
| `test_source1` | 1,732,544 | 3 | 92 | 23.8 | 24.0 | 18.0 | 29.0 | 36.0 | 42.0 |
| `test_source2` | 4,887,227 | 2 | 102 | 25.7 | 25.0 | 19.0 | 32.0 | 41.0 | 49.0 |
| `test_source3` | 5,082,257 | 2 | 103 | 25.6 | 25.0 | 19.0 | 32.0 | 42.0 | 50.0 |

- **Short Name Records (<= 5 characters)**: In `train_source1`, 6,252 records have lengths <= 5 characters (samples: `Meon`, `Proum`, `Faor`, `Buus`, `Minoo`, `Meex`, `Ceque`, `Peum`).
- **Long Name Records (>= 200 characters)**: Exactly 0 records across all six sources.

---

## 7. Actual Address Statistics

Character length distribution and formatting anomalies:

| Source | Non-Null | Null | No-Digit Count (%) | All-Uppercase Count (%) | Min | Max | Mean | Median | p25 | p75 | p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `train_source1` | 2,206,821 | 0 | 77,037 (3.49%) | 4 (0.00%) | 11 | 256 | 52.1 | 41.0 | 33.0 | 70.0 | 103.0 |
| `train_source2` | 4,865,649 | 168,967 | 301,974 (6.00%) | 3,191,104 (63.38%) | 8 | 249 | 47.9 | 37.0 | 31.0 | 63.0 | 97.0 |
| `train_source3` | 5,109,687 | 175,916 | 310,390 (5.87%) | 813 (0.02%) | 2 | 240 | 48.4 | 42.0 | 35.0 | 55.0 | 92.0 |
| `test_source1` | 1,732,544 | 0 | 71,838 (4.15%) | 2 (0.00%) | 11 | 268 | 57.1 | 50.0 | 36.0 | 74.0 | 105.0 |
| `test_source2` | 4,757,865 | 129,408 | 228,622 (4.68%) | 2,448,518 (50.10%) | 5 | 269 | 51.8 | 43.0 | 32.0 | 68.0 | 99.0 |
| `test_source3` | 4,946,218 | 136,098 | 245,395 (4.83%) | 914 (0.02%) | 5 | 267 | 50.0 | 44.0 | 36.0 | 59.0 | 94.0 |

---

## 8. Actual Ground-Truth Match-Count Distribution

Distribution of matched entities per `source1_entity_id` in `train_ground_truth`:

| Match Count Bucket | Entity Count | Percentage |
|---|---|---|
| `0` | 123,247 | 5.585% |
| `1` | 119,157 | 5.399% |
| `2` | 375,212 | 17.002% |
| `3` | 530,841 | 24.055% |
| `4+` | 1,058,364 | 47.959% |
| **Total S1 Entities** | **2,206,821** | **100.000%** |

---

## 9. Actual S2-Only / S3-Only / Both / No-Match Distribution

Classification of S1 entities by the origin sources of their ground-truth matches:

| Category | S1 Entity Count | Percentage |
|---|---|---|
| `both_S2_S3` | 1,776,047 | 80.480% |
| `S3_only` | 164,498 | 7.454% |
| `S2_only` | 143,029 | 6.481% |
| `no_match` | 123,247 | 5.585% |
| **Total** | **2,206,821** | **100.000%** |

---

## 10. Actual Match-Count Statistics

Summary statistics for `match_count` in `train_ground_truth`:
- **Total S1 Entities Evaluated**: 2,206,821
- **Total Matched Pairs**: 7,638,365
- **Mean**: 3.4613
- **Standard Deviation**: 1.7053
- **Minimum**: 0
- **10th Percentile (p10)**: 1.0
- **25th Percentile (p25)**: 2.0
- **Median (p50)**: 3.0
- **75th Percentile (p75)**: 5.0
- **90th Percentile (p90)**: 6.0
- **95th Percentile (p95)**: 6.0
- **99th Percentile (p99)**: 8.0
- **Maximum**: 11
- **Entities with match_count > 8 (p99)**: 4,776 entities (top 10 observed IDs with 11 matches: `S1-806895726`, `S1-971572082`, `S1-958050134`, `S1-662651729`, `S1-6278821`, `S1-765235386`, `S1-709509922`, `S1-439823948`, `S1-922469887`, `S1-667470626`).

---

## 11. Actual Representative Examples

### A. Zero Matches (`match_count = 0`)
| Entity ID | Business Name | Country |
|---|---|---|
| `S1-302869473` | International Automation Consultants Inc | US |
| `S1-262997549` | Gabriella's Preferred Security | US |
| `S1-508022910` | Twyla's Liquor Corp | US |
| `S1-666499407` | Vadapalani Projects Pvt. Ltd. | India |
| `S1-965524997` | Campbell Property Solutions | US |

### B. Exactly One Match (`match_count = 1`)
| S1 Entity ID | S1 Business Name | Country | Match Source | Matched Entity ID |
|---|---|---|---|---|
| `S1-116043204` | Red Consultants Pvt. Ltd. | India | `S3_only` | `S3-85523430` |
| `S1-473377609` | Wave & Brothers Ltd | India | `S3_only` | `S3-433876173` |
| `S1-439203009` | Ace Producer Private Limited | India | `S3_only` | `S3-967165288` |
| `S1-973290215` | Gulf Ministries V Associates | US | `S3_only` | `S3-925530888` |
| `S1-649259801` | Guru Solutions Pvt Ltd | India | `S2_only` | `S2-654066445` |

### C. Matches Spanning Both S2 and S3 (`both_S2_S3`)

#### Example 1: `S1-965667` (5 matches, US)
- **S1 Record**: `Maure Williams Colombier Inc`
- **Matched S2 Records**:
  - `S2-681193310`: `Maure Wilblims Colombier Inc` (address: `NaN`)
  - `S2-743505751`: `Maure Williams Colombier` (address: `NaN`)
- **Matched S3 Records**:
  - `S3-860443364`: `Maure Williams Inc Center` (address: `NaN`)
  - `S3-775321672`: `Dréxkor` (address: `85 Wanye Avenue, Ticonderoga Townshiip, New York`)
  - `S3-11291185`: `maurewilliamscolombier.com` (address: `Wayne Ave, Ticonderoga Townshiip, New York`)

#### Example 2: `S1-55344266` (4 matches, India)
- **S1 Record**: `Raj Investments LLP`
- **Matched S2 Records**:
  - `S2-197070651`: `Raj Investments LLP`
  - `S2-249013014`: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
- **Matched S3 Records**:
  - `S3-384364074`: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
  - `S3-478195123`: `Raj Investments எல்எல்பி`

#### Example 3: `S1-343815751` (3 matches, US)
- **S1 Record**: `Dahlia Power Reliable Scientific LLC`
- **Matched S2 Records**:
  - `S2-479876582`: `Dahlia Power Reliable Scientific` (address: `45ND TERRACE, null, KANSAS CITY, MO`)
  - `S2-790675320`: `Dahlia Power Reliable` (address: `KANSAS CITY, MO, 630 45ND TERRACE, null`)
- **Matched S3 Records**:
  - `S3-878454467`: `Dahlia Ponr Reliable Scientific LLC` (address: `Missouri, 630 45th Terrace, Kansas City`)

#### Example 4: `S1-656753428` (3 matches, India)
- **S1 Record**: `Ss Food Private Limited`
- **Matched S2 Records**:
  - `S2-153058913`: `एसएस फूड प्राइवेट लिमिटेड`
  - `S2-24659151`: `एसएस फूड प्राइवेट लिमिटेड`
- **Matched S3 Records**:
  - `S3-679606215`: `एसएस फूड प्राइवेट लिमिटेड`

#### Example 5: `S1-102811957` (6 matches, US)
- **S1 Record**: `Payne Enterprises`
- **Matched S2 Records**:
  - `S2-625774905`: `PAYNE-ENRTPRMISES` (address: `3315 FREMONT SAINT, PEORIA, IL`)
  - `S2-478959098`: `Payne Énterprises` (address: `3315 FREMONT ST, PEORIA, IL`)
  - `S2-553508714`: `Payne Enterpires` (address: `3315 FREMONT ST, PEORIA, IL`)
- **Matched S3 Records**:
  - `S3-449308785`: `Payne Enterprises  LLC` (address: `Fremont St, Peoria, Illinois`)
  - `S3-728090388`: `Payne Etrepndiels` (address: `3315 Fremont St, Peoria, Illinois`)
  - `S3-928796641`: `Payne Énterprises` (address: `3315 Fremont Street, Peoria, Illinois`)

---

## 12. Factual Data-Quality Observations Directly Supported by Results

1. **Split-Level Country Discrepancy**: Training datasets contain records exclusively from two countries: `US` (~59.95%) and `India` (~40.05%). In contrast, test datasets contain records from three countries: `India` (~47.1%), `US` (~38.3%), and `France` (~14.5% across all three test sources).
2. **Missing Address Field Asymmetry**: Source 1 datasets (`train_source1` and `test_source1`) contain 0 missing address values (0.000%). Source 2 datasets contain 168,967 (3.356%) in train and 129,408 (2.648%) in test missing address values. Source 3 datasets contain 175,916 (3.328%) in train and 136,098 (2.678%) in test missing address values.
3. **Address Casing Asymmetry**: Over 63% of addresses in `train_source2` (3,191,104 rows) and over 50% in `test_source2` (2,448,518 rows) are formatted in all-uppercase characters, compared to 4 rows in `train_source1` and 813 in `train_source3`.
4. **Digit-Free Addresses**: Non-digit addresses exist in all datasets: 77,037 (3.49%) in `train_source1`, 301,974 (6.00%) in `train_source2`, 310,390 (5.87%) in `train_source3`, 71,838 (4.15%) in `test_source1`, 228,622 (4.68%) in `test_source2`, and 245,395 (4.83%) in `test_source3`.
5. **Cross-Script Ground-Truth Equivalences**: Matched clusters for Indian entities contain business names recorded in multiple distinct writing scripts (Latin script, Tamil script, and Devanagari script).
6. **Entity ID Integrity**: Across all 24,193,173 rows examined across all 6 datasets, 100% of entity IDs match their respective source prefix (`S1-`, `S2-`, `S3-`) with zero duplicate IDs within any source file.
