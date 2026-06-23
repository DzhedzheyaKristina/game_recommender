# Case Studies

These case studies are generated from synthetic demo scenarios and are intended for technical validation only. LLM case categories are limited because reranking was skipped or unavailable. Only a small number of scenarios were available, so some case-study categories may be missing.

## synthetic_demo_1 - baseline_success

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Star Voyage. Preferred qualities: exploration, repetitive, management, liked, later. Avoided qualities: slow, ship, pacing, nice, mission.
- Ground truth: Galaxy Raiders (102)
- Interpretation: Baseline retrieved a ground-truth game at rank 1.

Baseline top recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 102 | Galaxy Raiders | 0.276898 | True |
| 2 | 104 | Harvest Haven | 0.190387 | False |
| 3 | 103 | Cozy Farm Days | 0.132915 | False |
| 4 | 106 | Castle Conquest | 0.111803 | False |
| 5 | 105 | Dungeon Tactics | 0.098482 | False |

## synthetic_demo_2 - baseline_success

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Galaxy Raiders. Preferred qualities: space, ship, variety, upgrades, satisfying. Avoided qualities: story, loop, heavy, grind, fun.
- Ground truth: Star Voyage (101)
- Interpretation: Baseline retrieved a ground-truth game at rank 1.

Baseline top recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 101 | Star Voyage | 0.280225 | True |
| 2 | 104 | Harvest Haven | 0.13336 | False |
| 3 | 105 | Dungeon Tactics | 0.109949 | False |
| 4 | 103 | Cozy Farm Days | 0.107765 | False |
| 5 | 106 | Castle Conquest | 0.106593 | False |

## synthetic_demo_3 - baseline_success

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Cozy Farm Days. Preferred qualities: farming, cozy, tasks, repeat, pace. Avoided qualities: village, feel, cute, crafting, controls.
- Ground truth: Harvest Haven (104)
- Interpretation: Baseline retrieved a ground-truth game at rank 1.

Baseline top recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 104 | Harvest Haven | 0.316807 | True |
| 2 | 101 | Star Voyage | 0.125116 | False |
| 3 | 102 | Galaxy Raiders | 0.098499 | False |
| 4 | 105 | Dungeon Tactics | 0.093741 | False |
| 5 | 106 | Castle Conquest | 0.090644 | False |
