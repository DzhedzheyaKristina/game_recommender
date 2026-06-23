# Recommendation Examples

- Warning: these examples use synthetic demo scenarios and are intended for technical validation only.
- Warning: the bundled sample dataset is tiny, so these examples are not suitable for scientific conclusions.

## synthetic_demo_1

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Star Voyage. Preferred qualities: exploration, repetitive, management, liked, later. Avoided qualities: slow, ship, pacing, nice, mission.
- Ground truth games: Galaxy Raiders (102)
- Interpretation: Baseline retrieved a ground-truth game, but the LLM-ranked list did not.

Baseline top-5 recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 102 | Galaxy Raiders | 0.276898 | True |
| 2 | 104 | Harvest Haven | 0.190387 | False |
| 3 | 103 | Cozy Farm Days | 0.132915 | False |
| 4 | 106 | Castle Conquest | 0.111803 | False |
| 5 | 105 | Dungeon Tactics | 0.098482 | False |

LLM top-5 recommendations: unavailable.


## synthetic_demo_2

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Galaxy Raiders. Preferred qualities: space, ship, variety, upgrades, satisfying. Avoided qualities: story, loop, heavy, grind, fun.
- Ground truth games: Star Voyage (101)
- Interpretation: Baseline retrieved a ground-truth game, but the LLM-ranked list did not.

Baseline top-5 recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 101 | Star Voyage | 0.280225 | True |
| 2 | 104 | Harvest Haven | 0.13336 | False |
| 3 | 105 | Dungeon Tactics | 0.109949 | False |
| 4 | 103 | Cozy Farm Days | 0.107765 | False |
| 5 | 106 | Castle Conquest | 0.106593 | False |

LLM top-5 recommendations: unavailable.


## synthetic_demo_3

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Cozy Farm Days. Preferred qualities: farming, cozy, tasks, repeat, pace. Avoided qualities: village, feel, cute, crafting, controls.
- Ground truth games: Harvest Haven (104)
- Interpretation: Baseline retrieved a ground-truth game, but the LLM-ranked list did not.

Baseline top-5 recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 104 | Harvest Haven | 0.316807 | True |
| 2 | 101 | Star Voyage | 0.125116 | False |
| 3 | 102 | Galaxy Raiders | 0.098499 | False |
| 4 | 105 | Dungeon Tactics | 0.093741 | False |
| 5 | 106 | Castle Conquest | 0.090644 | False |

LLM top-5 recommendations: unavailable.


## synthetic_demo_4

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Harvest Haven. Preferred qualities: farming, warm, village, routines, relaxing. Avoided qualities: tool, swapping, slow, pleasant, pacing.
- Ground truth games: Cozy Farm Days (103)
- Interpretation: Baseline retrieved a ground-truth game, but the LLM-ranked list did not.

Baseline top-5 recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 103 | Cozy Farm Days | 0.345607 | True |
| 2 | 101 | Star Voyage | 0.197543 | False |
| 3 | 102 | Galaxy Raiders | 0.128817 | False |
| 4 | 106 | Castle Conquest | 0.094088 | False |
| 5 | 105 | Dungeon Tactics | 0.072646 | False |

LLM top-5 recommendations: unavailable.


## synthetic_demo_5

- Scenario type: synthetic_demo
- Preference text: I want a game similar to Dungeon Tactics. Preferred qualities: dungeon, tactical, smart, run, positioning. Avoided qualities: strategy, runs, rough, punishing, interface.
- Ground truth games: Castle Conquest (106)
- Interpretation: Baseline retrieved a ground-truth game, but the LLM-ranked list did not.

Baseline top-5 recommendations:
| rank | game_id | game_title | score | is_ground_truth |
| --- | --- | --- | --- | --- |
| 1 | 106 | Castle Conquest | 0.265852 | True |
| 2 | 102 | Galaxy Raiders | 0.112569 | False |
| 3 | 103 | Cozy Farm Days | 0.097453 | False |
| 4 | 101 | Star Voyage | 0.095354 | False |
| 5 | 104 | Harvest Haven | 0.07305 | False |

LLM top-5 recommendations: unavailable.
