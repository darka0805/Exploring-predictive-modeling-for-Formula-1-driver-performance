import pickle
from pathlib import Path


def save_cache(path: Path, all_laps_list, race_results, track_statuses, wet_race_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "all_laps_list": all_laps_list,
                "race_results": race_results,
                "track_statuses": track_statuses,
                "wet_race_ids": wet_race_ids,
            },
            f,
        )


def load_cache(path: Path):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return (
        cache["all_laps_list"],
        cache["race_results"],
        cache["track_statuses"],
        cache.get("wet_race_ids", set()),
    )
