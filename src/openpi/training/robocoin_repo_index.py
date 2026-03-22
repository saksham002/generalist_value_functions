import numpy as np
import pathlib


_REPOS_PATH = pathlib.Path(__file__).resolve().parents[3] / "repos.txt"
_REPO_IDS = tuple(repo_id for repo_id in _REPOS_PATH.read_text().splitlines() if repo_id)
_REPO_ID_TO_INDEX = {repo_id: index for index, repo_id in enumerate(_REPO_IDS)}


def repo_id_to_index(repo_id: str | bytes) -> int:
    repo_id = np.asarray(repo_id).item()
    if isinstance(repo_id, bytes):
        repo_id = repo_id.decode("utf-8")
    if repo_id.startswith("RoboCOIN/"):
        repo_id = repo_id[len("RoboCOIN/") :]
    return _REPO_ID_TO_INDEX[repo_id]
