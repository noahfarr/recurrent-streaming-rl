import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

import wandb

PROJECT = "noahfarr/recurrent-streaming-rl"


def running_run_ids(api, project):
    ids = set()
    for run in api.runs(project, filters={"state": "running"}):
        ids.add(run.id)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--mode", choices=["all", "keep-last"], default="keep-last")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    api = wandb.Api()
    skip = running_run_ids(api, args.project)
    print(f"skipping {len(skip)} running runs", flush=True)

    def delete_one(artifact):
        try:
            artifact.delete(delete_aliases=True)
        except Exception:
            pass

    pool = ThreadPoolExecutor(max_workers=args.workers)
    artifact_type = api.artifact_type("model", project=args.project)
    freed = 0
    deleted = 0
    collections = 0

    for collection in artifact_type.collections():
        run_id = collection.name.removeprefix("model-")
        if run_id in skip:
            continue
        collections += 1
        if args.limit and collections > args.limit:
            break
        if collections % 100 == 0:
            print(
                f"{collections} collections, {deleted} versions, {freed / 1e9:.1f} GB",
                flush=True,
            )

        if args.mode == "all":
            try:
                versions = list(collection.artifacts())
            except Exception:
                continue
            freed += sum(getattr(a, "size", 0) or 0 for a in versions)
            deleted += len(versions)
            if args.execute:
                collection.delete()
        else:
            try:
                versions = list(collection.artifacts())
            except Exception:
                continue
            if len(versions) <= 1:
                continue
            keep = max(versions, key=lambda a: int(a.version.lstrip("v")))
            doomed = [a for a in versions if a.id != keep.id]
            freed += sum(getattr(a, "size", 0) or 0 for a in doomed)
            deleted += len(doomed)
            if args.execute:
                list(pool.map(delete_one, doomed))



    pool.shutdown(wait=True)
    verb = "deleted" if args.execute else "would delete"
    print(f"{verb} {deleted} versions across {collections} collections, {freed / 1e9:.1f} GB")
    if not args.execute:
        print("dry run, pass --execute to apply")


if __name__ == "__main__":
    sys.exit(main())
