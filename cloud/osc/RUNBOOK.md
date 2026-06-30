# OSC run-from-scratch runbook — T6.2 LLM extraction

The complete, ordered procedure with every OSC-specific gotcha we hit folded in. Two
environments matter, and mixing them up is the root of most errors:

- **Login / transfer node** (`ascend.osc.edu` etc.) — **has internet**. Do clone / build /
  weight-staging / file-transfer here.
- **Compute (GPU) node** — **offline**, reached via `sbatch` or an interactive session. Runs
  the extraction against cached weights.

---

## 0. Prerequisites — confirm GPU access FIRST (the easy-to-miss step)

Login access to a cluster is **not** the same as permission to submit jobs on it. You need a
project with a **GPU allocation on the specific cluster** you use.

```bash
# on the cluster's login node — lists every account/partition you may submit to:
sacctmgr -nP show assoc user=$USER format=cluster,account,partition,qos | sort -u
```
- **Non-empty** (e.g. `ascend|PAS0541|gpu|...`) → you can submit here; note the partition name.
- **Empty** → you have **no** submit rights on this cluster (this is what happened on Cardinal).
  Try another cluster or check my.osc.edu → Projects → PAS0541 → Allocations.

**Recommended cluster: Ascend (A100-80GB).** The package targets it; 7B and 32B each fit on one
GPU. Use Cardinal (H100) only if `sacctmgr` shows you an association there. **Do every step
below on that one cluster's login node** (weights live on per-cluster scratch — don't mix).

If the chosen cluster's partition isn't the default, note it now — you'll pass it in §5.

---

## 1. Code — one-time (`$HOME` is shared across OSC clusters)

```bash
cd ~ && git clone <repo-url> Earnings_call_project-main   # or `git pull` if already cloned
cd ~/Earnings_call_project-main
git log --oneline -1     # must be at/after the v2 schema + YaRN commits
```
Because `$HOME` is shared, if you already cloned on another OSC cluster the repo is already here.

---

## 2. Data — the two gitignored parquets (one-time, from your workstation)

Extraction reads only `chunks.parquet` (calls.parquet is not needed; `data/splits/*.csv` are
git-tracked and arrive with the clone). From the project root **on your workstation**:

```powershell
# Windows: scp is built in. <osc_user> is your personal login, NOT the project code.
ssh <osc_user>@ascend.osc.edu "mkdir -p ~/Earnings_call_project-main/data/fincall ~/Earnings_call_project-main/data/maec"
scp data/fincall/chunks.parquet <osc_user>@ascend.osc.edu:~/Earnings_call_project-main/data/fincall/
scp data/maec/chunks.parquet    <osc_user>@ascend.osc.edu:~/Earnings_call_project-main/data/maec/
```
Verify on OSC (catch a truncated copy): `ls -la data/fincall/chunks.parquet data/maec/chunks.parquet`
(expect ~138M and ~46M).

---

## 3. Build the container — login node, ~10 min, needs internet

Apptainer is on the PATH directly — **do NOT `module load apptainer`** (no such module on OSC).

```bash
cd ~/Earnings_call_project-main
apptainer build cloud/osc/ecvol-llm.sif cloud/osc/apptainer/ecvol-llm.def
```
The `.sif` is portable across OSC clusters (all x86_64 + NVIDIA), so build it once.
- If it errors about **fakeroot** or **/tmp space**, point the build at scratch and retry:
  ```bash
  export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp APPTAINER_CACHEDIR=$SCRATCH/apptainer_cache
  mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
  apptainer build --fakeroot cloud/osc/ecvol-llm.sif cloud/osc/apptainer/ecvol-llm.def
  ```
- Sanity check the CLI resolves in the image (no internet needed):
  `apptainer exec cloud/osc/ecvol-llm.sif ecvol --help | head -3`

---

## 4. Stage model weights — login node, needs internet

OSC defines no `$SCRATCH`; set it to your project scratch. `stage.sh` already (a) defaults
`$SCRATCH` and (b) unsets the image's offline flag and uses `python3` for the download.

```bash
export SCRATCH=/fs/scratch/PAS0541/$USER && mkdir -p "$SCRATCH"
bash cloud/osc/stage.sh Qwen/Qwen2.5-7B-Instruct          # ~15 GB; start with just 7B
# later, once the 7B run is validated:
bash cloud/osc/stage.sh Qwen/Qwen2.5-32B-Instruct         # ~65 GB
```
Verify: `ls $SCRATCH/ecvol_hf/hub/`  → a `models--Qwen--Qwen2.5-7B-Instruct` dir.

---

## 5. Smoke test — one tiny GPU job (validates the YaRN/vLLM path)

Keep `SCRATCH` exported in this shell (`--export=ALL` carries it into the job).

```bash
sbatch --export=ALL,MODEL_ID=Qwen/Qwen2.5-7B-Instruct,DATASET=fincall,LIMIT=3 cloud/osc/slurm/extract.sbatch
squeue -u $USER
cat cloud/osc/logs/ecvol-llm-*.out
```
- If `sbatch` says **"Invalid account or account/partition combination"**, add the partition
  from §0 to the job: edit `#SBATCH --partition=<name>` into `cloud/osc/slurm/extract.sbatch`
  (or `--partition=<name>` on the `sbatch` line).
- **Interactive alternative** (sidesteps batch routing entirely): grab a GPU node
  (`salloc -A PAS0541 --gpus-per-node=1 -t 1:00:00`, or OnDemand Desktop), then run the
  extraction in the foreground:
  ```bash
  export SCRATCH=/fs/scratch/PAS0541/$USER
  apptainer exec --nv -B "$PWD:/workspace" -B "$SCRATCH:$SCRATCH" \
    --env HF_HOME=$SCRATCH/ecvol_hf --env HF_HUB_OFFLINE=1 \
    cloud/osc/ecvol-llm.sif \
    ecvol featurize llm --dataset fincall --model-id Qwen/Qwen2.5-7B-Instruct \
      --engine vllm --max-model-len 65536 --yarn --limit 3 --root /workspace/data
  ```

Green = `done; features at data/...` with no traceback. A `rope_scaling` / `hf_overrides`
error = the YaRN kwarg form needs a one-line tweak — stop and report it.

Verify the output has the v2 fields:
```bash
apptainer exec cloud/osc/ecvol-llm.sif python3 -c "import pandas as pd; d=pd.read_parquet('data/fincall/llm_features__Qwen__Qwen2.5-7B-Instruct.parquet'); print(d.columns.tolist()); print(len(d),'rows')"
```
Expect ~6 rows and both `management_optimism` + `quantitative_specificity` columns.

---

## 6. Full panel — once the smoke test is clean

```bash
for M in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct; do
  for D in fincall maec; do
    sbatch --export=ALL,MODEL_ID=$M,DATASET=$D cloud/osc/slurm/extract.sbatch
  done
done
```
Jobs are resumable (the per-model parquet is the checkpoint), so a walltime hit + resubmit is
safe. The 50 κ-audit calls are a FinCall subset, so they're extracted by the same job — no
separate audit run.

---

## 7. Results back to your workstation + the κ-gate (local)

```powershell
# on your workstation, project root:
scp "<osc_user>@ascend.osc.edu:~/Earnings_call_project-main/data/fincall/llm_features__*.parquet" data/fincall/
scp "<osc_user>@ascend.osc.edu:~/Earnings_call_project-main/data/maec/llm_features__*.parquet"    data/maec/

# the content gate — gate is on the confirmatory core; weak/exploratory fields reported only:
ecvol llm-kappa --sheet data/coverage/fincall_llm_labels_rater1.csv `
                --features data/fincall/llm_features__Qwen__Qwen2.5-7B-Instruct.parquet
```

---

## Quick gotcha index (everything we hit, and the fix)

| Symptom | Cause | Fix (already in the scripts) |
|---|---|---|
| `module: apptainer unknown` | Apptainer isn't a module on OSC | Don't `module load`; it's on PATH |
| `SCRATCH: unbound variable` | OSC defines no `$SCRATCH` | scripts default it to `/fs/scratch/PAS0541/$USER` |
| `python: not found` | image has `python3`, not `python` | scripts use `python3` |
| `OfflineModeIsEnabled` on download | image bakes `HF_HUB_OFFLINE=1` (right for compute) | `stage.sh` unsets it for the login-node download |
| `Invalid account or account/partition` | no GPU association on that cluster | use a cluster where `sacctmgr` is non-empty (§0); add `--partition` if needed |
