#!/bin/bash
# Login-node helper (no CPU partition on this allocation): wait for T1 job $1 to finish, and if it
# COMPLETED, build the T3 labeling pages (~1 min CPU). Run detached:
#   setsid nohup bash smoke/env/wait_t1_then_pages.sh <T1_JOBID> > logs/t3_pages.log 2>&1 < /dev/null &
J=$1
cd $PROJECT/rl-lighting-smoke
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 120; done
st=$(sacct -n -X -j $J -o State | head -1 | tr -d ' ')
echo "$(date) T1 job $J: $st"
[ "$st" = COMPLETED ] || exit 1
source smoke/env/psc_env.sh
module load anaconda3/2024.10-1
eval "$(conda shell.bash hook)"; conda activate $PROJECT/envs/gen
nice python smoke/label/make_label_pages.py --manifest smoke/results/T1/manifest_40steps.jsonl --root smoke/results/T1
ls -la smoke/label/
echo PAGES_DONE
