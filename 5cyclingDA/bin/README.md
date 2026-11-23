# LSF array-state wrappers (plan B)

These wrappers work around an LSF 10.1 output quirk on lililei1:

- `bjobs -a -noheader -o "jobid stat" <array_id>` returns the **base** job ID
  without the `[member]` suffix;
- completed array elements are purged from `bjobs`;
- `bhist -l "<array_id>[<member>]"` is very slow for individual elements.

`run_driver_50mem_cyclingDA.sh` expects `jobid[member] state`, so its
`scheduler.wait()` gives up after five unknown-state polls.  The wrappers
reconstruct per-element states from `bhist -l <array_id>` using the member list
in `$RUN_ROOT/*/*/jobs/*/attempt_*/lsf_job.json` and cache them.

If LSF has already purged the array from both `bjobs` and `bhist`, the wrapper
falls back to the filesystem: `run_wrf/<cycle>/<member>/wrf_done` or
`wrfout/<cycle>/<member>/member_done.json` means `DONE`; a missing marker on a
terminal/purged array is treated as `EXIT` so the driver retries that member.

## Files

- `bjobs` - intercepts the driver's array query; non-array calls pass through.
- `bhist` - answers the driver's per-element fallback from the cache; other
  calls pass through.
- `lsf_state_cache.py` - shared parser/cache helpers.
- `submit_50mem_cyclingDA.sh` (in the parent directory) prepends this `bin`
  directory to `PATH` before `exec bash "$DRIVER"`.

The driver itself is **not modified**, so the experiment signature and
`RESUME=1` checkpoints remain valid.

## Deployment

```bash
scp -r 5cyclingDA/bin \
  lililei1:/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/
scp 5cyclingDA/submit_50mem_cyclingDA.sh \
  lililei1:/share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/
ssh lililei1 'chmod +x /share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA/bin/{bjobs,bhist}'
```

The submit script now contains:

```bash
export PATH="${SCRIPT_DIR}/bin:${PATH}"
```

## Verification

Local tests (from the repository root):

```bash
/Users/kcfu/miniforge3/bin/python -m unittest discover \
  -s 5cyclingDA/tests -p 'test_bjobs_wrapper.py' -v
```

On the server, after deployment, a non-destructive check:

```bash
cd /share/home/lililei1/kcfu/tc_mangkhut/5cyclingDA
RUN_ROOT=/scratch/lililei1/kcfu/tc_mangkhut/5cyclingDA/50mem_hourly_d01hx \
  bin/bjobs -a -noheader -o "jobid stat" 20776452 | head
```

Expected: 50 lines of `20776452[member] DONE|EXIT`, not just base IDs.
