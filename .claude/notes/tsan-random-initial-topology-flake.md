# The TSan CI job is intermittently killed on one ablation test

The `sanitize (thread, tsan)` job in `ci.yml` sometimes fails with
`Process completed with exit code 143` (SIGTERM) and **no ThreadSanitizer report**. Every
other test has passed by then, and exactly one has started but never finished:

```text
fast.Random Initial Topology: structure is not frozen - mutate() can still grow the genome beyond the initial seed
```

Seen on `main` (the merge of #111, run 33487586720, 2026-09-01) and on PR #117 (run 36559196254,
2026-09-29), whose diff adds code nothing calls. The same job passed on the v0.3.0 release
commit, so it is intermittent.

Read it as a slow or hanging test under TSan's roughly 5-15x slowdown, not as a data race: with
`halt_on_error=1` a race would print a `WARNING: ThreadSanitizer` block and stop, and there is
none. Before treating a TSan failure on a PR as a regression, check which test was still
running. If it is this one, re-run the job.

To find the unfinished test in a downloaded log, subtract the finished tests from the started
ones:

```bash
grep -oE "Start +[0-9]+: .*" log | sed -E 's/Start +([0-9]+): /\1 /' | sort -n > started
grep -oE "Test +#[0-9]+:" log | grep -oE "[0-9]+" | sort -n > done
awk 'NR==FNR{d[$1]=1;next} !($1 in d)' done started
```
