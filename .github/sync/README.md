# Keeping the fork in step with Codeberg

BART is developed at [codeberg.org/mrirecon/bart](https://codeberg.org/mrirecon/bart),
which is canonical; the archived GitHub repository `mrirecon/bart` is not used.
This fork has two branches that matter:

| Branch | Contents | Moved by |
| --- | --- | --- |
| `master` | Codeberg's `master`, exactly; never committed to | the *Upstream sync* workflow |
| `downstream` | `master` plus the reviewed portability and embedding patches; the default branch | a person, by rebase and `--force-with-lease` |

## What the workflow does

`.github/workflows/upstream-sync.yml` runs `upstream-sync.sh` every Monday at
04:17 UTC and on demand (*Actions > Upstream sync > Run workflow*).  It

1. fetches Codeberg's `master` and tags;
2. pushes Codeberg's `master` to this `master` when that is a fast-forward, and
   otherwise fails and names both commits, leaving `master` alone;
3. pushes each Codeberg tag this fork lacks, and fails, naming it, on a tag
   both have at different objects, leaving the existing one alone; tags only
   this fork has (`downstream-YYYYMMDD`, `github-mirror-final`) are never
   touched;
4. keeps one issue, *Sync downstream with current Codeberg master*, open while
   `downstream` does not contain `master`, with the commits involved and the
   procedure below, and closes it once a run finds `downstream` rebased.

It never forces a ref and never writes `downstream`.  Each run's summary gives
both `master` commits, the tags added, `downstream` and whether it needs a
rebase.

`bash .github/sync/test.sh` runs the script against scratch repositories.

## Rebasing downstream

```bash
git remote add upstream https://codeberg.org/mrirecon/bart.git   # once
git fetch upstream --tags
git fetch origin

git checkout master
git merge --ff-only upstream/master

git checkout downstream
git merge --ff-only origin/downstream
git tag downstream-$(date +%Y%m%d) downstream
git push origin downstream-$(date +%Y%m%d)

git rebase master
# resolve each conflict explicitly; drop a patch upstream has made redundant
# make all && make utest, and the Portability workflow on every platform

git push --force-with-lease origin downstream
```

The tag keeps the previous stack, and with it every commit an earlier bartorch
pinned, reachable once `downstream` is rewritten.  bartorch pins an exact
`downstream` commit, so nothing here changes a bartorch build until a bartorch
pull request moves its submodule.

`master` is never the base of a pull request here: a merge into it makes it
something other than Codeberg's, and the workflow then stops until `master` is
put back on Codeberg's commit by hand.
