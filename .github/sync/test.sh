#!/usr/bin/env bash
# Runs upstream-sync.sh against scratch repositories and a stand-in for gh.
#   bash .github/sync/test.sh

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t
export GIT_CONFIG_NOSYSTEM=1 HOME=$t

# gh, over a JSON file of issues
mkdir "$t/bin"
cat > "$t/bin/gh" <<'GH'
#!/usr/bin/env python3
import json, os, sys
path = os.environ["GH_STATE"]
issues = json.load(open(path)) if os.path.exists(path) else []
a = sys.argv[1:]
def opt(name):
    return a[a.index(name) + 1] if name in a else None
def issue(n):
    return next(i for i in issues if i["number"] == int(n))
cmd = a[1]
if cmd == "list":
    print(json.dumps([{k: i[k] for k in ("number", "title", "body")} for i in issues if i["state"] == "open"]))
elif cmd == "create":
    issues.append({"number": len(issues) + 1, "title": opt("--title"), "body": open(opt("--body-file")).read(), "state": "open", "comments": []})
elif cmd == "view":
    print(issue(a[2])["body"])
elif cmd == "edit":
    issue(a[2])["body"] = open(opt("--body-file")).read()
elif cmd == "comment":
    issue(a[2])["comments"].append(opt("--body"))
elif cmd == "close":
    issue(a[2])["state"] = "closed"
    issue(a[2])["comments"].append(opt("--comment"))
json.dump(issues, open(path, "w"))
GH
chmod +x "$t/bin/gh"
export PATH="$t/bin:$PATH" GH_STATE=$t/issues.json

git init -q --bare "$t/upstream.git"
git init -q --bare "$t/origin.git"
git init -q -b master "$t/w"
w() { git -C "$t/w" "$@"; }
commit() { w commit -q --allow-empty -m "$1"; w rev-parse HEAD; }

c1=$(commit one)
w tag -a v1 -m v1
w push -q "$t/upstream.git" master v1
w push -q "$t/origin.git" master v1
w checkout -q -b downstream
d1=$(commit patch)
w push -q "$t/origin.git" downstream
w tag fork-only && w push -q "$t/origin.git" fork-only
w checkout -q master

export UPSTREAM_URL=$t/upstream.git ORIGIN_URL=$t/origin.git GH_REPO=o/r
sync() { GITHUB_STEP_SUMMARY=$t/summary "$here/upstream-sync.sh" 2> "$t/stderr"; }
ref() { git --git-dir="$t/origin.git" rev-parse "$1"; }
field() { sed -n "s/^$1: *//p" "$t/summary"; }
open_sync_issues() { jq '[.[] | select(.state == "open" and (.body | contains("bart-upstream-sync")))] | length' "$GH_STATE"; }
fail() { echo "FAIL: $*"; cat "$t/summary" "$t/stderr" 2> /dev/null; exit 1; }
pass() { echo "ok   $*"; }

# An unrelated issue with the same title, which the sync must never close.
echo '[{"number": 1, "title": "Sync downstream with current Codeberg master", "body": "by hand", "state": "open", "comments": []}]' > "$GH_STATE"

: > "$t/summary"; sync || fail "identical run failed"
[ "$(field 'master updated')" = no ] && [ "$(ref master)" = "$c1" ] || fail "identical: master moved"
[ "$(field 'needs rebase')" = no ] && [ "$(open_sync_issues)" = 0 ] || fail "identical: issue opened"
pass "1. upstream and master identical: no push, no issue"

c2=$(commit two); c3=$(commit three)
w tag v2 && w push -q "$t/upstream.git" master v2
: > "$t/summary"; sync || fail "fast-forward run failed"
[ "$(ref master)" = "$c3" ] && [ "$(field 'master updated')" = yes ] || fail "fast-forward: master not updated"
[ "$(ref downstream)" = "$d1" ] || fail "fast-forward: downstream touched"
pass "2. upstream ahead: master fast-forwarded to exactly Codeberg's commit"
[ "$(ref 'v2^{}')" = "$c3" ] && [ "$(field 'upstream tags added')" = 1 ] || fail "new tag not mirrored"
pass "7. new upstream tag mirrored"
[ "$(field 'needs rebase')" = "yes (2 commits behind master)" ] || fail "rebase not detected"
[ "$(open_sync_issues)" = 1 ] || fail "issue not opened"
grep -q "$d1" "$GH_STATE" && grep -q "$c3" "$GH_STATE" || fail "issue lacks the SHAs"
pass "5. downstream lacks master: one issue opened"

: > "$t/summary"; sync || fail "repeat run failed"
[ "$(open_sync_issues)" = 1 ] || fail "repeat: duplicate issue"
[ "$(jq '.[1].comments | length' "$GH_STATE")" = 0 ] || fail "repeat: commented without news"
c4=$(commit four) && w push -q "$t/upstream.git" master
: > "$t/summary"; sync || fail "second advance failed"
[ "$(open_sync_issues)" = 1 ] && [ "$(jq '.[1].comments | length' "$GH_STATE")" = 1 ] || fail "advance: not one comment on one issue"
grep -q "$c4" "$GH_STATE" || fail "advance: issue body not updated"
pass "6. repeated runs: still one issue, a comment only when master moved"

w checkout -q downstream && w rebase -q master && d2=$(w rev-parse HEAD)
w push -q -f "$t/origin.git" downstream && w checkout -q master
: > "$t/summary"; sync || fail "rebased run failed"
[ "$(field 'needs rebase')" = no ] && [ "$(open_sync_issues)" = 0 ] || fail "sync issue not closed"
[ "$(jq -r '.[0].state' "$GH_STATE")" = open ] || fail "unrelated issue closed"
pass "4. downstream contains master: the sync issue closed, the unrelated one left open"

w tag -f -a v1 -m moved "$c2" > /dev/null && w push -q -f "$t/upstream.git" v1
v1=$(ref v1)
: > "$t/summary"; if sync; then fail "tag conflict did not fail"; fi
[ "$(ref v1)" = "$v1" ] && [ "$(ref fork-only)" = "$d1" ] || fail "tag rewritten"
grep -q 'tag differs' "$t/stderr" || fail "tag conflict not reported"
pass "8. conflicting tag: not overwritten, run fails, fork-only tag kept"
w push -q -f "$t/upstream.git" "$v1:refs/tags/v1"

w reset -q --hard "$c3" && c5=$(commit rewritten)
w push -q -f "$t/upstream.git" master
: > "$t/summary"; if sync; then fail "rewritten upstream did not fail"; fi
[ "$(ref master)" = "$c4" ] || fail "rewritten: master moved"
grep -q "$c5" "$t/stderr" && grep -q "$c4" "$t/stderr" || fail "rewritten: SHAs not reported"
pass "3. rewritten upstream: update refused, both SHAs reported, master untouched"
