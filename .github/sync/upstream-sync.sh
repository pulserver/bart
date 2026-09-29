#!/usr/bin/env bash
# Fast-forward the fork's master to Codeberg's, carry over Codeberg's new tags,
# and keep one issue open while downstream does not contain master.
#
# Nothing here forces a ref: master moves only by a verified fast-forward,
# a tag is only ever created, and downstream is only read.
#
# Environment:
#   UPSTREAM_URL   canonical repository (Codeberg)
#   ORIGIN_URL     the fork
#   GH_TOKEN       token for pushing to a github.com ORIGIN_URL and for gh
#   GH_REPO        owner/name, for gh and the links in the issue
#   ISSUE          "on" (default) to maintain the issue through gh, "off" to skip
#   GITHUB_STEP_SUMMARY, GITHUB_SERVER_URL, GITHUB_RUN_ID   set by Actions

set -euo pipefail

UPSTREAM_URL=${UPSTREAM_URL:-https://codeberg.org/mrirecon/bart.git}
ORIGIN_URL=${ORIGIN_URL:?ORIGIN_URL is required}
GH_REPO=${GH_REPO:-pulserver/bart}
ISSUE=${ISSUE:-on}
SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/stdout}

ISSUE_TITLE="Sync downstream with current Codeberg master"
ISSUE_MARKER="<!-- bart-upstream-sync -->"

die() { echo "::error::$*" >&2; exit 1; }

retry() {
	local n
	for n in 1 2 3; do
		"$@" && return 0
		[ "$n" = 3 ] || sleep $((n * 10))
	done
	return 1
}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q --bare "$work/repo"
export GIT_DIR="$work/repo"
git remote add upstream "$UPSTREAM_URL"
git remote add origin "$ORIGIN_URL"
if [ -n "${GH_TOKEN:-}" ] && [[ "$ORIGIN_URL" == https://github.com/* ]]; then
	auth=$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)
	echo "::add-mask::$auth"
	git config http.https://github.com/.extraheader "AUTHORIZATION: basic $auth"
fi

# Everything is fetched into refs/sync/, so no ref of one remote can shadow
# a ref of the other.
retry git fetch -q --no-tags upstream \
	'+refs/heads/master:refs/sync/upstream/master' \
	'+refs/tags/*:refs/sync/upstream-tags/*' \
	|| die "cannot fetch master and tags from $UPSTREAM_URL"
retry git fetch -q --no-tags origin \
	'+refs/heads/master:refs/sync/origin/master' \
	'+refs/heads/downstream:refs/sync/origin/downstream' \
	|| die "cannot fetch master and downstream from the fork"

upstream=$(git rev-parse --verify -q refs/sync/upstream/master^{commit}) \
	|| die "Codeberg master cannot be resolved"
master_old=$(git rev-parse --verify -q refs/sync/origin/master^{commit}) \
	|| die "the fork's master cannot be resolved"

# --- master ---------------------------------------------------------------

master_updated=no
if [ "$master_old" = "$upstream" ]; then
	:
elif git merge-base --is-ancestor "$master_old" "$upstream"; then
	# A plain push: the server refuses anything but a fast-forward as well.
	git push -q origin "$upstream:refs/heads/master" \
		|| die "the fork refused the fast-forward of master $master_old -> $upstream"
	master_updated=yes
else
	{
		echo "## Upstream sync refused"
		echo
		echo "Codeberg master is not a descendant of the fork's master: upstream history appears rewritten."
		echo
		echo '```'
		echo "Codeberg master:       $upstream"
		echo "pulserver master:      $master_old"
		echo '```'
		echo
		echo "master was left untouched."
	} >> "$SUMMARY"
	die "upstream history rewritten: Codeberg master $upstream does not contain the fork's master $master_old; master left untouched"
fi
master_new=$upstream

# --- tags -----------------------------------------------------------------

# Tags the fork already has, by name and object.  A tag that exists only in the
# fork is never looked at again.
origin_tags=$(retry git ls-remote --tags origin) || die "cannot list the fork's tags"
declare -A origin_tag
while read -r sha ref; do
	[ -n "$ref" ] || continue
	case "$ref" in *'^{}') continue ;; esac
	origin_tag[${ref#refs/tags/}]=$sha
done <<< "$origin_tags"

missing=()
conflicts=()
while read -r sha ref; do
	name=${ref#refs/sync/upstream-tags/}
	if [ -z "${origin_tag[$name]+set}" ]; then
		missing+=("$ref:refs/tags/$name")
	elif [ "${origin_tag[$name]}" != "$sha" ]; then
		conflicts+=("$name: Codeberg $sha, fork ${origin_tag[$name]}")
	fi
done < <(git for-each-ref --format='%(objectname) %(refname)' refs/sync/upstream-tags/)

if [ "${#missing[@]}" -gt 0 ]; then
	# No '+' and no --force: an existing tag is never replaced.
	git push -q origin "${missing[@]}" || die "the fork refused the new tags"
fi
for c in "${conflicts[@]}"; do
	echo "::error::tag differs between Codeberg and the fork, left as it is: $c" >&2
done

# --- downstream -----------------------------------------------------------

downstream=$(git rev-parse --verify -q refs/sync/origin/downstream^{commit}) \
	|| die "the fork's downstream cannot be resolved"
if git merge-base --is-ancestor "$master_new" "$downstream"; then
	needs_rebase=no
	behind=
else
	needs_rebase=yes
	behind=$(git rev-list --count "$downstream..$master_new")
fi

{
	echo "## Upstream sync"
	echo
	echo '```'
	echo "Codeberg master:       $upstream"
	echo "pulserver master old:  $master_old"
	echo "pulserver master new:  $master_new"
	echo "master updated:        $master_updated"
	echo "upstream tags added:   ${#missing[@]}"
	echo "tag conflicts:         ${#conflicts[@]}"
	echo "downstream:            $downstream"
	echo "needs rebase:          $needs_rebase${behind:+ ($behind commits behind master)}"
	echo '```'
	for c in "${conflicts[@]}"; do
		echo "- tag conflict, left as it is: \`$c\`"
	done
} >> "$SUMMARY"

# --- issue ----------------------------------------------------------------

open_issue() {
	gh issue list --repo "$GH_REPO" --state open --limit 500 --json number,title,body \
		| jq -r --arg t "$ISSUE_TITLE" --arg m "$ISSUE_MARKER" \
			'[.[] | select(.title == $t and (.body | contains($m)))] | sort_by(.number) | .[0].number // empty'
}

issue_body() {
	local run=""
	[ -n "${GITHUB_RUN_ID:-}" ] && run=" ([workflow run](${GITHUB_SERVER_URL:-https://github.com}/$GH_REPO/actions/runs/$GITHUB_RUN_ID))"
	cat <<-BODY
	$ISSUE_MARKER
	<!-- master: $master_new -->
	Codeberg \`master\` has advanced past what \`downstream\` is based on.
	The fork's \`master\` has been fast-forwarded to it automatically$run;
	\`downstream\` was **not** modified.

	| | |
	| --- | --- |
	| \`master\` | \`$master_new\` |
	| \`downstream\` | \`$downstream\` |
	| upstream commits missing from \`downstream\` | $behind |

	[Commits on \`master\` that \`downstream\` lacks](https://github.com/$GH_REPO/compare/$downstream...$master_new)

	To rebase the patch stack:

	\`\`\`bash
	git fetch upstream --tags
	git fetch origin

	git checkout master
	git merge --ff-only upstream/master

	git checkout downstream
	git merge --ff-only origin/downstream
	git tag downstream-\$(date +%Y%m%d) downstream
	git push origin downstream-\$(date +%Y%m%d)

	git rebase master

	# review conflicts and new upstream code carefully
	# run make all && make utest on supported platforms

	git push --force-with-lease origin downstream
	\`\`\`

	Then move bartorch's BART submodule to the new \`downstream\` in a separate bartorch pull request.

	This issue is closed automatically by the next sync run that finds \`master\` contained in \`downstream\`.
	BODY
}

if [ "$ISSUE" = on ]; then
	number=$(open_issue) || die "cannot list the fork's issues"
	if [ "$needs_rebase" = yes ]; then
		issue_body > "$work/body.md"
		if [ -z "$number" ]; then
			gh issue create --repo "$GH_REPO" --title "$ISSUE_TITLE" --body-file "$work/body.md" > /dev/null
			echo "Opened the rebase issue." >> "$SUMMARY"
		else
			previous=$(gh issue view "$number" --repo "$GH_REPO" --json body --jq .body \
				| sed -n 's/^<!-- master: \([0-9a-f]*\) -->$/\1/p')
			gh issue edit "$number" --repo "$GH_REPO" --body-file "$work/body.md" > /dev/null
			if [ "$previous" != "$master_new" ]; then
				gh issue comment "$number" --repo "$GH_REPO" \
					--body "Codeberg master advanced again, to \`$master_new\`; \`downstream\` is $behind commits behind it." > /dev/null
			fi
			echo "Updated rebase issue #$number." >> "$SUMMARY"
		fi
	elif [ -n "$number" ]; then
		gh issue close "$number" --repo "$GH_REPO" \
			--comment "\`downstream\` ($downstream) now contains the current upstream master ($master_new)." > /dev/null
		echo "Closed rebase issue #$number." >> "$SUMMARY"
	fi
fi

[ "${#conflicts[@]}" -eq 0 ] || die "${#conflicts[@]} upstream tag(s) conflict with the fork's; see the summary"
