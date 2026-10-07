#!/usr/bin/env bash
#
# Run one long job on another machine: detached, locked per job, logged to a file.
#
#   remote_build.sh run    <host> [options] -- <command>
#   remote_build.sh status <host> [options]
#   remote_build.sh logs   <host> [options] [--lines N]
#   remote_build.sh kill   <host> [options]
#   remote_build.sh list   <host> [options]
#
# <host> is any ssh destination your own ssh configuration already resolves -- the same
# string you would type after `ssh`. This script **never touches a credential**: it holds
# no key, asks for no password, passes no identity file, and reads no ssh config. It shells
# out to `ssh <host> ...` and whatever that does on its own is what happens. Run
# `ssh <host> true` once first; if that works, this works.
#
# Nothing about any particular machine is compiled in -- no address, no account, no path
# outside the remote's own home. Hosts and directories are arguments.
#
# options
#   --dir D     working directory on the remote (default: the remote's own $HOME)
#   --tag T     job name, `[A-Za-z0-9._-]+`; default is derived from the command. Reusing a
#               tag reuses that job's directory, so its logs are the previous run's.
#   --wait      (run) block until the job finishes and exit with its status
#   --lines N   (logs) trailing lines to print (default 40)
#   --dry-run   print the remote script that would be sent, then stop
#   -h, --help  this text
#
# The command runs under `bash -c` on the remote, so quote it as one argument if it needs
# shell syntax:
#
#   remote_build.sh run build-box --dir /srv/src -- 'cmake --build build -j && ctest -j'
#
# Why this shape
# --------------
# A build or a test suite is minutes to an hour. Three things go wrong with the obvious
# approaches, and this exists to avoid all three:
#
#   * foreground in an ssh call -- a dropped connection kills the job;
#   * `nohup ... &` inside a one-shot ssh command -- the process group can still be
#     signalled when the channel closes, and whether it survives is luck;
#   * two jobs at once on one machine -- they contend for the same cores, which presents
#     as "this machine is broken" rather than "you ran two".
#
# So: detach into a new session so no signal reaches it; take a per-job lock so a second
# run refuses (exit 75) instead of racing; and write the log and the final status to files
# so the result is readable long after the connection is gone.
#
# The lock is `flock` where available and a `mkdir` otherwise. It is released when the
# process exits, however it exits.

set -euo pipefail

PROG=${0##*/}
JOBS_DIR_REL=.remote-jobs

say() { printf '%s: %s\n' "$PROG" "$*" >&2; }
die() { say "$*"; exit 2; }

usage() { sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------- arguments

ACTION=
HOST=
DIR=
TAG=
WAIT=0
LINES=40
DRY=0

need_arg() { [ "$2" -ge 2 ] || die "$1 needs a value"; }

while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    run|status|logs|kill|list) ACTION=$1 ;;
    --dir) need_arg --dir $#; DIR=$2; shift ;;
    --tag) need_arg --tag $#; TAG=$2; shift ;;
    --lines) need_arg --lines $#; LINES=$2; shift ;;
    --wait) WAIT=1 ;;
    --dry-run) DRY=1 ;;
    --) shift; break ;;
    -*) die "unknown option $1 (see --help)" ;;
    *) if [ -z "$HOST" ]; then HOST=$1; else CMD="${CMD:+$CMD }$1"; fi ;;
  esac
  shift
done
while [ $# -gt 0 ]; do CMD="${CMD:+$CMD }$1"; shift; done

[ -n "$ACTION" ] || die "no action: one of run, status, logs, kill, list (see --help)"
[ -n "$HOST" ] || die "$ACTION needs a host (an ssh destination)"

# The tag is the job's directory name, so it has to be a name and not a path.
case ${TAG:-x} in
  *[!A-Za-z0-9._-]*) die "--tag must be [A-Za-z0-9._-]+, got '$TAG'" ;;
esac
if [ -z "$TAG" ]; then
  if [ -n "${CMD:-}" ]; then
    first=${CMD%% *}
    first=$(printf '%s' "$first" | tr -c 'A-Za-z0-9._-' '-')
    TAG="${first%-}-$(printf '%s' "$CMD" | cksum | tr -d ' ' | cut -c1-8)"
  else
    TAG=job
  fi
fi

# ---------------------------------------------------------------- the remote side

# Every variable the remote script needs is written with `printf %q`, which is bash's own
# "quote this so it reads back as one word" -- so a working directory with a space, or a
# command with a quote in it, survives the trip as itself.
remote_q() { printf '%q' "$1"; }

# The job directory, spelled for the *remote* shell: the string carries a literal `$HOME`
# so the expansion happens on the far side, where the home directory actually is. Quoting
# it instead freezes this machine's answer into the path -- which is how a first cut here
# created a directory literally named `${HOME:-.}` on the remote.
remote_job_dir() { printf '%s' "\$HOME/$JOBS_DIR_REL/$TAG"; }

# The generated script. Kept as a heredoc with no expansion of its own body (`<<'REMOTE'`)
# so that every `$` inside it belongs to the remote shell, not to this one.
build_remote_script() {
  cat <<'REMOTE'
#!/usr/bin/env bash
# Generated by remote_build.sh. Safe to delete; nothing reads it again.
set -uo pipefail

JOB_DIR="__JOB_DIR__"
WORKDIR=__WORKDIR__
TAG=__TAG__
CMD=__CMD__

LOG=$JOB_DIR/log
STATUS=$JOB_DIR/status
PIDFILE=$JOB_DIR/pid
LOCK=$JOB_DIR/lock

mkdir -p "$JOB_DIR" || exit 91
printf '%s\n' "$$" > "$PIDFILE"

if [ -n "$WORKDIR" ] && [ "$WORKDIR" != "-" ]; then
  if ! cd "$WORKDIR" 2>/dev/null; then
    printf 'cannot cd to %s\n' "$WORKDIR" >> "$LOG"
    printf '90\n' > "$STATUS"
    exit 90
  fi
fi

LOCK_KIND=
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$LOCK" || { printf '91\n' > "$STATUS"; exit 91; }
  if ! flock -n 9; then
    printf 'refused: another job holds %s\n' "$LOCK" >> "$LOG"
    printf '75\n' > "$STATUS"
    exit 75
  fi
  LOCK_KIND=flock
else
  if ! mkdir "$LOCK.d" 2>/dev/null; then
    printf 'refused: another job holds %s.d\n' "$LOCK" >> "$LOG"
    printf '75\n' > "$STATUS"
    exit 75
  fi
  LOCK_KIND=mkdir
fi

# A killed job must not be recorded as a success. Without these two traps, `bash -c "$CMD"`
# dying on a signal leaves the shell to reach its EXIT trap with `$?` already 0, and the
# status file then says "finished exit=0" for work that was terminated halfway.
SIGNALLED=
finish() {
  rc=$?
  [ -n "$SIGNALLED" ] && rc=$SIGNALLED
  [ "$LOCK_KIND" = mkdir ] && rmdir "$LOCK.d" 2>/dev/null
  printf '%s\n' "$rc" > "$STATUS"
  if [ -n "$SIGNALLED" ]; then
    printf '=== %s killed (status %s) at %s ===\n' "$TAG" "$rc" "$(date -Is 2>/dev/null || date)" >> "$LOG"
  else
    printf '=== %s exit=%s at %s ===\n' "$TAG" "$rc" "$(date -Is 2>/dev/null || date)" >> "$LOG"
  fi
  rm -f "$PIDFILE"
  exit "$rc"
}
trap 'SIGNALLED=143; exit 143' TERM
trap 'SIGNALLED=130; exit 130' INT
trap finish EXIT

printf '=== %s start=%s host=%s dir=%s\n' \
  "$TAG" "$(date -Is 2>/dev/null || date)" "$(hostname 2>/dev/null || echo -)" "$PWD" >> "$LOG"
printf '=== command: %s\n' "$CMD" >> "$LOG"
bash -c "$CMD" >> "$LOG" 2>&1
REMOTE
}

render_remote_script() {
  local job_dir workdir script
  job_dir=$(remote_job_dir)
  workdir=${DIR:--}
  script=$(build_remote_script)
  # `__JOB_DIR__` is spliced *raw* on purpose. It is `$HOME/.remote-jobs/<tag>` and the
  # `$HOME` has to reach the remote unexpanded so the far side resolves its own home --
  # `printf %q` would escape the `$` and the script would then create a directory literally
  # named `$HOME`. It is safe to splice unquoted because the tag is validated to
  # `[A-Za-z0-9._-]+` and the rest is a constant.
  #
  # The other three go through `printf %q`, and via bash's own substitution rather than
  # `sed`: `%q` quotes with backslashes and a `sed` replacement eats them (`\ ` becomes a
  # bare space), which would silently split a path containing a space into two arguments.
  script=${script//__JOB_DIR__/$job_dir}
  script=${script//__WORKDIR__/$(remote_q "$workdir")}
  script=${script//__TAG__/$(remote_q "$TAG")}
  script=${script//__CMD__/$(remote_q "${CMD:-true}")}
  printf '%s\n' "$script"
}

# ---------------------------------------------------------------- actions

do_run() {
  [ -n "${CMD:-}" ] || die "run needs a command: remote_build.sh run $HOST -- <command>"
  local script
  script=$(render_remote_script)

  if [ "$DRY" = 1 ]; then
    printf '%s\n' "$script"
    return 0
  fi

  # Staged through a file on the remote and then run by path: sending a script by
  # redirection is exact, where a long quoted argument through the ssh channel is a
  # quoting problem waiting to happen.
  #
  # `$job_dir` holds a literal `$HOME` (see `remote_job_dir`), and it is interpolated into
  # the remote command inside **double** quotes so the far side expands it. Single quotes
  # here would hand the remote the four characters `$HOME` and it would create a directory
  # by that name -- which is exactly what a first cut of this did.
  local job_dir runner
  job_dir=\$HOME/$JOBS_DIR_REL/$TAG
  runner=$job_dir/runner.sh
  ssh "$HOST" "mkdir -p \"$job_dir\" && cat > \"$runner\" && chmod +x \"$runner\"" \
    <<<"$script" || die "could not stage the runner on $HOST"

  # `setsid` puts it in its own session so that closing the ssh channel cannot signal it;
  # if it is missing, `nohup` plus a closed stdin is the next best thing. The subshell is
  # for shells that would otherwise wait on the background job before exiting.
  #
  # The launcher then looks once at `status` before reporting. A job that could not take
  # the lock writes 75 and is gone within that second, and without this check `run` would
  # say "started" for a job that never ran -- a refusal the caller cannot see is worse than
  # no lock at all.
  local out
  out=$(ssh "$HOST" "cd \"$job_dir\" && \
    ( if command -v setsid >/dev/null 2>&1; then setsid nohup bash \"$runner\" >/dev/null 2>&1 </dev/null & \
      else nohup bash \"$runner\" >/dev/null 2>&1 </dev/null & fi ) ; \
    sleep 1; \
    if [ -f \"$job_dir/status\" ]; then printf 'EXIT=%s' \"\$(cat \"$job_dir/status\")\"; \
    else printf 'STARTED'; fi") || die "could not start the job on $HOST"

  case $out in
    STARTED)
      say "started tag=$TAG on $HOST"
      say "  logs:   $PROG logs $HOST --tag $TAG"
      say "  status: $PROG status $HOST --tag $TAG"
      if [ "$WAIT" = 1 ]; then do_wait; fi
      return 0
      ;;
    EXIT=75)
      say "refused: another job on $HOST holds the lock for tag '$TAG'."
      say "  Use a different --tag, or wait for the running job: $PROG status $HOST --tag $TAG"
      return 75
      ;;
    EXIT=*)
      # The job finished inside the one-second probe -- normal for a short command, and
      # also what a job that could not start looks like (a bad --dir is 90; a command that
      # died on a signal is 143). Either way the status is its status.
      local code=${out#EXIT=}
      if [ "$code" = 0 ]; then
        say "job tag=$TAG finished with status 0"
      else
        say "job tag=$TAG already finished with status $code"
        say "  (90 means it could not start -- check --dir; 143 means it was signalled)"
        say "  Its log: $PROG logs $HOST --tag $TAG"
      fi
      return "$code"
      ;;
    *) say "unexpected reply from $HOST: $out"; return 2 ;;
  esac
}

do_wait() {
  # Waits on the remote so the poll costs no repeated handshakes. If the channel drops the
  # job keeps running; only the waiting stops, which is why not passing --wait is fine too.
  local rc
  rc=$(ssh "$HOST" "d=\$HOME/$JOBS_DIR_REL/$TAG; n=0; \
    while [ ! -f \"\$d/status\" ] && [ \$n -lt 14400 ]; do sleep 5; n=\$((n+1)); done; \
    cat \"\$d/status\" 2>/dev/null || echo 124")
  rc=$(printf '%s' "$rc" | tr -dc 0-9)
  [ -n "$rc" ] || rc=124
  if [ "$rc" = 75 ]; then
    say "refused: another job on $HOST holds the lock for this tag"
    return 75
  fi
  [ "$rc" != 0 ] && say "job exited $rc; see $PROG logs $HOST --tag $TAG"
  return "$rc"
}

do_status() {
  ssh "$HOST" "d=\$HOME/$JOBS_DIR_REL/$TAG; \
    if [ -f \"\$d/status\" ]; then printf 'finished exit=%s\n' \"\$(cat \"\$d/status\")\"; \
    elif [ -f \"\$d/pid\" ]; then printf 'running pid=%s\n' \"\$(cat \"\$d/pid\")\"; \
    elif [ -d \"\$d\" ]; then printf 'no status and no pid: not running\n'; \
    else printf 'no such job\n'; fi" || die "could not read status from $HOST"
}

do_logs() {
  ssh "$HOST" "tail -n $LINES \$HOME/$JOBS_DIR_REL/$TAG/log 2>/dev/null \
    || echo 'no log for tag $TAG'" || die "could not read the log from $HOST"
}

do_kill() {
  # The remote script runs under `setsid`, so its pid is its process-group id and a
  # negative pid signals the whole group -- the build and everything it spawned.
  ssh "$HOST" "d=\$HOME/$JOBS_DIR_REL/$TAG; \
    if [ ! -f \"\$d/pid\" ]; then printf 'nothing to kill for tag $TAG\n'; exit 0; fi; \
    p=\$(cat \"\$d/pid\"); \
    if kill -TERM -\$p 2>/dev/null || kill -TERM \$p 2>/dev/null; then \
      printf 'signalled pid %s\n' \"\$p\"; \
    else printf 'pid %s is already gone\n' \"\$p\"; rm -f \"\$d/pid\"; fi" \
    || die "could not signal the job on $HOST"
}

do_list() {
  ssh "$HOST" "r=\$HOME/$JOBS_DIR_REL; \
    if [ ! -d \"\$r\" ]; then printf 'no jobs\n'; exit 0; fi; \
    for d in \"\$r\"/*/; do [ -d \"\$d\" ] || continue; \
      t=\$(basename \"\$d\"); \
      if [ -f \"\$d/status\" ]; then s=\"finished exit=\$(cat \"\$d/status\")\"; \
      elif [ -f \"\$d/pid\" ]; then s=\"running pid=\$(cat \"\$d/pid\")\"; \
      else s=stale; fi; \
      printf '%-32s %s\n' \"\$t\" \"\$s\"; \
    done" || die "could not list jobs on $HOST"
}

case $ACTION in
  run) do_run ;;
  status) do_status ;;
  logs) do_logs ;;
  kill) do_kill ;;
  list) do_list ;;
esac