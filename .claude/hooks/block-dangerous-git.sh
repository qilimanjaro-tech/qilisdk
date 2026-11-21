#!/usr/bin/env bash
# qili-pairing git guardrail: a Claude Code PreToolUse hook for the Bash tool.
#
# Blocks git push, reset --hard, clean (all but a dry run), branch -D,
# checkout . and restore . Exit 2 blocks the call and shows the message to the
# agent; exit 0 allows it.
#
# The command is split the way the shell splits it (quotes, ;, &&, |, $(...),
# backticks, heredocs), so a quoted commit message or a file path that mentions
# one of these commands does not match. Git's global options (-C <path>,
# -c key=value, --git-dir, --work-tree, ...) are skipped before the subcommand
# is read, and long options match in the abbreviated forms git accepts
# (--har). Strings run by bash -c, sh -c or eval, and heredocs fed to a shell,
# are checked too. An unquoted word `git` anywhere in a command counts as a
# call, so `sudo git push` and `xargs git push` are caught, and so is
# `echo git push`.
#
# A guardrail against an agent's mistakes, not a sandbox: git aliases, commands
# built from variables ($G push), text piped into a shell and scripts that call
# git are not inspected.
# Written for bash 3.2 (macOS /bin/bash). Tests: test-block-dangerous-git.sh.

export LC_ALL=C
shopt -s extglob

block() {
  local message=$1
  printf 'BLOCKED: %s\n' "$message" >&2
  exit 2
}

deny() {
  local what=$1
  printf 'BLOCKED: %s. The git guardrail keeps agents from pushing or discarding work; ask the person to run it.\n' "$what" >&2
  exit 2
}

command -v jq >/dev/null 2>&1 ||
  block "jq is not on PATH, so the git guardrail cannot read any command and blocks every Bash call, an install command included. Stop and ask the person to install jq (macOS: brew install jq; Debian or Ubuntu: apt install jq)."

input=$(</dev/stdin)
S=$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null) ||
  block "the git guardrail could not read its input as JSON."
[[ -n $S ]] || exit 0

# A pathspec that means the whole tree.
is_all() {
  local spec=$1
  case $spec in
    .|./|:/|:/.) return 0 ;;
    *) return 1 ;;
  esac
}

# True when the word is --name or an abbreviation of it git accepts, at least
# min_len characters long.
is_opt() {
  local word=$1 name=$2 min_len=$3
  [[ ${#word} -ge $min_len ]] || return 1
  case --$name in
    "$word"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Arguments: the words after "git".
check_git() {
  local -a args=("$@")
  local n=${#args[@]} k=0 sub="" a all=0 force=0 delete=0 staged=0 worktree=0 dry=0
  while ((k < n)); do
    case ${args[k]} in
      -C|-c|--git-dir|--work-tree|--namespace|--config-env|--attr-source|--super-prefix)
        k=$((k + 2)) ;;
      -*) k=$((k + 1)) ;;
      *) sub=${args[k]}; k=$((k + 1)); break ;;
    esac
  done
  local -a rest=("${args[@]:k}")
  case $sub in
    push) deny "git push" ;;
    reset)
      for a in "${rest[@]}"; do is_opt "$a" hard 4 && deny "git reset --hard"; done ;;
    clean)
      # clean.requireForce=false (in -c or in any config) deletes without -f,
      # so only a dry run passes.
      for a in "${rest[@]}"; do
        case $a in
          --*) is_opt "$a" dry-run 4 && dry=1 ;;
          -*[!dfinqxX]*) ;;
          -*n*) dry=1 ;;
          *) ;;
        esac
      done
      [[ $dry == 1 ]] || deny "git clean (only a dry run, -n, is allowed)" ;;
    branch)
      for a in "${rest[@]}"; do
        case $a in
          --*) is_opt "$a" delete 4 && delete=1; is_opt "$a" force 4 && force=1 ;;
          -*D*) deny "git branch -D" ;;
          -*)
            [[ $a == -*d* ]] && delete=1
            [[ $a == -*f* ]] && force=1 ;;
          *) ;;
        esac
      done
      [[ $delete == 1 && $force == 1 ]] && deny "git branch -D" ;;
    checkout)
      for a in "${rest[@]}"; do is_all "$a" && deny "git checkout ."; done ;;
    restore)
      for a in "${rest[@]}"; do
        case $a in
          --*) is_opt "$a" staged 4 && staged=1; is_opt "$a" worktree 4 && worktree=1 ;;
          -*)
            [[ $a == -*S* ]] && staged=1
            [[ $a == -*W* ]] && worktree=1 ;;
          *) is_all "$a" && all=1 ;;
        esac
      done
      # restore --staged alone only unstages; anything else rewrites the working tree.
      [[ $all == 1 ]] && [[ $staged == 0 || $worktree == 1 ]] && deny "git restore ." ;;
    *) ;;
  esac
  return 0
}

# Arguments: the words of one simple command.
check_words() {
  local -a a=("$@")
  local i j n=${#a[@]}
  for ((i = 0; i < n; i++)); do
    case ${a[i]} in
      git|*/git) check_git "${a[@]:i+1}" ;;
      bash|sh|zsh|dash|ksh|*/bash|*/sh|*/zsh|*/dash|*/ksh)
        for ((j = i + 1; j < n; j++)); do
          case ${a[j]} in
            --*) ;;
            -o|+o) j=$((j + 1)) ;;
            -*c*) ((j + 1 < n)) && scan_string "${a[j + 1]}"; break ;;
            -*) ;;
            *) break ;;
          esac
        done ;;
      eval) ((i + 1 < n)) && scan_string "${a[*]:i+1}" ;;
      *) ;;
    esac
  done
  return 0
}

# The scanner walks the global string S, of length N, from index I. The helpers
# below change scan's locals (w, inword, words, heredocs): bash scopes locals
# dynamically.
end_word() {
  [[ $inword == 1 ]] && words+=("$w")
  w="" inword=0
  return 0
}

end_command() {
  end_word
  ((${#words[@]} > 0)) && check_words "${words[@]}"
  words=()
  return 0
}

# Sets part to the text from I up to the first character matching the bracket
# pattern, or to the end. Searches 256 bytes at a time: bash's pattern removal
# is slow on long strings.
run_until() {
  local pattern=$1 chunk p
  part=""
  while ((I + ${#part} < N)); do
    chunk=${S:I+${#part}:256}
    p=${chunk%%$pattern*}
    part+=$p
    ((${#p} < ${#chunk})) && return 0
  done
  return 0
}

# Scans another string as commands, then puts back the caller's S, N and I.
scan_string() {
  local text=$1 saved_s=$S saved_n=$N saved_i=$I
  S=$text N=${#text} I=0
  scan ""
  S=$saved_s N=$saved_n I=$saved_i
  return 0
}

# Body of a heredoc whose delimiter is unquoted: only $(...) and `...` run.
# dquote appends the body to the caller's word, so w is put back too.
scan_expansions() {
  local text=$1 saved_s=$S saved_n=$N saved_i=$I saved_w=$w
  S=$text N=${#text} I=0
  dquote 0
  S=$saved_s N=$saved_n I=$saved_i w=$saved_w
  return 0
}

# Inside double quotes (or an unquoted heredoc body). With 1, stops at the
# closing double quote.
dquote() {
  local stop_at_quote=$1 c part
  while ((I < N)); do
    c=${S:I:1}
    case $c in
      \\)
        c=${S:I+1:1}
        case $c in
          '$'|'`'|'"'|\\) w+=$c; I=$((I + 2)) ;;
          $'\n') I=$((I + 2)) ;;
          *) w+='\'; I=$((I + 1)) ;;
        esac ;;
      '"')
        I=$((I + 1))
        [[ $stop_at_quote == 1 ]] && return 0
        w+=$c ;;
      '$')
        if [[ ${S:I+1:1} == "(" ]]; then I=$((I + 2)); scan ")"; w+='$()'
        else w+=$c; I=$((I + 1)); fi ;;
      '`') I=$((I + 1)); scan '`'; w+='``' ;;
      *)
        run_until "$DQ_SPECIAL"; [[ -n $part ]] || part=$c
        w+=$part; I=$((I + ${#part})) ;;
    esac
  done
  return 0
}

# True when the words run a shell that reads its script from stdin (no -c and
# no script file), so a heredoc fed to it is code, not data.
shell_reads_stdin() {
  local -a a=("$@")
  local i j n=${#a[@]}
  for ((i = 0; i < n; i++)); do
    case ${a[i]} in
      bash|sh|zsh|dash|ksh|*/bash|*/sh|*/zsh|*/dash|*/ksh)
        for ((j = i + 1; j < n; j++)); do
          case ${a[j]} in
            -o|+o) j=$((j + 1)) ;;
            --*) ;;
            -*c*) return 1 ;;
            -*s*) return 0 ;;
            -*) ;;
            *) return 1 ;;
          esac
        done
        return 0 ;;
      *) ;;
    esac
  done
  return 1
}

# Reads a heredoc delimiter at I and queues it. strip_tabs is 1 for <<-;
# script is 1 when the body feeds a shell that runs it.
heredoc_start() {
  local strip_tabs=$1 script=$2 d="" q=0 c part
  while ((I < N)); do
    c=${S:I:1}
    case $c in
      ' '|$'\t') [[ -n $d || $q == 1 ]] && break; I=$((I + 1)) ;;
      $'\n'|';'|'&'|'|'|'<'|'>'|'('|')') break ;;
      "'"|'"')
        q=1; I=$((I + 1)); run_until "[$c]"
        d+=$part; I=$((I + ${#part} + 1)) ;;
      \\) q=1; d+=${S:I+1:1}; I=$((I + 2)) ;;
      *) d+=$c; I=$((I + 1)) ;;
    esac
  done
  # Three flags (tabs, quoted, script), then the delimiter.
  heredocs+=("$strip_tabs$q$script$d")
  return 0
}

# Skips the queued heredoc bodies, which start at I (just after a newline). A
# body with no closing delimiter is left to be scanned as commands: the << was
# probably a shift, as in (( x = 1 << 2 )).
heredoc_bodies() {
  local k nl=$'\n' flags delim tabs rest pre body
  for ((k = 0; k < ${#heredocs[@]}; k++)); do
    flags=${heredocs[k]:0:3}
    delim=${heredocs[k]:3}
    tabs=""
    [[ ${flags:0:1} == 1 ]] && tabs='*([	])'
    rest=$nl${S:I}$nl
    pre=${rest%%$nl$tabs"$delim"$nl*}
    ((${#pre} == ${#rest})) && continue
    body=${pre#$nl}; I=$((I + ${#pre} + ${#delim} + 1))
    # <<- also strips the tabs before the closing delimiter.
    while [[ ${S:I-1:1} != "$nl" ]] && ((I < N)); do I=$((I + 1)); done
    if [[ ${flags:2:1} == 1 ]]; then scan_string "$body"
    elif [[ ${flags:1:1} == 0 ]]; then
      case $body in
        *'$('*|*'`'*) scan_expansions "$body" ;;
        *) ;;
      esac
    fi
  done
  heredocs=()
  return 0
}

# Scans commands from I. The argument says what ends this level: "" (end of
# string), ")" for $( ), "`" for backticks.
scan() {
  local stop=$1 c next w="" inword=0 depth=0 part script
  local -a words=() heredocs=()
  while ((I < N)); do
    c=${S:I:1}
    case $c in
      \\)
        next=${S:I+1:1}
        [[ $next == $'\n' ]] || { w+=$next; inword=1; }
        I=$((I + 2)) ;;
      "'")
        I=$((I + 1)); run_until "[']"
        w+=$part; inword=1; I=$((I + ${#part} + 1)) ;;
      '"') I=$((I + 1)); inword=1; dquote 1 ;;
      '`')
        I=$((I + 1))
        if [[ $stop == '`' ]]; then end_command; return 0; fi
        scan '`'; w+='``'; inword=1 ;;
      '$'|'<'|'>')
        next=${S:I+1:1}
        if [[ $next == "(" ]]; then
          I=$((I + 2)); scan ")"; w+='$()'; inword=1
        elif [[ $c$next${S:I+2:1} == "<<<" ]]; then
          w+="<<<"; inword=1; I=$((I + 3))
        elif [[ $c$next == "<<" ]]; then
          end_word; I=$((I + 2))
          script=0
          shell_reads_stdin "${words[@]}" && script=1
          if [[ ${S:I:1} == "-" ]]; then I=$((I + 1)); heredoc_start 1 "$script"; else heredoc_start 0 "$script"; fi
        else
          w+=$c; inword=1; I=$((I + 1))
        fi ;;
      ' '|$'\t') end_word; I=$((I + 1)) ;;
      $'\n')
        end_command; I=$((I + 1))
        ((${#heredocs[@]} > 0)) && heredoc_bodies ;;
      '&')
        # 2>&1 and >&2 are redirections, not a background operator.
        if [[ $w == *'>' || $w == *'<' ]]; then w+=$c; I=$((I + 1)); continue; fi
        end_command; I=$((I + 1)) ;;
      ';'|'|') end_command; I=$((I + 1)) ;;
      '(') end_command; depth=$((depth + 1)); I=$((I + 1)) ;;
      ')')
        end_command; I=$((I + 1))
        if ((depth > 0)); then depth=$((depth - 1))
        elif [[ $stop == ")" ]]; then return 0; fi ;;
      '#')
        if [[ $inword == 1 ]]; then w+=$c; I=$((I + 1))
        else run_until $'[\n]'; I=$((I + ${#part})); fi ;;
      *)
        run_until "$SPECIAL"; [[ -n $part ]] || part=$c
        w+=$part; inword=1; I=$((I + ${#part})) ;;
    esac
  done
  end_command
  return 0
}

# Characters that end a run of plain text, unquoted and inside double quotes.
# Taking a run at a time instead of a character keeps long commands fast.
SPECIAL=$'[ \t\n\\\\\'"`$<>&;|()#]'
DQ_SPECIAL=$'[\\\\"`$]'
N=${#S} I=0
scan ""
exit 0
