#!/usr/bin/env node
// The claims bot: keeps one live claim per open task (TASKS.md, "Claims").
//
//   node scripts/claims.mjs [--issue <n>] [--dry-run]
//
// Run by .github/workflows/claims.yml with the workflow's own token: every 15 minutes over every open task, and on a
// new comment over that comment's issue (ISSUE_NUMBER). `--dry-run` prints what it would do and changes nothing.
//
// Issue, comment and pull request text is untrusted. It is read as data with anchored patterns, never reaches a
// shell, and is never copied into the bot's own comments: those name logins, comment ids, times and a model's
// normalised name only. `decide` holds every rule as a pure function of the issue, its comments, the pushes to its
// linked pull requests and the time; `main` only reads and writes through the REST API.
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const AGENT_WINDOW_MS = 60 * 60 * 1000; // an agent's claim: a check-in at least every 60 minutes
export const HUMAN_WINDOW_MS = 48 * 60 * 60 * 1000; // `model: human`: a person does not check in hourly
export const EDIT_GRACE_MS = 60 * 1000; // GitHub's own timestamps can differ by a second on an unedited comment
export const TASK_LABELS = ['task', 'job']; // the labels the two task templates set
export const LABELS = {
  claimed: { name: 'claimed', color: 'd93f0b', description: 'Held by a live claim (TASKS.md)' },
  open: { name: 'open', color: '0e8a16', description: 'Open to claim (TASKS.md)' },
};
export const BOT_LOGIN = 'github-actions[bot]';
const MAINTAINER = new Set(['OWNER', 'MEMBER', 'COLLABORATOR']);

// The model requirement (AGENTS.md, "Which model"), in this one table: each named model with the least effort it may
// run at. A name is compared after lower-casing and turning every run of other characters into one hyphen, so
// "Claude Opus 5.5" and "claude-opus-5-5" are one entry. A model at that level on the Artificial Analysis
// intelligence index is not in the table: the bot cannot know it, so it says "unknown" and a maintainer decides.
export const MODEL_REQUIREMENT = [
  { name: 'GPT-6.2 Sol', pattern: /(?:^|-)gpt-6-2-sol(?:-|$)/, minEffort: 'high' },
  { name: 'Claude Opus 5.5', pattern: /(?:^|-)opus-5-5(?:-|$)/, minEffort: 'high' },
  { name: 'Claude Fable 5.2', pattern: /(?:^|-)fable-5-2(?:-|$)/, minEffort: 'medium' },
  { name: 'Gemini 4 Neon', pattern: /(?:^|-)gemini-4-neon(?:-|$)/, minEffort: 'medium' },
];
// Effort levels, lowest first. The vendors' own spellings are read case-insensitively and mapped onto these; a level
// below `low` (`minimal`, `none`) ranks below every entry's minimum.
export const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];
const EFFORT_SPELLINGS = new Map([
  ['none', -1],
  ['minimal', -1],
  ['low', 0],
  ['medium', 1],
  ['high', 2],
  ['xhigh', 3],
  ['extrahigh', 3],
  ['max', 4],
  ['maximum', 4],
]);

const KEYWORD = /^(CHECK-IN|PROGRESS|RELEASE|CHECK-OUT):(?=[ \t]|\r?\n|$)/;
const LOGIN = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;
const BRANCH = /^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$/;
const FIELD = /^([A-Za-z][A-Za-z -]{0,30}?)[ \t]*:[ \t]*(.*)$/;
const LINKS = /^[ \t]*(?:[-*][ \t]+)?(?:Task:[ \t]*#(\d{1,9})|(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)[ \t]+#(\d{1,9}))\b/gim;

/// Lower case, every run of other characters one hyphen, at most 40 characters: safe to print in a comment.
export function slug(text) {
  return String(text ?? '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40);
}

const time = (value) => (typeof value === 'number' ? value : Date.parse(value));
const sameLogin = (a, b) => String(a).toLowerCase() === String(b).toLowerCase();
const firstLine = (body) => String(body ?? '').split(/\r?\n/, 1)[0];

/// `YYYY-MM-DD HH:MM UTC`.
export function stamp(ms) {
  return `${new Date(ms).toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

/// The keyword a comment begins with (leading blank lines and spaces allowed) and its first line, or null. A keyword
/// later in the text, quoted or in a code block is not one.
export function keywordOf(body) {
  const text = String(body ?? '').replace(/^[ \t\r\n]+/, '');
  const match = KEYWORD.exec(text);
  return match ? { keyword: match[1], line: firstLine(text) } : null;
}

/// A CHECK-IN line's fields, or { error } with a fixed reason (never text from the line).
export function parseClaim(line) {
  if (line.length > 500) return { error: 'the line is longer than 500 characters' };
  const parts = line.slice('CHECK-IN:'.length).split('|').map((part) => part.trim());
  const handle = parts.shift();
  if (!LOGIN.test(handle)) return { error: 'the handle is not a GitHub login' };
  const fields = {};
  for (const part of parts) {
    const match = FIELD.exec(part);
    if (!match) return { error: 'a field is not in the form "name: value"' };
    const key = match[1].trim().toLowerCase();
    if (key in fields) return { error: 'a field is given twice' };
    fields[key] = match[2].trim();
  }
  for (const key of ['model', 'effort', 'plan', 'branch']) {
    if (!fields[key]) return { error: `the ${key} field is missing` };
  }
  if (!BRANCH.test(fields.branch) || fields.branch.includes('..')) {
    return { error: 'the branch is not a plain branch name' };
  }
  return { handle, model: fields.model, effort: fields.effort, plan: fields.plan, branch: fields.branch };
}

/// An effort level's rank in [`EFFORTS`] (-1 below `low`), or null for a word the bot does not know. Case, spaces,
/// hyphens and underscores are ignored, so "Extra high", "x-high" and "XHIGH" are one level.
export function effortRank(effort) {
  const word = String(effort ?? '').toLowerCase().replace(/[\s_-]+/g, '');
  return EFFORT_SPELLINGS.get(word) ?? null;
}

/// Whether a claim's model and effort meet the model requirement. `verdict` is `meets`, `below` (a named model under
/// its least effort, which `needs` names), `effort-unknown` (a named model with an effort word the bot cannot rank),
/// `unknown` (a model not in the table: a maintainer decides) or `human`. `model` and `effort` are normalised and safe
/// to print; `name` and `needs` come from the table.
export function requirementVerdict(model, effort) {
  const name = slug(model);
  const level = slug(effort).replace(/-/g, '');
  if (name === 'human') return { verdict: 'human', model: name, effort: level };
  const entry = MODEL_REQUIREMENT.find((row) => row.pattern.test(name));
  if (!entry) return { verdict: 'unknown', model: name, effort: level };
  const found = { model: name, effort: level, name: entry.name, needs: entry.minEffort };
  const rank = effortRank(effort);
  if (rank === null) return { verdict: 'effort-unknown', ...found };
  return { verdict: rank >= EFFORTS.indexOf(entry.minEffort) ? 'meets' : 'below', ...found };
}

/// The issue numbers a pull request's description links, by `Task: #n` or `Closes #n` at the start of a line.
export function linkedIssues(body) {
  const out = new Set();
  for (const match of String(body ?? '').matchAll(LINKS)) out.add(Number(match[1] ?? match[2]));
  return [...out];
}

/// The newest time a pull request's head could have been pushed: its head commit's committer date, but never later
/// than the fork's own last push (a date written into a commit cannot move a check-in past the real push).
export function pushTime(commitDate, forkPushedAt) {
  const commit = time(commitDate);
  const fork = time(forkPushedAt);
  if (!Number.isFinite(commit) || !Number.isFinite(fork)) return null;
  return Math.min(commit, fork);
}

/// The actions for one task. `issue`: { number, body, labels: [name] }. `comments`: [{ id, login, type,
/// association, body, created, updated }] (times as ISO strings or ms). `pushes`: [{ login, branch, at }].
/// Returns { status, holder, actions }: status is `held`, `review` (after CHECK-OUT: no clock), `lapsed` or `free`;
/// actions are { kind: 'comment', body } and { kind: 'labels', add, remove }.
export function decide({ issue, comments = [], pushes = [], now, bot = BOT_LOGIN }) {
  const nowMs = time(now);
  const botComments = [];
  const events = [];
  for (const comment of comments) {
    if (sameLogin(comment.login, bot) && comment.type === 'Bot') {
      botComments.push({ line: firstLine(comment.body), at: time(comment.created) });
      continue;
    }
    const found = keywordOf(comment.body);
    if (!found) continue;
    const at = time(comment.created);
    if (!Number.isFinite(at) || at > nowMs) continue;
    const edited = time(comment.updated) - at > EDIT_GRACE_MS;
    events.push({ at, id: Number(comment.id), comment, edited, ...found });
  }
  for (const push of pushes) {
    const at = time(push.at);
    if (Number.isFinite(at) && at <= nowMs) events.push({ at, id: 0, push });
  }
  events.sort((a, b) => a.at - b.at || a.id - b.id);

  const replies = [];
  const answered = (keyword, id) =>
    botComments.some((c) => c.line.startsWith(`${keyword}:`) && new RegExp(`\\bcomment ${id}\\b`).test(c.line));
  const reply = (keyword, id, text) => {
    if (!answered(keyword, id)) replies.push(`${keyword}: comment ${id} ${text}`);
  };
  // Only normalised names, the table's own names and fixed text reach the comment.
  const flagRequirement = (comment, claim) => {
    const result = requirementVerdict(claim.model, claim.effort);
    const names = `by ${comment.login} names ${result.model || 'no model'} at effort ${result.effort || 'none'}`;
    const where = '(AGENTS.md, "Which model")';
    if (result.verdict === 'below') {
      reply(
        'MODEL REQUIREMENT',
        comment.id,
        `${names}, below the requirement: ${result.name} needs ${result.needs} effort or above; a maintainer may ask for a rebuild ${where}`,
      );
    } else if (result.verdict === 'effort-unknown') {
      reply(
        'MODEL REQUIREMENT',
        comment.id,
        `${names}, an effort level the bot cannot rank: ${result.name} needs ${result.needs} effort or above; a maintainer decides ${where}`,
      );
    } else if (result.verdict === 'unknown') {
      reply('MODEL REQUIREMENT', comment.id, `${names}: unknown: a maintainer decides ${where}`);
    }
  };

  let holder = null;
  let lapse = null;
  const windowOf = (h) => (h.human ? HUMAN_WINDOW_MS : AGENT_WINDOW_MS);
  const checkLapse = (at) => {
    if (holder && !holder.review && at - holder.last > windowOf(holder)) {
      lapse = { login: holder.login, since: holder.last, claimedAt: holder.claimedAt };
      holder = null;
    }
  };

  for (const event of events) {
    checkLapse(event.at);
    if (event.push) {
      if (holder && sameLogin(event.push.login, holder.login) && event.push.branch === holder.branch) {
        holder.last = event.at;
      }
      continue;
    }
    const { comment, keyword, line, at, edited } = event;
    const mine = holder && sameLogin(comment.login, holder.login);
    if (keyword === 'CHECK-IN') {
      if (holder && !mine) {
        reply('CLAIM HELD', comment.id, `by ${comment.login} is not a claim: ${holder.login} holds this issue since ${stamp(holder.claimedAt)}; wait for a RELEASE or a CLAIM LAPSED notice, or take another issue`);
        continue;
      }
      if (edited) {
        reply('CLAIM NOT READ', comment.id, `by ${comment.login}: it was edited after it was posted, and an edited comment is not read; post a new one`);
        continue;
      }
      const claim = parseClaim(line);
      const reason = claim.error ?? (sameLogin(claim.handle, comment.login) ? null : 'the handle is not the login that posted it');
      if (reason) {
        if (mine) holder.last = at; // still a check-in by the holder
        else reply('CLAIM NOT READ', comment.id, `by ${comment.login}: ${reason}; post a new comment in the form in TASKS.md`);
        continue;
      }
      if (mine) {
        Object.assign(holder, { last: at, review: false, model: claim.model, effort: claim.effort, branch: claim.branch });
        holder.human = slug(claim.model) === 'human';
      } else {
        holder = {
          login: comment.login,
          claimedAt: at,
          last: at,
          review: false,
          human: slug(claim.model) === 'human',
          model: claim.model,
          effort: claim.effort,
          branch: claim.branch,
        };
        lapse = null;
      }
      flagRequirement(comment, claim);
    } else if (edited) {
      continue; // an edited PROGRESS, RELEASE or CHECK-OUT is not read either
    } else if (keyword === 'PROGRESS') {
      if (mine) holder.last = at;
    } else if (keyword === 'CHECK-OUT') {
      if (mine) Object.assign(holder, { last: at, review: true });
    } else if (keyword === 'RELEASE') {
      if (holder && (mine || MAINTAINER.has(comment.association))) {
        holder = null;
        lapse = null;
      }
    }
  }
  checkLapse(nowMs);

  const status = holder ? (holder.review ? 'review' : 'held') : lapse ? 'lapsed' : 'free';
  const actions = replies.map((body) => ({ kind: 'comment', body }));
  if (lapse) {
    const already = botComments.some((c) => c.line.startsWith(`CLAIM LAPSED: ${lapse.login},`) && c.at >= lapse.claimedAt);
    if (!already) {
      actions.push({ kind: 'comment', body: `CLAIM LAPSED: ${lapse.login}, no check-in since ${stamp(lapse.since)}; open to claim` });
    }
  }
  const have = new Set((issue.labels ?? []).map((label) => (typeof label === 'string' ? label : label.name)));
  const want = holder ? LABELS.claimed.name : LABELS.open.name;
  const drop = holder ? LABELS.open.name : LABELS.claimed.name;
  const add = have.has(want) ? [] : [want];
  const remove = have.has(drop) ? [drop] : [];
  if (add.length || remove.length) actions.push({ kind: 'labels', add, remove });
  return {
    status,
    holder: holder && { login: holder.login, since: holder.claimedAt, last: holder.last, human: holder.human },
    actions,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// The REST side. Nothing below decides anything.

function client({ token, api = 'https://api.github.com', repo, fetchImpl }) {
  const call = async (method, route, body) => {
    const response = await fetchImpl(`${api}/repos/${repo}${route}`, {
      method,
      headers: {
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'User-Agent': 'claims-bot',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (response.status === 404 && method === 'GET') return null;
    if (!response.ok) throw new Error(`${method} ${route.split('?')[0]}: HTTP ${response.status}`);
    return response.status === 204 ? null : response.json();
  };
  const paged = async (route) => {
    const out = [];
    for (let page = 1; page <= 50; page += 1) {
      const sep = route.includes('?') ? '&' : '?';
      const batch = (await call('GET', `${route}${sep}per_page=100&page=${page}`)) ?? [];
      out.push(...batch);
      if (batch.length < 100) break;
    }
    return out;
  };
  return { call, paged };
}

const isTask = (issue) =>
  issue && !issue.pull_request && issue.state === 'open' && issue.labels.some((l) => TASK_LABELS.includes(l.name));

async function tasks(gh, number) {
  if (number) {
    const issue = await gh.call('GET', `/issues/${number}`);
    return isTask(issue) ? [issue] : [];
  }
  const byNumber = new Map();
  for (const label of TASK_LABELS) {
    for (const issue of await gh.paged(`/issues?state=open&labels=${encodeURIComponent(label)}`)) {
      if (isTask(issue)) byNumber.set(issue.number, issue);
    }
  }
  return [...byNumber.values()].sort((a, b) => a.number - b.number);
}

async function pushesFor(gh, pulls, number) {
  const out = [];
  for (const pull of pulls.filter((p) => linkedIssues(p.body).includes(number))) {
    if (!pull.head?.repo || !/^[0-9a-f]{40}$/.test(pull.head.sha)) continue;
    const commit = await gh.call('GET', `/commits/${pull.head.sha}`);
    const at = pushTime(commit?.commit?.committer?.date, pull.head.repo.pushed_at);
    if (at !== null) out.push({ login: pull.user.login, branch: pull.head.ref, at });
  }
  return out;
}

async function ensureLabel(gh, name, made) {
  if (made.has(name)) return;
  const spec = Object.values(LABELS).find((label) => label.name === name);
  if (!(await gh.call('GET', `/labels/${encodeURIComponent(name)}`))) await gh.call('POST', '/labels', spec);
  made.add(name);
}

/// One pass: reads the tasks and applies `decide`'s actions. Returns the number of issues that failed.
export async function runClaims({ args = [], env = process.env, now = Date.now(), fetchImpl = fetch, log = console.log, warn = console.error } = {}) {
  const dryRun = args.includes('--dry-run');
  const at = args.indexOf('--issue');
  const issueArg = at >= 0 ? (args[at + 1] ?? '') : env.ISSUE_NUMBER || null;
  if (issueArg !== null && !/^\d{1,9}$/.test(String(issueArg))) throw new Error('--issue needs an issue number');
  const repo = env.GITHUB_REPOSITORY ?? '';
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo)) throw new Error('GITHUB_REPOSITORY must be owner/name');
  const token = env.GITHUB_TOKEN;
  if (!token && !dryRun) throw new Error('GITHUB_TOKEN is not set (a dry run may go without it)');
  const gh = client({ token, api: env.GITHUB_API_URL || undefined, repo, fetchImpl });
  const bot = env.CLAIMS_BOT_LOGIN || BOT_LOGIN;

  const open = await tasks(gh, issueArg ? Number(issueArg) : null);
  const pulls = open.some((issue) => issue.comments > 0) ? await gh.paged('/pulls?state=open') : [];
  const made = new Set();
  let failed = 0;
  for (const issue of open) {
    try {
      const raw = issue.comments > 0 ? await gh.paged(`/issues/${issue.number}/comments`) : [];
      const comments = raw.map((c) => ({
        id: c.id,
        login: c.user?.login ?? '',
        type: c.user?.type ?? '',
        association: c.author_association,
        body: c.body,
        created: c.created_at,
        updated: c.updated_at,
      }));
      const pushes = comments.length ? await pushesFor(gh, pulls, issue.number) : [];
      const result = decide({ issue: { number: issue.number, body: issue.body, labels: issue.labels }, comments, pushes, now, bot });
      const done = [];
      for (const action of result.actions) {
        if (action.kind === 'comment') {
          done.push(action.body.split(':', 1)[0]);
          if (!dryRun) await gh.call('POST', `/issues/${issue.number}/comments`, { body: action.body });
        } else {
          done.push([...action.add.map((n) => `+${n}`), ...action.remove.map((n) => `-${n}`)].join(' '));
          if (dryRun) continue;
          for (const name of action.add) await ensureLabel(gh, name, made);
          if (action.add.length) await gh.call('POST', `/issues/${issue.number}/labels`, { labels: action.add });
          for (const name of action.remove) await gh.call('DELETE', `/issues/${issue.number}/labels/${encodeURIComponent(name)}`);
        }
      }
      const who = result.holder ? ` by ${result.holder.login}` : '';
      log(`#${issue.number}: ${result.status}${who}${done.length ? `; ${done.join('; ')}` : ''}${dryRun ? ' (dry run)' : ''}`);
    } catch (err) {
      failed += 1;
      warn(`#${issue.number}: ${err.message}`);
    }
  }
  log(`claims: ${open.length} task(s) read, ${failed} failed`);
  return failed;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runClaims({ args: process.argv.slice(2) }).then(
    (failed) => process.exit(failed ? 1 : 0),
    (err) => {
      console.error(`claims: ${err.message}`);
      process.exit(1);
    },
  );
}
