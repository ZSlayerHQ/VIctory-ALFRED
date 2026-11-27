// The runtime's log lines a harness reader parses, as `world` writes them: the message after the timestamp, level
// and module `tracing` puts before it (a test adds that prefix itself where it matters).
//
// Read at world 2f6b742fb, unchanged through 7cab1c669 by the drift test (VIctory, `crates/victory-gpu/src/live.rs` `segment_line`, `marker_line`, `done_line`;
// `host/stats.rs` `frame_line`, `window_text`, `Budget::line`; `host/queue_stamps.rs` `text`). Where a runtime test
// pins a whole line it is copied here character for character; where it pins only a part (a segment line up to its
// skipped calls, a budget line's start and end), the rest is what the line's format string writes for the same
// test's inputs, the switches in force being a run's with no environment (IN_FORCE_DEFAULTS). Each constant names
// its test. fixtures/runtime-shapes.test.mjs re-reads the format strings and the tests and fails on drift.
import { IN_FORCE_DEFAULTS } from './runtime-stats.mjs';

const TAIL = `, skipped 0 draws and 0 dispatches, in force: ${IN_FORCE_DEFAULTS}`;

/// live.rs `the_marker_and_segment_lines_have_their_fixed_shape`: the open line (pinned up to `, skipped`).
export const SEGMENT_OPEN_LINE =
  'live segment 2 open: frames 10..19, 4 counted, 3 excluded as transition (the first 3 after a change; their mean ' +
  '310.0), 1 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their ' +
  'mean 90.0), 2 excluded as long (over 1000 ms; their mean 1500.0), mean 25.0, p50 20.0, p95 40.0 of the counted, ' +
  'begun at a change, not ended (the segment under way, as of this line)' +
  TAIL;

/// The same test: the line the same segment logs when a change closes it (the test pins it as the open line with
/// ` open:` and `not ended (...)` replaced).
export const SEGMENT_CLOSED_LINE =
  'live segment 2: frames 10..19, 4 counted, 3 excluded as transition (the first 3 after a change; their mean ' +
  '310.0), 1 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their ' +
  'mean 90.0), 2 excluded as long (over 1000 ms; their mean 1500.0), mean 25.0, p50 20.0, p95 40.0 of the counted, ' +
  'begun at a change, ended by a change' +
  TAIL;

/// live.rs `a_segment_leaves_its_transition_light_and_long_frames_out_and_says_so`: a run's first segment, a load
/// frame left out as light and a long one as long.
export const SEGMENT_FIRST_LINE =
  'live segment 1: frames 1..4, 2 counted, 0 excluded as transition (the first 3 after a change; their mean 0.0), ' +
  '1 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their mean ' +
  '1500.0), 1 excluded as long (over 1000 ms; their mean 1200.0), mean 200.0, p50 200.0, p95 200.0 of the counted, ' +
  'begun at the start of the run, ended by a change' +
  TAIL;

/// live.rs `a_mark_cuts_a_segment_and_leaves_no_frame_out`: the stretch before a plan, ended by its mark.
export const SEGMENT_ENDED_BY_MARK_LINE =
  'live segment 1: frames 1..2, 2 counted, 0 excluded as transition (the first 3 after a change; their mean 0.0), ' +
  '0 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their mean ' +
  '0.0), 0 excluded as long (over 1000 ms; their mean 0.0), mean 550.0, p50 200.0, p95 900.0 of the counted, ' +
  'begun at the start of the run, ended by a mark' +
  TAIL;

/// The same test: the plan's first arm, begun by the mark (no frame left out as transition), ended by a change.
export const SEGMENT_BEGUN_AT_MARK_LINE =
  'live segment 2: frames 3..5, 3 counted, 0 excluded as transition (the first 3 after a change; their mean 0.0), ' +
  '0 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their mean ' +
  '0.0), 0 excluded as long (over 1000 ms; their mean 0.0), mean 220.0, p50 220.0, p95 230.0 of the counted, ' +
  'begun at a mark, ended by a change' +
  TAIL;

/// live.rs `a_run_ended_by_the_timer_has_its_last_segment_written`: the line of the segment under way (a run killed
/// from outside has only this), then the line the same frames give at a run end the runtime sees.
export const SEGMENT_TIMER_OPEN_LINE =
  'live segment 2 open: frames 2..7, 3 counted, 3 excluded as transition (the first 3 after a change; their mean ' +
  '433.3), 0 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their ' +
  'mean 0.0), 0 excluded as long (over 1000 ms; their mean 0.0), mean 240.0, p50 240.0, p95 244.0 of the counted, ' +
  'begun at a change, not ended (the segment under way, as of this line)' +
  TAIL;
export const SEGMENT_RUN_END_LINE =
  'live segment 2: frames 2..7, 3 counted, 3 excluded as transition (the first 3 after a change; their mean ' +
  '433.3), 0 excluded as light (under 1000 draws: a load, a menu or a pause, the runtime cannot tell which; their ' +
  'mean 0.0), 0 excluded as long (over 1000 ms; their mean 0.0), mean 240.0, p50 240.0, p95 244.0 of the counted, ' +
  'begun at a change, ended by the end of the run' +
  TAIL;

/// live.rs `the_marker_and_segment_lines_have_their_fixed_shape`: the marker a change logs (pinned whole).
export const CHANGE_LINE = 'live: VICTORY_TEXTURE_LOW 70 -> 60 at frame 13';

/// live.rs `marker_line` for the mark of `a_mark_cuts_a_segment_and_leaves_no_frame_out` (`LIVE_MARK=7` read at the
/// boundary of frame 2): a mark is logged as a change of the switch `LIVE_MARK`.
export const MARK_LINE = 'live: LIVE_MARK 0 -> 7 at frame 2';

/// live.rs `the_done_line_has_its_fixed_shape_and_a_run_stops_once` (pinned whole).
export const KEYS_DONE_LINE = 'live keys: sequence done at frame 1234, 2 presses';

/// stats.rs `frame_line` over the interval of `a_frame_line_covers_every_second_since_the_last` (frames 1..6 in
/// 2000 ms, 40 ms of copies a frame; its window clause pinned), every counter of the line at its start, no usage
/// sample yet (`Usage::text(None)`).
export const FRAME_LINE =
  "frame 6, 3.0 fps, 333.3 ms/frame (0.0 preparing: copies 40.0 ms 0.0 MB, textures hashed 0.0 ms 0.0 MB; 0.0 host " +
  "waits (for the relay 0.0, for flights' fence 0.0, synchronous submissions sent to retired 0.0); 0.0 copying back " +
  "0.0 MB merged into title memory (host time, fetches included), 0.0 MB fetched from copies in the card's memory), " +
  '0 draws/frame, dropped 0 without a host pipeline (0 being made) and 0 indirect; heap buffers 0.0 passes 0.0 ms; ' +
  'texture slots written so far: 0 placed, 0 of objects; textures picked by index: 0 uploaded so far (0 MB), 0 ' +
  'waiting, 0 passed over unmapped; depth textures with unordered access: 0 made, bound as depth 0x, cleared as ' +
  'depth 0x, unordered views refused 0x; buffer views left null as the host would refuse them: 0; depth twins: 0 ' +
  'alive, 0 depth textures left without one; twin copies per frame: 0.0 into depth textures, 0.0 into twins; depth ' +
  'planes: 0.0 drawn from their twins per frame, 0 times not given back to a twin, 0 times used before the pass; ' +
  'host waits summed across threads 0.0 ms a frame, cpu - ram -; means over frames 1..6 (6 frames in 2000 ms, p50 ' +
  '250.0, p95 500.0 ms/frame): every frame since the line before; the figures in the parentheses are summed across ' +
  "threads and overlap, they are not parts of the frame (the frame budget line's are)";

/// stats.rs `title_is_only_the_title_and_the_calls_are_counted`: `Budget::line(1230, 1234, 5)` (its processor and
/// calls clauses pinned): the title's calls as phases of their own, no other thread measured.
export const BUDGET_LINE =
  'frame budget, the presenting thread, frames 1230..1234 (2 frames): 200.00 ms/frame = title 60.00 + list_calls ' +
  '30.00 + unaccounted 100.00 + object_calls 6.00 + hle_calls 4.00 (sum 200.00); of it on the processor: the title ' +
  'and its calls together 90.00, unaccounted 50.00; 30 scopes a frame, 2000 calls from the title a frame, the ' +
  'largest stretch of title between two 3.50 ms; other threads per frame, overlapping, not parts of the frame: ' +
  'nothing measured';

/// stats.rs `the_budget_is_in_the_object_and_other_threads_are_apart`: `Budget::line(1230, 1234, 5)` (its start and
/// end pinned): a wait among the phases and another thread's work apart.
export const BUDGET_LINE_OTHERS =
  'frame budget, the presenting thread, frames 1230..1234 (2 frames): 200.00 ms/frame = title 100.00 + unaccounted ' +
  '20.00 + relay_wait 80.00 (sum 200.00); of it on the processor: the title and its calls together 0.00, ' +
  'unaccounted 10.00, relay_wait 40.00; 30 scopes a frame, 0 calls from the title a frame, the largest stretch of ' +
  'title between two 0.00 ms; other threads per frame, overlapping, not parts of the frame: relay_land 60.00 (cpu ' +
  '15.00)';

/// host/queue_stamps.rs `a_copy_queue_stamp_appears_in_the_report`: the per-queue line (`text`) for its copy queue
/// figures, every drawing-queue kind not stamped as `not_stamped_with(true, Some(true))` lists them.
export const GPU_LINE =
  "card queues by GPU timestamps, each queue by itself (VICTORY_GPU_STAMPS=1): copy queue per frame: 2.00 fetch " +
  "lists, 1.50 of them stamped, 3.000 ms between their first and last copy on the copy queue's clock (0.50 pairs " +
  "unread); these spans overlap the drawing queue's and are not added to it or taken from its uncovered time; not " +
  'stamped: drawing queue: the resolves of the stamps themselves; drawing queue: lists of ours run under the ' +
  'submission lock; drawing queue: our indirect, in-place and tile work lists; drawing queue: the hash list a ' +
  'synchronous submission without a title list sends under VICTORY_GPU_TIMING; drawing queue: the hash check\'s own ' +
  'fetch lists (VICTORY_HASH_CHECK=1)';
