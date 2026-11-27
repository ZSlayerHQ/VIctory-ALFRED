// The runtime's live stats object, as `world` writes it: one set of fixtures every reader's tests use.
//
// Read at world 2f6b742fb, unchanged through 7cab1c669 by the drift test (VIctory, `crates/victory-gpu/src/host/stats.rs` `json`, `optional`, `Budget::json`;
// `live.rs` `segment_json`; `host/queue_stamps.rs` `json`; `kernelstat.rs` `text`). No test of the runtime pins a
// whole object, so each object below is what `json` writes for the inputs of the tests named beside it, every
// string those tests pin appearing in it character for character; a value no test pins (a counter the tests leave at
// its start) is that start, zero. fixtures/runtime-shapes.test.mjs re-reads the runtime's format strings and those
// tests and fails when a fixture here has drifted from them.

/// Every switch off (no optional object): `json(&interval(), &now(7))` of stats.rs
/// `the_object_has_every_field` (and `totals_say_total_and_the_worst_frame_is_called_that`,
/// `dispatches_are_per_frame`), with the budget of `the_budget_is_in_the_object_and_other_threads_are_apart` (a
/// diagnostics build) and the copy-back parts and copy-queue figures the first test remembers.
export const STATS_ALL_OFF = `{"seq":7,"t_ms":7000.0,"unix_ms":1800000007000,"interval":{"first_frame":1230,"last_frame":1234,"frames":5,"wall_ms":1300.00},"frame":1234,"fps":3.85,"frame_ms":260.00,"frame_ms_p50":220.00,"frame_ms_worst":420.00,"frame_ms_list":[220.00,220.00,420.00,220.00,220.00],"frame_ms_list_missing":0,"budget":{"thread":"presenting","frames":2,"frame_ms":200.000,"sum_ms":200.000,"scopes":30.0,"calls":0.0,"largest_title_gap_ms":0.000,"title_and_calls_cpu_ms":0.000,"phases":{"title":{"ms":100.000,"cpu_ms":null},"list_calls":{"ms":0.000,"cpu_ms":null},"unaccounted":{"ms":20.000,"cpu_ms":10.000},"submission_lock_wait":{"ms":0.000,"cpu_ms":0.000},"settle":{"ms":0.000,"cpu_ms":0.000},"relay_wait":{"ms":80.000,"cpu_ms":40.000},"card_wait":{"ms":0.000,"cpu_ms":0.000},"sweep":{"ms":0.000,"cpu_ms":0.000},"slots":{"ms":0.000,"cpu_ms":0.000},"refresh":{"ms":0.000,"cpu_ms":0.000},"textures":{"ms":0.000,"cpu_ms":0.000},"pool_wait":{"ms":0.000,"cpu_ms":0.000},"snapshots":{"ms":0.000,"cpu_ms":0.000},"own_lists":{"ms":0.000,"cpu_ms":0.000},"send":{"ms":0.000,"cpu_ms":0.000},"copy_back":{"ms":0.000,"cpu_ms":0.000},"present_call":{"ms":0.000,"cpu_ms":0.000},"report":{"ms":0.000,"cpu_ms":0.000},"object_calls":{"ms":0.000,"cpu_ms":null},"hle_calls":{"ms":0.000,"cpu_ms":null},"queue_lock_wait":{"ms":0.000,"cpu_ms":0.000},"pace_wait":{"ms":0.000,"cpu_ms":0.000}}},"other_threads":{"scopes":2.0,"phases":{"list_calls":{"ms":0.000,"cpu_ms":null},"unaccounted":{"ms":0.000,"cpu_ms":0.000},"submission_lock_wait":{"ms":0.000,"cpu_ms":0.000},"settle":{"ms":0.000,"cpu_ms":0.000},"relay_wait":{"ms":0.000,"cpu_ms":0.000},"card_wait":{"ms":0.000,"cpu_ms":0.000},"sweep":{"ms":0.000,"cpu_ms":0.000},"slots":{"ms":0.000,"cpu_ms":0.000},"refresh":{"ms":0.000,"cpu_ms":0.000},"textures":{"ms":0.000,"cpu_ms":0.000},"pool_wait":{"ms":0.000,"cpu_ms":0.000},"snapshots":{"ms":0.000,"cpu_ms":0.000},"own_lists":{"ms":0.000,"cpu_ms":0.000},"send":{"ms":0.000,"cpu_ms":0.000},"copy_back":{"ms":0.000,"cpu_ms":0.000},"present_call":{"ms":0.000,"cpu_ms":0.000},"report":{"ms":0.000,"cpu_ms":0.000},"pool_work":{"ms":0.000,"cpu_ms":0.000},"relay_land":{"ms":60.000,"cpu_ms":15.000},"object_calls":{"ms":0.000,"cpu_ms":null},"hle_calls":{"ms":0.000,"cpu_ms":null},"queue_lock_wait":{"ms":0.000,"cpu_ms":0.000},"pace_wait":{"ms":0.000,"cpu_ms":0.000}}},"summed_across_threads":{"preparing_ms":96.00,"copies_ms":40.00,"copies_mb":156.00,"textures_ms":4.90,"textures_mb":16.70,"host_wait_ms":78.00,"back_ms":122.00,"back_mb":45.30,"back_fetch_ms":1.00,"back_card_ms":2.00,"back_scan_ms":3.00,"back_merge_ms":4.00,"back_done_ms":5.00,"pool_mb":0.00,"pool_join_ms":0.00,"fetch_queue_fetches":11.50,"fetch_queue_mb":74.25,"fetch_queue_ms":3.50,"fetch_queue_held":9.00,"fetch_queue_held_ms":12.75,"fetch_queue_fallback":4.00,"fetch_queue_fallback_by":{"numbered":3.00,"quarantine":0.00,"stopped":0.00,"released":1.00,"late":0.00,"shared":0.00},"fetch_queue_stops":0.00,"fetch_queue_qualified":7.50,"fetch_queue_held_by_address":2.25},"gpu_stamps":null,"draws":6500.0,"dispatched":12.0,"dropped":2.0,"stores_failed_total":0,"stores_failed_bytes_total":0,"pipelines_created_total":0,"pipelines_per_minute":0.0,"vram_mb":18600,"live":{"VICTORY_LAND_SCAN":"off","VICTORY_EXACT_BACK":"on"},"segments_closed_total":3,"segment_open":null,"kernel_calls":{"per_frame":{"reads":0.0,"writes":0.0,"protects":0.0,"queries":0.0,"write_watch":0.0},"reads_total":0,"writes_total":0,"protects_total":0,"queries_total":0,"write_watch_total":0,"classes":["1B","<=4K","<=64K",">64K"],"read_fields":["ok_calls","ok_bytes","failed_calls","failed_asked","failed_moved"],"reads_by_caller_total":{"copy_mapped":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"read_sparse":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"read_pieces":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"read_into":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"readable":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"recheck":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"io_status":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]},"fallbacks_by_caller_total":{"copy_mapped":0,"read_sparse":0,"read_pieces":0,"read_into":0,"readable":0,"recheck":0,"io_status":0},"write_nt_calls_total":0,"write_nt_bytes_asked_total":0,"write_nt_bytes_written_total":0,"write_nt_short_total":0,"write_process_calls_total":0,"write_process_bytes_asked_total":0,"write_process_bytes_written_total":0,"write_process_short_total":0,"protects_by_place_total":{"guard":0,"inplace":0,"copy":0,"texture":0,"title":0,"copystat":0},"queries_by_place_total":{"guard":0,"inplace":0,"copy":0,"texture":0,"title":0,"copystat":0},"working_set_by_place_total":{"guard":[0,0],"inplace":[0,0],"copy":[0,0],"texture":[0,0],"title":[0,0],"copystat":[0,0]},"write_watch_parts_total":{"calls":0,"failed":0,"pages_asked":0,"pages_written":0,"resets":0,"reset_pages":0}}}`;

/// A run with `VICTORY_LIVE` set and the GPU stamps on: the same interval as `now(8)`, the budget of stats.rs
/// `title_is_only_the_title_and_the_calls_are_counted`, the `live` map the registry writes with no environment
/// (live.rs `pairs`: `SPECS` in order, the panel's switch left out while it is off), `segment_open` as live.rs
/// `the_marker_and_segment_lines_have_their_fixed_shape` pins it (so one segment closed), and `gpu_stamps` as
/// host/queue_stamps.rs `a_copy_queue_stamp_appears_in_the_report` and `idle_is_said_not_to_be_derivable` make it.
export const STATS_LIVE = `{"seq":8,"t_ms":8000.0,"unix_ms":1800000008000,"interval":{"first_frame":1230,"last_frame":1234,"frames":5,"wall_ms":1300.00},"frame":1234,"fps":3.85,"frame_ms":260.00,"frame_ms_p50":220.00,"frame_ms_worst":420.00,"frame_ms_list":[220.00,220.00,420.00,220.00,220.00],"frame_ms_list_missing":0,"budget":{"thread":"presenting","frames":2,"frame_ms":200.000,"sum_ms":200.000,"scopes":30.0,"calls":2000.0,"largest_title_gap_ms":3.500,"title_and_calls_cpu_ms":90.000,"phases":{"title":{"ms":60.000,"cpu_ms":null},"list_calls":{"ms":30.000,"cpu_ms":null},"unaccounted":{"ms":100.000,"cpu_ms":50.000},"submission_lock_wait":{"ms":0.000,"cpu_ms":0.000},"settle":{"ms":0.000,"cpu_ms":0.000},"relay_wait":{"ms":0.000,"cpu_ms":0.000},"card_wait":{"ms":0.000,"cpu_ms":0.000},"sweep":{"ms":0.000,"cpu_ms":0.000},"slots":{"ms":0.000,"cpu_ms":0.000},"refresh":{"ms":0.000,"cpu_ms":0.000},"textures":{"ms":0.000,"cpu_ms":0.000},"pool_wait":{"ms":0.000,"cpu_ms":0.000},"snapshots":{"ms":0.000,"cpu_ms":0.000},"own_lists":{"ms":0.000,"cpu_ms":0.000},"send":{"ms":0.000,"cpu_ms":0.000},"copy_back":{"ms":0.000,"cpu_ms":0.000},"present_call":{"ms":0.000,"cpu_ms":0.000},"report":{"ms":0.000,"cpu_ms":0.000},"object_calls":{"ms":6.000,"cpu_ms":null},"hle_calls":{"ms":4.000,"cpu_ms":null},"queue_lock_wait":{"ms":0.000,"cpu_ms":0.000},"pace_wait":{"ms":0.000,"cpu_ms":0.000}}},"other_threads":{"scopes":0.0,"phases":{"list_calls":{"ms":0.000,"cpu_ms":null},"unaccounted":{"ms":0.000,"cpu_ms":0.000},"submission_lock_wait":{"ms":0.000,"cpu_ms":0.000},"settle":{"ms":0.000,"cpu_ms":0.000},"relay_wait":{"ms":0.000,"cpu_ms":0.000},"card_wait":{"ms":0.000,"cpu_ms":0.000},"sweep":{"ms":0.000,"cpu_ms":0.000},"slots":{"ms":0.000,"cpu_ms":0.000},"refresh":{"ms":0.000,"cpu_ms":0.000},"textures":{"ms":0.000,"cpu_ms":0.000},"pool_wait":{"ms":0.000,"cpu_ms":0.000},"snapshots":{"ms":0.000,"cpu_ms":0.000},"own_lists":{"ms":0.000,"cpu_ms":0.000},"send":{"ms":0.000,"cpu_ms":0.000},"copy_back":{"ms":0.000,"cpu_ms":0.000},"present_call":{"ms":0.000,"cpu_ms":0.000},"report":{"ms":0.000,"cpu_ms":0.000},"pool_work":{"ms":0.000,"cpu_ms":0.000},"relay_land":{"ms":0.000,"cpu_ms":0.000},"object_calls":{"ms":0.000,"cpu_ms":null},"hle_calls":{"ms":0.000,"cpu_ms":null},"queue_lock_wait":{"ms":0.000,"cpu_ms":0.000},"pace_wait":{"ms":0.000,"cpu_ms":0.000}}},"summed_across_threads":{"preparing_ms":96.00,"copies_ms":40.00,"copies_mb":156.00,"textures_ms":4.90,"textures_mb":16.70,"host_wait_ms":78.00,"back_ms":122.00,"back_mb":45.30,"back_fetch_ms":1.00,"back_card_ms":2.00,"back_scan_ms":3.00,"back_merge_ms":4.00,"back_done_ms":5.00,"pool_mb":0.00,"pool_join_ms":0.00,"fetch_queue_fetches":11.50,"fetch_queue_mb":74.25,"fetch_queue_ms":3.50,"fetch_queue_held":9.00,"fetch_queue_held_ms":12.75,"fetch_queue_fallback":4.00,"fetch_queue_fallback_by":{"numbered":3.00,"quarantine":0.00,"stopped":0.00,"released":1.00,"late":0.00,"shared":0.00},"fetch_queue_stops":0.00,"fetch_queue_qualified":7.50,"fetch_queue_held_by_address":2.25},"gpu_stamps":{"copy_queue":{"fetch_lists":2.00,"stamped":1.50,"ms":3.000,"pairs_unread":0.50},"not_stamped":["drawing queue: the resolves of the stamps themselves","drawing queue: lists of ours run under the submission lock","drawing queue: our indirect, in-place and tile work lists","drawing queue: the hash list a synchronous submission without a title list sends under VICTORY_GPU_TIMING","drawing queue: the hash check's own fetch lists (VICTORY_HASH_CHECK=1)"],"idle":"not derivable: the queues' spans are on separate clocks with no calibration to a common time base, and some lists carry no stamps"},"draws":6500.0,"dispatched":12.0,"dropped":2.0,"stores_failed_total":0,"stores_failed_bytes_total":0,"pipelines_created_total":0,"pipelines_per_minute":0.0,"vram_mb":18600,"live":{"LIVE_KEYS":"","LIVE_MARK":"0","LIVE_SKIP_DISPATCH":"","LIVE_SKIP_DRAW":"","VICTORY_EXACT_BACK":"off","VICTORY_FETCH_QUEUE":"off","VICTORY_KEEP_TEXTURES":"on","VICTORY_LAND_SCAN":"off","VICTORY_LATE_WAITS":"off","VICTORY_TEXTURE_HIGH":"85","VICTORY_TEXTURE_LOW":"70","VICTORY_TEXTURE_MADE":"off","VICTORY_TEXTURE_RECHECK":"off","VICTORY_UPLOAD_POOL":"off"},"segments_closed_total":1,"segment_open":{"segment":2,"first_frame":10,"last_frame":19,"counted":4,"transition_excluded":3,"transition_mean_ms":310.00,"light_excluded":1,"light_mean_ms":90.00,"long_excluded":2,"long_mean_ms":1500.00,"mean_ms":25.00,"p50_ms":20.00,"p95_ms":40.00,"begun":"change"},"kernel_calls":{"per_frame":{"reads":0.0,"writes":0.0,"protects":0.0,"queries":0.0,"write_watch":0.0},"reads_total":0,"writes_total":0,"protects_total":0,"queries_total":0,"write_watch_total":0,"classes":["1B","<=4K","<=64K",">64K"],"read_fields":["ok_calls","ok_bytes","failed_calls","failed_asked","failed_moved"],"reads_by_caller_total":{"copy_mapped":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"read_sparse":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"read_pieces":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"read_into":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"readable":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"recheck":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],"io_status":[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]},"fallbacks_by_caller_total":{"copy_mapped":0,"read_sparse":0,"read_pieces":0,"read_into":0,"readable":0,"recheck":0,"io_status":0},"write_nt_calls_total":0,"write_nt_bytes_asked_total":0,"write_nt_bytes_written_total":0,"write_nt_short_total":0,"write_process_calls_total":0,"write_process_bytes_asked_total":0,"write_process_bytes_written_total":0,"write_process_short_total":0,"protects_by_place_total":{"guard":0,"inplace":0,"copy":0,"texture":0,"title":0,"copystat":0},"queries_by_place_total":{"guard":0,"inplace":0,"copy":0,"texture":0,"title":0,"copystat":0},"working_set_by_place_total":{"guard":[0,0],"inplace":[0,0],"copy":[0,0],"texture":[0,0],"title":[0,0],"copystat":[0,0]},"write_watch_parts_total":{"calls":0,"failed":0,"pages_asked":0,"pages_written":0,"resets":0,"reset_pages":0}}}`;

/// The optional parts `optional` appends after `kernel_calls`, each only while its switch is on: the hash check's
/// three fields (`VICTORY_HASH_CHECK=1`, host/hash_check.rs `fields`), then the objects in `optional`'s order:
/// `texture_receipts` (`VICTORY_TEXTURE_RECEIPTS=1`, receipts.rs `stats_json`), `landed_classes`
/// (`VICTORY_LANDED_CLASSES=1`, memclass/census/landings.rs `landed_json`: `null` until its first 600-frame
/// window ends), `texture_recheck` (`VICTORY_TEXTURE_RECHECK=1`, stats.rs `recheck_text`) and `textures_made`
/// (`VICTORY_TEXTURE_MADE=1`, stats.rs `made_text`; diagnostics builds only). Every count at its start.
export const STATS_OPTIONAL_PARTS = Object.freeze({
  hash_check: `,"hash_missed_pieces_total":0,"hash_checked_pieces_total":0,"hash_check_skipped_total":0`,
  texture_receipts: `,"texture_receipts":{"installs_total":0,"lowered_total":0,"caused_total":0,"absent_total":0,"unwatched_total":0,"partial_skips_total":0,"never_total":0,"publishes_total":0,"fanned_out_total":0,"heaps":[]}`,
  landed_classes: `,"landed_classes":null`,
  texture_recheck: `,"texture_recheck":{"uploads_total":0,"too_large_total":0,"again_total":0,"stale_total":0,"stale_textures_total":0,"unused_total":0,"released_total":0,"following":0,"stale_storage_total":0,"stale_landing_total":0,"stale_no_writer_seen_total":0,"stale_writer_unknown_total":0,"not_looked_total":0,"follows_dropped_total":0,"lines_capped_total":0,"frames_skipped_total":0}`,
  textures_made: `,"textures_made":{"new_keys_total":0,"storage_requests_total":0,"last_minute":null}`,
});

/// STATS_ALL_OFF with one optional part on, as the runtime writes it: the part right after `kernel_calls`.
export function statsWith(name) {
  const part = STATS_OPTIONAL_PARTS[name];
  if (part === undefined) throw new Error(`no optional part ${name}`);
  return `${STATS_ALL_OFF.slice(0, -1)}${part}}`;
}

/// The keys only an optional part adds (STATS_OPTIONAL_PARTS).
const OPTIONAL_KEYS = Object.freeze([
  'hash_missed_pieces_total',
  'hash_checked_pieces_total',
  'hash_check_skipped_total',
  'texture_receipts',
  'landed_classes',
  'texture_recheck',
  'textures_made',
]);

/// A parsed copy of a fixture object (STATS_LIVE unless `base` names another) with some fields replaced, for a test
/// that needs its own seq, clock or frames. Only a field the runtime writes may be replaced, an optional part's
/// among them, and inside `segment_open` only its own fields: a test cannot bring back a name the runtime dropped.
export function statsObject(overrides = {}, base = STATS_LIVE) {
  const object = JSON.parse(base);
  const open = JSON.parse(STATS_LIVE).segment_open;
  for (const [key, value] of Object.entries(overrides)) {
    if (!Object.hasOwn(object, key) && !OPTIONAL_KEYS.includes(key)) throw new Error(`${key} is not a field of the runtime's object`);
    if (key === 'segment_open' && value && typeof value === 'object') {
      for (const inner of Object.keys(value)) {
        if (!Object.hasOwn(open, inner)) throw new Error(`segment_open.${inner} is not a field of the runtime's object`);
      }
    }
  }
  return { ...object, ...overrides };
}

/// The presenting thread's phases, by number: budget.rs `NAMES` (the stats object's `budget.phases` leaves out
/// `pool_work` and `relay_land`, `other_threads.phases` leaves out `title`; the budget line names a phase only when it
/// is not zero, and `title` and `unaccounted` always). The one list of phases every reader uses.
export const RUNTIME_PHASES = Object.freeze([
  'title',
  'list_calls',
  'unaccounted',
  'submission_lock_wait',
  'settle',
  'relay_wait',
  'card_wait',
  'sweep',
  'slots',
  'refresh',
  'textures',
  'pool_wait',
  'snapshots',
  'own_lists',
  'send',
  'copy_back',
  'present_call',
  'report',
  'pool_work',
  'relay_land',
  'object_calls',
  'hle_calls',
  'queue_lock_wait',
  'pace_wait',
]);

/// The phases with no processor time of their own (budget.rs `ENTRY_PHASES`): their `cpu_ms` is always null, and
/// the time of all four together is `budget.title_and_calls_cpu_ms`.
export const ENTRY_PHASES = Object.freeze(['title', 'list_calls', 'object_calls', 'hle_calls']);

/// The live registry's names in its own order (live.rs `SPECS`, alphabetical), with the start values of a run with
/// no environment: the `in force` list of a segment line from such a run (`in_force`), the panel's switch left out.
export const IN_FORCE_DEFAULTS = `LIVE_KEYS= LIVE_MARK=0 LIVE_SKIP_DISPATCH= LIVE_SKIP_DRAW= VICTORY_EXACT_BACK=off VICTORY_FETCH_QUEUE=off VICTORY_KEEP_TEXTURES=on VICTORY_LAND_SCAN=off VICTORY_LATE_WAITS=off VICTORY_TEXTURE_HIGH=85 VICTORY_TEXTURE_LOW=70 VICTORY_TEXTURE_MADE=off VICTORY_TEXTURE_RECHECK=off VICTORY_UPLOAD_POOL=off`;
