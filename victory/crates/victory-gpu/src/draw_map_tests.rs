//! Draws and dispatches under the slot map (`VICTORY_SLOT_MAP=1`; design
//! `jobs/2026-11-25-slot-lookup`), end to end: synthetic GFX10 compute programs (wave32,
//! 32 threads, their tables of views at s0, s1 and s2: title root slots 0, 1 and 2)
//! translated by the translator with the map on, made host pipelines as the title's are
//! ([`crate::pipeline::host::HostPipeline::new`]) and bound by [`State::bind`] on the
//! host device (WARP with `VICTORY_D3D12_WARP=1`).
//!
//! The translator reads the switch once per process, so each fixture is a child test
//! run in a process of its own with the map on, its pool capped at [`POOL`] shared
//! slots behind a title heap of [`TITLE`] (far larger, a size of these tests' own), by a parent test.
//! Each program stores what each lane read through table 2's entry 0, a raw unordered
//! view the title wrote; once the dispatch is bound, that view's host slot is pointed at
//! a buffer the processor reads.

use crate::draw::State;
use crate::object::This;
use crate::pipeline::host::HostPipeline;
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType as Root, StageIo, UsedResource};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_SAMPLE_DESC};

/// The title's view heap in these tests: a size of their own, above every slot the
/// fixtures name and above the largest host heap they are run against.
const TITLE: u32 = 1_179_648;
/// Shared host slots the children's pool is capped at (`VICTORY_SLOT_MAP_POOL`).
const POOL: u32 = 4096;
/// Title slot of the view the programs store through (table 2's entry 0).
const OUT: u32 = 4;

/// Runs the child test `name` in a process of its own with the map on and its pool
/// capped, and `env` besides.
fn child(name: &str, env: &[(&str, &str)]) {
    let exe = std::env::current_exe().expect("the test binary");
    let out = std::process::Command::new(exe)
        .args(["--exact", &format!("draw_map_tests::{name}"), "--ignored", "--test-threads", "1", "--nocapture"])
        .env("VICTORY_SLOT_MAP", "1")
        .env("VICTORY_SLOT_MAP_POOL", POOL.to_string())
        .envs(env.iter().copied())
        .output()
        .expect("the child ran");
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    for line in text.lines().filter(|l| l.contains("NOT RUN")) {
        eprintln!("{name}: {line}");
    }
    assert!(out.status.success() && text.contains("1 passed"), "{name}: {text}");
}

/// The child's own check that its parent ran it.
fn mapped() {
    assert!(crate::descriptor::slot_map::on(), "run by its parent, with VICTORY_SLOT_MAP=1");
}

/// `s_mov_b32 s5, 0`, `s_mov_b32 s4, s<table>`: the table at s[4:5].
fn table_at_s4(table: u32) -> [u32; 2] {
    [0xBE85_0380, 0xBE84_0300 | table]
}

/// The output table at s[6:7] and its entry 0 at s[40:43]; v16 the lane's byte
/// (lane * 4).
const OUTPUT: [u32; 6] = [
    0xBE87_0380, // s_mov_b32 s7, 0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xF408_0A03,
    0xFA00_0000, // s_load_dwordx4 s[40:43], s[6:7], 0x0
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
];

/// Reads dword `lane` of the buffer view at entry `entry` of table `table` (an inline
/// constant from -16 to 64), the entry's byte offset shifted into s8 by the code. Each
/// lane stores what it read at its dword of the output.
fn read_shifted(table: u32, entry: i32) -> Vec<u32> {
    let inline = if entry >= 0 { 0x80 + entry as u32 } else { 0xC0 + (-entry) as u32 };
    let mut code = table_at_s4(table).to_vec();
    code.extend(OUTPUT);
    code.extend([
        0x8F08_8500 | inline, // s_lshl_b32 s8, <entry>, 5
        0xF408_0302,
        0x1000_0000, // s_load_dwordx4 s[12:15], s[4:5], s8
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    code
}

/// Entry `k` of root table `slot`, four dwords: a structured buffer view of dwords
/// (`Srv`), or a raw unordered one (`Uav`).
fn entry(slot: u8, k: u32, kind: ResourceType) -> UsedResource {
    UsedResource {
        kind,
        root: Root::Table,
        table_offset_dwords: k * 8,
        lower_bound: 0,
        stride: if kind == ResourceType::Srv { 4 } else { 0 },
        dwords: 4,
        user_data: slot,
        slot,
    }
}

/// `code` translated with the console's list naming `used`, and its host pipeline.
fn translated(code: &[u32], used: Vec<UsedResource>) -> HostPipeline {
    translated_with(code, used, false)
}

/// [`translated`]; with `heap`, the view heap's base at s3 (title root slot 3,
/// `ResourceHeap`) besides the three tables.
fn translated_with(code: &[u32], used: Vec<UsedResource>, heap: bool) -> HostPipeline {
    let host = translated_host(code, &used, heap);
    let shader = crate::pipeline::test_translated_program(host, used);
    HostPipeline::new(&[shader], None).expect("a host pipeline")
}

/// [`translated_with`]'s translation, before its pipeline is made.
fn translated_host(code: &[u32], used: &[UsedResource], heap: bool) -> victory_shader::emit::HostShader {
    // As `pipeline::translate`: the translation depends on the heap's limit.
    crate::descriptor::host::probe_view_limit();
    let param = |root: Root, n: u8| RootParam { root, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: n, slot: n };
    let mut layout = vec![param(Root::Table, 0), param(Root::Table, 1), param(Root::Table, 2)];
    if heap {
        layout.push(param(Root::ResourceHeap, 3));
    }
    let sgprs = layout.len() as u32;
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout,
        used: used.to_vec(),
        io: StageIo::Cs(ComputeIo { rsrc2: sgprs << 1, threads: [32, 1, 1] }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    // A compute program's translation reads nothing else of its container.
    let c = victory_shader::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let p = victory_shader::spill::build(&native).expect("the program decodes");
    let st = victory_shader::structure::structure(&p).expect("the program structures");
    let res = victory_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("traced");
    let io = victory_shader::io::raise(&p, &native).expect("its inputs");
    let host = victory_shader::emit::emit(&c, &native, &p, &st, &io, &res).expect("the program translates");
    assert!(
        crate::descriptor::slot_map::reads_map(&host.root_srvs),
        "translated with the map on: the shader reads it"
    );
    host
}

/// The fixtures' device, the title's heap, and buffers of 32 dwords the processor reads.
struct Bench {
    host: &'static crate::host::Host,
    heap: This,
    /// A sampler heap, bound with the view heap as the title binds them (the host's
    /// root signatures index both directly).
    samplers: This,
    step: usize,
}

/// A buffer of `n` dwords in memory the processor reads.
struct Seen {
    buffer: ID3D12Resource,
    at: *mut u32,
    n: u32,
}

impl Seen {
    fn words(&self) -> Vec<u32> {
        // SAFETY: the buffer's mapping of `n` dwords; no list that writes it runs.
        (0..self.n as usize).map(|i| unsafe { self.at.add(i).read_volatile() }).collect()
    }
}

impl Bench {
    fn new() -> Bench {
        let host = crate::host::test_host().expect("a host device (VICTORY_D3D12_WARP=1 for WARP)");
        let heap = crate::descriptor::test_view_heap(TITLE);
        let samplers = crate::descriptor::test_sampler_heap(16);
        // SAFETY: a plain query.
        let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) } as usize;
        let bench = Bench { host, heap, samplers, step };
        assert!(bench.hosted().mapped.is_some(), "the title's view heap is under the map");
        bench
    }

    fn start(&self) -> u64 {
        crate::descriptor::heap(self.heap).start
    }

    fn hosted(&self) -> &'static crate::descriptor::host::HostHeap {
        crate::descriptor::heap(self.heap).host().expect("a host view heap")
    }

    /// A buffer of `n` dwords, all `fill`.
    fn seen(&self, fill: u32, n: u32) -> Seen {
        let place = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::from(n) * 4,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        // SAFETY: plain object creation from valid descriptions; the mapping holds the
        // 32 dwords filled here.
        unsafe {
            let mut buffer: Option<ID3D12Resource> = None;
            self.host
                .device
                .CreateCommittedResource(&place, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .expect("a buffer the processor reads");
            let buffer = buffer.expect("a buffer the processor reads");
            let mut at = std::ptr::null_mut();
            buffer.Map(0, None, Some(&mut at)).expect("mapped");
            let at = at.cast::<u32>();
            for i in 0..n as usize {
                at.add(i).write(fill);
            }
            Seen { buffer, at, n }
        }
    }

    /// Host slot `index` becomes a raw unordered view of `seen`.
    fn point(&self, index: u32, seen: &Seen) {
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { NumElements: seen.n, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
            },
        };
        // SAFETY: a slot of the live host heap; a live buffer.
        unsafe {
            let start = self.hosted().heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let at = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: start + index as usize * self.step };
            self.host.device.CreateUnorderedAccessView(&seen.buffer, None, Some(&desc), at);
        }
    }

    /// The title writes slot `k`: a view of 256 bytes of title buffer memory (64 dwords,
    /// dword `i` holding `tag + i`), a structured shader view of dwords, or with `uav` a
    /// raw unordered one.
    fn view(&self, id: usize, k: u32, tag: u32, uav: bool) {
        let memory: &'static mut [u32] = Box::leak((0..64).map(|i| tag + i).collect::<Vec<u32>>().into_boxed_slice());
        let address = memory.as_ptr() as u64;
        crate::shadow::register(id, address, 256);
        let mut resource = [0u32; 14];
        resource[0] = 1; // a buffer
        resource[4] = 256;
        let mut desc = [0u32; 16];
        if uav {
            desc[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
            desc[1] = D3D12_UAV_DIMENSION_BUFFER.0 as u32;
            desc[4] = 64;
            desc[8] = D3D12_BUFFER_UAV_FLAG_RAW.0 as u32;
        } else {
            desc[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
            desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc[6] = 64;
            desc[7] = 4;
        }
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        let at = self.start() + u64::from(k) * 32;
        if uav {
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &desc, at);
        } else {
            crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, at);
        }
    }

    /// `p` bound with tables 0 and 1 at title slots `tables` and table 2 at [`OUT`];
    /// `before` runs between `bind` and the dispatch. What each lane stored through
    /// table 2 (`0xdeadbeef` where it stored nothing).
    fn run(&self, p: &HostPipeline, tables: [u32; 2], before: impl FnOnce(&State)) -> Vec<u32> {
        self.run_out(p, tables, 32, before)
    }

    /// [`Bench::run`] with an output of `dwords` dwords.
    fn run_out(&self, p: &HostPipeline, tables: [u32; 2], dwords: u32, before: impl FnOnce(&State)) -> Vec<u32> {
        let seen = self.seen(0xdead_beef, dwords);
        let list = self.host.new_list().expect("a host list");
        let mut state = State::default();
        state.set_heaps(&[self.heap, self.samplers]);
        for (slot, k) in tables.into_iter().chain([OUT]).enumerate() {
            state.set_table(true, slot as u32, self.start() + u64::from(k) * 32);
        }
        assert!(state.bind(&list, p, true), "the dispatch is bound");
        // The output's host slot, as the map names it.
        let out = self.hosted().host_slot(OUT).expect("the output's title slot is mapped");
        self.point(out, &seen);
        before(&state);
        // SAFETY: an open list with the program bound by `bind`.
        unsafe { list.list.Dispatch(1, 1, 1) };
        list.close();
        self.host.execute(&[&list]);
        // SAFETY: a plain query.
        unsafe { self.host.device.GetDeviceRemovedReason() }.expect("the device is not removed");
        seen.words()
    }
}

/// What lane `lane` reads of a view whose dword `i` holds `tag + i`.
fn of(tag: u32) -> Vec<u32> {
    (0..32).map(|lane| tag + lane).collect()
}

/// Each lane reads dword `lane` of the constant buffer at entry 0 of table 0
/// (`buffer_load_dword` through its descriptor, at the lane's byte `lane * 4`) and
/// stores it at its dword of the output: [`read_shifted`]'s registers, the entry's
/// offset an immediate 0.
fn read_constants() -> Vec<u32> {
    let mut code = table_at_s4(0).to_vec();
    code.extend(OUTPUT);
    code.extend([
        0xF408_0302,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], 0x0
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    code
}

impl Bench {
    /// The title writes slot `k`: a constant buffer view of 128 bytes from byte `from`
    /// of 512 bytes of title buffer memory (dword `i` holding `tag + i`).
    fn constants(&self, id: usize, k: u32, tag: u32, from: u64) {
        let memory: &'static mut [u32] = Box::leak((0..128).map(|i| tag + i).collect::<Vec<u32>>().into_boxed_slice());
        let address = memory.as_ptr() as u64;
        crate::shadow::register(id, address, 512);
        let at = address + from;
        let desc = [at as u32, (at >> 32) as u32, 128, 0];
        crate::descriptor::create_cbv(std::ptr::null(), &desc, self.start() + u64::from(k) * 32);
    }
}

/// A table's constant buffer whose record starts 48 bytes into its memory (16-byte
/// aligned, as the console requires; not D3D12's 256) is read by every lane exactly,
/// through the map: the host makes the record a raw view of the memory's host copy,
/// and the draw brings its range up to date. Before, the host made such a record no
/// view at all (a constant buffer view must start on 256 bytes), and it read zeros.
#[test]
fn a_table_constant_buffer_on_16_bytes_is_read_through_the_map() {
    child("a_table_constant_buffer_on_16_bytes_is_read_through_the_map_child", &[]);
}

#[test]
#[ignore = "a child process of the test above (VICTORY_SLOT_MAP=1)"]
fn a_table_constant_buffer_on_16_bytes_is_read_through_the_map_child() {
    mapped();
    let bench = Bench::new();
    let table = TITLE - 2000;
    bench.constants(0x5eed_0311, table, 0xc000, 48);
    bench.view(0x5eed_0312, OUT, 0, true);
    let used = vec![entry(0, 0, ResourceType::Cbv), entry(2, 0, ResourceType::Uav)];
    let p = translated(&read_constants(), used);
    let read = bench.run(&p, [table, table], |_| {});
    assert_eq!(read, of(0xc000 + 12), "the record's dwords from byte 48, every lane");
}

/// A dispatch whose shader reads the map is bound and runs with the view heap under
/// the map: it reads the view at a title slot far past the host heap's end (1,244,184,
/// behind a heap of 64 + 65,536 + 4,096 slots) through the map, exactly, for every lane.
/// Before, the bind guard skipped every such dispatch.
#[test]
fn a_dispatch_reading_the_map_runs_and_reads_a_view_past_the_host_heap() {
    child("a_dispatch_reading_the_map_runs_and_reads_a_view_past_the_host_heap_child", &[]);
}

#[test]
#[ignore = "a child process of the test above (VICTORY_SLOT_MAP=1)"]
fn a_dispatch_reading_the_map_runs_and_reads_a_view_past_the_host_heap_child() {
    mapped();
    let bench = Bench::new();
    let high = TITLE - 1000;
    bench.view(0x5eed_0301, high, 0xa000, false);
    bench.view(0x5eed_0302, OUT, 0, true);
    let used = vec![entry(0, 0, ResourceType::Srv), entry(1, 0, ResourceType::Srv), entry(2, 0, ResourceType::Uav)];
    let p = translated(&read_shifted(0, 0), used);
    let read = bench.run(&p, [high, high], |_| {});
    assert_eq!(read, of(0xa000), "the view at title slot {high}, through the map");
    let host = bench.hosted().host_slot(high).expect("mapped");
    assert!(host < 64 + 65_536 + POOL, "a host slot of the capped heap: {host}");
}

/// SOP1 `s_mov_b32 s<dst>, <src>` (an SGPR, 0x80 the inline 0, 0xFF a literal next).
const fn s_mov(dst: u32, src: u32) -> u32 {
    0xBE80_0300 | dst << 16 | src
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], <imm>` (signed, 21 bits).
const fn s_load4(dst: u32, base: u32, imm: i32) -> [u32; 2] {
    [0xF408_0000 | dst << 6 | base >> 1, 0xFA00_0000 | (imm as u32 & 0x1F_FFFF)]
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], s<soffset>`.
const fn s_load4_by(dst: u32, base: u32, soffset: u32) -> [u32; 2] {
    [0xF408_0000 | dst << 6 | base >> 1, soffset << 25]
}

/// Each lane loads dword `lane` of the buffer descriptor at s[12:15] and stores it at
/// its dword of output part `part` (32 dwords a part).
fn load_store(part: u32) -> [u32; 6] {
    [
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000 | (part * 128),
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen offset:part*128
    ]
}

/// The acceptance fixture's title slots (Neon's, `jobs/2026-11-25-dynamic-slots`): the
/// mixed table near the title heap's end (its entry 8 lies past it), a table at a title
/// slot numbered like a slot of the host's ring, and two views a program picks from the
/// heap by index: one in the band where NVIDIA's capped mirror put the copy ring, one
/// past NVIDIA's limit of 1,000,000.
const MIXED: u32 = TITLE - 6;
const LOW: u32 = 100;
const BAND: u32 = 800_000;
const PAST: u32 = 1_100_000;

/// One program reads, part by part: the mixed table's entries 0 (a structured view), 1
/// (a constant buffer), 2 (never written), 3 (an unordered view, read as a shader
/// view), 4 (at an offset the code shifts into place), 8 (past the title's heap) and
/// -1 (32 bytes before the table: the title's own neighbour); entry 0 of the table at
/// [`LOW`]; and the views at [`BAND`] and [`PAST`], picked from the heap by index.
fn read_everything() -> Vec<u32> {
    let mut code = table_at_s4(0).to_vec();
    code.extend(OUTPUT);
    // s[20:21]: table 1; s[18:19]: the heap.
    code.extend([s_mov(21, 0x80), s_mov(20, 1), s_mov(19, 0x80), s_mov(18, 3)]);
    let by = s_load4_by(12, 4, 8);
    let picked = s_load4_by(12, 18, 8);
    let parts: [Vec<u32>; 10] = [
        s_load4(12, 4, 0).to_vec(),
        s_load4(12, 4, 32).to_vec(),
        s_load4(12, 4, 64).to_vec(),
        s_load4(12, 4, 96).to_vec(),
        vec![0x8F08_8584, by[0], by[1]], // s_lshl_b32 s8, 4, 5
        s_load4(12, 4, 256).to_vec(),
        s_load4(12, 4, -32).to_vec(),
        s_load4(12, 20, 0).to_vec(),
        vec![s_mov(8, 0xFF), BAND * 32, picked[0], picked[1]],
        vec![s_mov(8, 0xFF), PAST * 32, picked[0], picked[1]],
    ];
    for (part, load) in parts.iter().enumerate() {
        code.extend_from_slice(load);
        code.extend(load_store(part as u32));
    }
    code.push(0xBF81_0000); // s_endpgm
    code
}

/// Neon's acceptance fixture (`jobs/2026-11-25-dynamic-slots/neon-brainstorm-10-report.md`),
/// with the debug layer and GPU-based validation on: a host view heap capped far below
/// the title's ([`POOL`] shared slots behind [`TITLE`]), its ring of the host's own slots
/// taken and holding poison views, and one dispatch that reads live views at title slots
/// past the cap, in the band NVIDIA's capped mirror gave the ring and past NVIDIA's
/// limit (picked from the heap by index), at a title slot numbered like a ring slot, and
/// every entry of a mixed table: a structured view, a constant buffer on 16 bytes, a
/// slot never written, a view of the wrong kind, an entry at a shifted offset, one past
/// the title's heap and the title's own neighbour before it. Every live view reads
/// exactly; the empty, the wrong-kind and the past-the-heap entries read zeros; nothing
/// reads or writes the poison; the debug layer has nothing to say.
#[test]
fn the_acceptance_fixture_reads_every_view_right_through_a_capped_heap() {
    child(
        "the_acceptance_fixture_reads_every_view_right_through_a_capped_heap_child",
        &[("VICTORY_D3D12_DEBUG", "1"), ("VICTORY_D3D12_GBV", "1")],
    );
}

#[test]
#[ignore = "a child process of the test above (VICTORY_SLOT_MAP=1, the debug layer and GPU-based validation on)"]
fn the_acceptance_fixture_reads_every_view_right_through_a_capped_heap_child() {
    use crate::descriptor::slot_map::POOL_BASE;
    mapped();
    let bench = Bench::new();
    if crate::host::take_host_debug_errors(bench.host).is_none() {
        eprintln!("NOT RUN (no debug layer): the debug layer's and GPU-based validation's check; the rest is checked");
    }
    // The ring: taken and holding poison views.
    let poison = bench.seen(0xbad0_0000, 32);
    let host = bench.hosted();
    let ring = host.dynamic(64).expect("ring slots");
    assert!(ring < POOL_BASE && (ring..ring + 64).contains(&LOW), "the ring holds slot {LOW}'s number: {ring}");
    for k in 0..64 {
        host.raw_uav_at(ring + k, &poison.buffer, 0, 128);
    }
    bench.view(0x5eed_0401, MIXED, 0xa000, false);
    bench.constants(0x5eed_0402, MIXED + 1, 0xc000, 48);
    bench.view(0x5eed_0403, MIXED + 3, 0x3000, true);
    bench.view(0x5eed_0404, MIXED + 4, 0xb000, false);
    bench.view(0x5eed_0405, MIXED - 1, 0xe000, false);
    bench.view(0x5eed_0406, LOW, 0xd000, false);
    bench.view(0x5eed_0407, BAND, 0xf000, false);
    bench.view(0x5eed_0408, PAST, 0x9000, false);
    bench.view(0x5eed_0409, OUT, 0, true);
    let used = vec![
        entry(0, 0, ResourceType::Srv),
        entry(0, 1, ResourceType::Cbv),
        entry(0, 3, ResourceType::Srv),
        entry(1, 0, ResourceType::Srv),
        entry(2, 0, ResourceType::Uav),
    ];
    let p = translated_with(&read_everything(), used, true);
    assert!(
        p.stages[0].translated().host.root_srvs.contains(&victory_shader::emit::SLOT_STRIDES),
        "the heap picks read the strides: the host makes every buffer view of the heap ready"
    );
    let read = bench.run_out(&p, [MIXED, LOW], 320, |_| {});
    let zeros = vec![0; 32];
    let want: [(&str, Vec<u32>); 10] = [
        ("entry 0: a structured view", of(0xa000)),
        ("entry 1: a constant buffer on 16 bytes", of(0xc000 + 12)),
        ("entry 2: never written", zeros.clone()),
        ("entry 3: an unordered view read as a shader view", zeros.clone()),
        ("entry 4: at a shifted offset", of(0xb000)),
        ("entry 8: past the title's heap", zeros.clone()),
        ("entry -1: the title's neighbour", of(0xe000)),
        ("a table at a ring slot's number", of(0xd000)),
        ("picked in the band of the old ring", of(0xf000)),
        ("picked past 1,000,000", of(0x9000)),
    ];
    for (part, (what, want)) in want.iter().enumerate() {
        assert_eq!(&read[part * 32..part * 32 + 32], &want[..], "part {part}, {what}");
    }
    assert_eq!(poison.words(), vec![0xbad0_0000; 32], "nothing wrote through the ring");
    for t in [MIXED, MIXED + 1, MIXED + 4, MIXED - 1, LOW, BAND, PAST, OUT] {
        let h = host.host_slot(t).unwrap_or_else(|| panic!("title slot {t} is mapped"));
        assert!((POOL_BASE..POOL_BASE + POOL).contains(&h), "title slot {t} reads pool slot {h}");
    }
    if let Some(errors) = crate::host::take_host_debug_errors(bench.host) {
        assert!(errors.is_empty(), "the debug layer said: {errors:#?}");
    }
}

/// Each lane's d16 load (`buffer_load_format_d16_x`) of element `lane` of the typed view
/// at entry 0 of table 0, stored at its dword of the output.
fn read_d16() -> Vec<u32> {
    let mut code = table_at_s4(0).to_vec();
    code.extend(OUTPUT);
    code.extend([
        0xF408_0302,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], 0x0
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0x7E18_0280, // v_mov_b32 v12, 0
        0xE200_2000,
        0x8003_0C00, // buffer_load_format_d16_x v12, v0, s[12:15], 0 idxen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    code
}

impl Bench {
    /// The title writes slot `k`: a typed shader view of DXGI format `format` over 64
    /// elements of 2 bytes of title buffer memory, element `i` holding `0xa000 + i`.
    fn typed_view(&self, id: usize, k: u32, format: u32) {
        let memory: &'static mut [u32] =
            Box::leak((0..32u32).map(|j| (0xa000 + 2 * j) | (0xa001 + 2 * j) << 16).collect::<Vec<u32>>().into_boxed_slice());
        let address = memory.as_ptr() as u64;
        crate::shadow::register(id, address, 128);
        let mut resource = [0u32; 14];
        resource[0] = 1; // a buffer
        resource[4] = 128;
        let mut desc = [0u32; 16];
        desc[0] = format;
        desc[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[6] = 64;
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(std::ptr::null(), &resource, &desc, self.start() + u64::from(k) * 32);
    }
}

/// Root-signature DWORDs of a stage as the builder lays it out by default: user data;
/// 2 per root CBV and per root view of ours; 1 per title root SRV or UAV (a one-view
/// table).
fn root_dwords(host: &victory_shader::emit::HostShader) -> u32 {
    use victory_shader::emit::{SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES, slot_map::{MAP_DIR, MAP_POOL}};
    let ours = |s: &u8| [SLOT_STRIDES, SLOT_MEMORY, MAP_DIR, MAP_POOL].contains(s);
    host.user_data_dwords()
        + 2 * host.root_cbvs.len() as u32
        + host.root_srvs.iter().map(|s| if ours(s) { 2 } else { 1 }).sum::<u32>()
        + host.root_uavs.iter().map(|&s| if s == SLOT_FAULTS { 2 } else { 1 }).sum::<u32>()
}

/// With `VICTORY_D16_INTEGER_VIEWS=1` (plan K5): a d16 load through a title view of
/// R16_UINT reads each element's bits, under the map (the view's kind by host slot), with
/// the slot table a root descriptor and, for a stage whose root signature had 63 DWORDs
/// without it, a one-view table. The debug layer and GPU-based validation are on.
#[test]
fn a_d16_load_through_an_integer_view_reads_its_bits_through_either_binding_of_the_slot_table() {
    child(
        "a_d16_load_through_an_integer_view_reads_its_bits_through_either_binding_of_the_slot_table_child",
        &[("VICTORY_D16_INTEGER_VIEWS", "1"), ("VICTORY_D3D12_DEBUG", "1"), ("VICTORY_D3D12_GBV", "1")],
    );
}

#[test]
#[ignore = "a child process of the test above (VICTORY_SLOT_MAP=1, VICTORY_D16_INTEGER_VIEWS=1, the debug layer and GPU-based validation on)"]
fn a_d16_load_through_an_integer_view_reads_its_bits_through_either_binding_of_the_slot_table_child() {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_UINT};
    mapped();
    assert!(crate::descriptor::host::view_kind::on(), "run by its parent, with VICTORY_D16_INTEGER_VIEWS=1");
    let bench = Bench::new();
    if crate::host::take_host_debug_errors(bench.host).is_none() {
        eprintln!("NOT RUN (no debug layer): the debug layer's and GPU-based validation's check; the rest is checked");
    }
    let slot = TITLE - 3000;
    bench.typed_view(0x5eed_0501, slot, DXGI_FORMAT_R16_UINT.0 as u32);
    bench.view(0x5eed_0502, OUT, 0, true);
    let typed = UsedResource { stride: 0, ..entry(0, 0, ResourceType::Srv) };
    let used = vec![typed, entry(2, 0, ResourceType::Uav)];
    let host = translated_host(&read_d16(), &used, false);
    assert!(host.root_srvs.contains(&victory_shader::emit::SLOT_STRIDES), "with the switch the load reads the slot table");
    let bits: Vec<u32> = (0..32).map(|lane| 0xa000 + lane).collect();
    // As the program is: the slot table a root descriptor.
    let p = HostPipeline::new(&[crate::pipeline::test_translated_program(host.clone(), used.clone())], None)
        .expect("a host pipeline");
    assert!(!p.stages[0].strides_table);
    assert_eq!(bench.run(&p, [slot, slot], |_| {}), bits, "R16_UINT, the slot table a root descriptor");
    // Brought to 63 DWORDs without the slot table by title root SRVs it does not read.
    let mut padded = host;
    let without = root_dwords(&padded) - 2;
    padded.root_srvs.extend((0..63 - without).map(|k| 100 + k as u8));
    assert_eq!(root_dwords(&padded), 65);
    let p = HostPipeline::new(&[crate::pipeline::test_translated_program(padded, used.clone())], None)
        .expect("a stage of 63 DWORDs and the slot table has its pipeline");
    assert!(p.stages[0].strides_table, "the slot table a one-view table");
    assert_eq!(bench.run(&p, [slot, slot], |_| {}), bits, "R16_UINT, the slot table a one-view table");
    // The same title slot rewritten as R16_FLOAT: read by its new kind, as halves.
    bench.typed_view(0x5eed_0503, slot, DXGI_FORMAT_R16_FLOAT.0 as u32);
    let halves = bench.run(&p, [slot, slot], |_| {});
    assert_eq!(halves, bits, "R16_FLOAT: a half widened and narrowed again is itself (0xa000 on are numbers)");
    if let Some(errors) = crate::host::take_host_debug_errors(bench.host) {
        assert!(errors.is_empty(), "the debug layer said: {errors:#?}");
    }
}
