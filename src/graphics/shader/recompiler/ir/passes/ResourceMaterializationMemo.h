#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATIONMEMO_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATIONMEMO_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Read from the environment once; tests may assign it.
struct MaterializeMemoConfig {
	// KYTY_MATERIALIZE_MEMO_WAYS: recordings kept per cached stage.
	uint32_t ways = 8;
	// KYTY_MEMO_EXACT_WORKGROUPS=1: a dispatch size change misses even without indirect tables.
	bool exact_workgroups = false;
	// KYTY_MEMO_NO_REFUSAL_STEPS=1: a walk that saw a read refuse is not memoized.
	bool refusal_steps = true;
	// KYTY_MEMO_NO_BINDLESS=1: plans with bindless image heaps are not memoized.
	bool bindless = true;
	// KYTY_MEMO_NO_CAPTURE_RELOCATION=1: a plan that captures its read ranges never relocates.
	bool capture_relocation = true;
	// KYTY_MEMO_NO_INTEGER_OPS=1: offsets built with shifts, masks or multiplies stay pinned.
	bool integer_ops = true;
	// KYTY_MEMO_NO_SELF_COMPARE=1: a word only compared with itself still counts as structure.
	bool self_compare = true;

	static MaterializeMemoConfig& Get();
};

// Records the guest reads of one cached stage's materialization and replays them on the next
// draw: words that only feed addresses or flat SRT slots are re-read, possibly at relocated
// addresses, and every other input must be unchanged, so a replay yields exactly what a walk would.
class MaterializeMemo {
public:
	enum class Miss : uint8_t {
		None,
		Empty,
		Unsupported,
		Unreplayable,
		InputChanged,
		StructureChanged,
		ReadRefused,
		AddressChanged,
	};

	// Indirect tables also depend on process-wide key state; only bindless heaps are memoized.
	[[nodiscard]] static bool Supports(const ResourcePlan& plan);

	// The runtime to walk with: it forwards every callback and records the guest reads.
	[[nodiscard]] SrtRuntime Record(const SrtRuntime& runtime);
	// Keeps the recording after a successful walk, or forgets the memo if it cannot be replayed.
	void Commit(const ResourcePlan& plan, const ResourceSnapshot& snapshot);
	void Invalidate();
	// Re-reads through `runtime` and refreshes `snapshot` when that equals walking again.
	bool Replay(const ResourcePlan& plan, const SrtRuntime& runtime, ResourceSnapshot& snapshot);
	// Shares another recording's per-plan analysis instead of deriving it again.
	void               AdoptAnalysis(const MaterializeMemo& other);
	[[nodiscard]] bool Valid() const { return m_valid; }
	[[nodiscard]] Miss LastMiss() const { return m_miss; }

private:
	enum class Reader : uint8_t { Memory, Specialization, Condition };

	struct Read {
		Reader                  reader          = Reader::Memory;
		uint32_t                component_bytes = 0;
		const Inst*             inst            = nullptr;
		uint64_t                address         = 0;
		uint32_t                word            = 0;
		std::array<uint64_t, 5> operands {};
		uint32_t                slot = UINT32_MAX;
		// The reader refused; the walk took its fallback, so a replay must see it refuse again.
		bool refused = false;
	};

	// A postfix program that re-derives one operand from user data and earlier words.
	struct Op {
		enum Kind : uint8_t {
			Imm,
			UserData,
			Key,
			Add32,
			CarryLow,
			CarryHigh,
			Sub32,
			Mul32,
			And32,
			Or32,
			Shl32,
			Shr32
		};
		Kind     kind  = Imm;
		uint32_t index = 0;
		uint64_t imm   = 0;
	};

	struct Step {
		uint32_t key = 0;
		// Operands with a program are re-derived; the rest keep their recorded value.
		std::array<uint32_t, 5> op_begin {};
		std::array<uint32_t, 5> op_end {};
		uint8_t                 derived_mask = 0;
		bool                    structural   = true;
		// The walk asks gpu_owned before this read and refuses it when the GPU owns the bytes.
		bool     ownership_checked = false;
		bool     refused           = false;
		uint32_t slot_begin        = 0;
		uint32_t slot_end          = 0;
		// The snapshot.specialization_reads entry this read made, or UINT32_MAX.
		uint32_t capture = UINT32_MAX;
	};

	struct Analysis;

	static bool ReadMemory(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadSpecialization(void* userdata, uint64_t address, std::span<uint32_t> values);
	static bool ReadCondition(void* userdata, uint64_t address, std::span<uint32_t> values);
	static uint64_t    ReadableExtent(void* userdata, uint64_t address, uint64_t size);
	static const char* DescribeRefusal(void* userdata, uint64_t address);
	static bool        ReadBlock(void* userdata, uint64_t address, void* data, uint64_t size);
	static bool        HashBlock(void* userdata, uint64_t address, uint64_t size, uint64_t* hash);
	static bool        GpuOwned(void* userdata, uint64_t address, uint64_t size);
	bool Note(Reader reader, bool read, uint64_t address, std::span<const uint32_t> values);
	bool Build(const ResourcePlan& plan, const ResourceSnapshot& snapshot);
	bool CanDerive(const ResourcePlan& plan, Value value, uint32_t before) const;
	void Compile(const ResourcePlan& plan, Value value);
	[[nodiscard]] uint64_t Evaluate(uint32_t begin, uint32_t end,
	                                std::span<const uint32_t> user_data) const;
	[[nodiscard]] bool     SameShape(const ResourcePlan& plan) const;
	[[nodiscard]] uint32_t KeyOf(Value value) const;

	SrtRuntime        m_source;
	std::vector<Read> m_reads;
	bool              m_replayable = false;

	const ResourcePlan*                    m_plan = nullptr;
	std::shared_ptr<Analysis>              m_analysis;
	std::vector<Read>                      m_recorded;
	std::vector<Step>                      m_steps;
	std::vector<uint32_t>                  m_slots;
	std::unordered_map<uint64_t, uint32_t> m_keys;
	std::vector<uint32_t>                  m_key_first;
	std::vector<uint8_t>                   m_fixed_registers;
	std::vector<uint32_t>                  m_user_data;
	std::vector<uint32_t>                  m_workgroup_counts;
	uint64_t                               m_shader_base       = 0;
	bool                                   m_capture_addresses = false;
	// Every captured range is one recorded read, so a relocated read can move its range too.
	bool                          m_capture_mapped = false;
	bool                          m_valid          = false;
	bool                          m_shape_ok       = false;
	Miss                          m_miss           = Miss::Empty;
	mutable std::vector<uint32_t> m_key_words;
	std::vector<uint32_t>         m_new_words;
	std::vector<uint64_t>         m_new_addresses;
	std::vector<Op>               m_ops;
	std::vector<uint8_t>          m_visited_blocks;
	mutable std::vector<uint64_t> m_stack;

public:
	MaterializeMemo();
	~MaterializeMemo();
	MaterializeMemo(MaterializeMemo&&) noexcept;
	MaterializeMemo& operator=(MaterializeMemo&&) noexcept;
};

// A few recordings per cached stage, each with the snapshot and specialization its walk produced,
// so a stage that alternates between bindings replays instead of walking on every switch.
class MaterializeMemoWays {
public:
	using Miss = MaterializeMemo::Miss;

	// Replays the matching recording into `snapshot` and `specialization`.
	bool Replay(const ResourcePlan& plan, const SrtRuntime& runtime, ResourceSnapshot& snapshot,
	            ResourceSpecialization& specialization);
	[[nodiscard]] SrtRuntime Record(const SrtRuntime& runtime);
	void                     Commit(const ResourcePlan& plan, const ResourceSnapshot& snapshot,
	                                const ResourceSpecialization& specialization);
	[[nodiscard]] bool       Valid() const;
	[[nodiscard]] Miss       LastMiss() const { return m_miss; }

private:
	struct Way {
		MaterializeMemo        memo;
		ResourceSnapshot       snapshot;
		ResourceSpecialization specialization;
		uint64_t               used = 0;
	};

	std::vector<Way> m_ways;
	uint32_t         m_current   = UINT32_MAX;
	uint32_t         m_recording = UINT32_MAX;
	uint64_t         m_clock     = 0;
	Miss             m_miss      = Miss::Empty;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATIONMEMO_H_
