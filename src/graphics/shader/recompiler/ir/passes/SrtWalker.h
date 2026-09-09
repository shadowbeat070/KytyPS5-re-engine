#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>
#include <string_view>

namespace Libs::Graphics::ShaderRecompiler::IR {

class Value;

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	SrtMemoryReader           read_memory                = nullptr;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
};

enum class RuntimeValueType { Any, Integer };

// Why a value cannot be re-executed on the host. Reported next to the rejected descriptor dword
// so the log names the instruction that stopped the walk instead of only the dword index.
enum class RuntimeValueReject {
	None,
	// The chain reached an opcode the host evaluator has no rule for.
	UnsupportedOpcode,
	// A leaf operand the host cannot hold: a register handle or an opaque type.
	UnsupportedOperand,
	// Host floating point does not model the shader rounding and denormal modes.
	FloatInIntegerChain,
	// Wrong arity, a non-immediate index, or a handle of the wrong kind.
	MalformedInstruction,
	// An undefined or void value.
	UndefinedValue,
	// A definition cycle with no loop-invariant phi to break it.
	CyclicValue,
	// Not a 32-bit scalar, so it cannot be a descriptor dword at all.
	NonScalarType,
};

// Trivially copyable and default constructed by the caller, so recording a reason allocates
// nothing and costs nothing on the accepting path.
struct RuntimeValueFailure {
	RuntimeValueReject reason     = RuntimeValueReject::None;
	ValueOpcode        opcode     = ValueOpcode::Void;
	bool               has_opcode = false;
};

[[nodiscard]] std::string_view RuntimeValueRejectName(RuntimeValueReject reason);

// Optionally reports why the first rejected instruction could not be re-executed. Pass a sink
// only where that reason is logged: it is written at most once, and only when validation fails.
bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType     type    = RuntimeValueType::Any,
                          RuntimeValueFailure* failure = nullptr);
// Uses the strict reader for values that affect shader specialization.
SrtRuntime CleanRuntime(SrtRuntime runtime);


// One memoized evaluation session shared by the entire shader resource refresh.
class SrtWalker {
public:
	SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
	          std::span<const uint8_t> clean_flat_slots = {}, SrtWalker* clean_evaluator = nullptr,
	          Value active_mask = {});
	~SrtWalker();
	SrtWalker(const SrtWalker&)            = delete;
	SrtWalker& operator=(const SrtWalker&) = delete;

	bool Evaluate(Value value, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Refreshes reachable scalar reads and active descriptor sources in one walk.
	bool RefreshFlatBuffer(std::vector<uint32_t>& flat);

private:
	static ResourcePlan::EvaluationContext& AcquireContext(const ResourcePlan& program);
	static float Float32(uint64_t bits);
	bool EvaluateWide(Value value, uint64_t& result);
	bool Arg(const Inst& inst, size_t index, uint64_t& result);
	bool EvaluatePhi(const Inst& inst, uint64_t& result);
	bool EvaluateExtract(const Inst& inst, uint64_t& result);
	bool EvaluateRawRead(const Inst& inst, uint64_t& result);
	bool EvaluateInst(const Inst& inst, uint64_t& result);

	const ResourcePlan&              m_program;
	SrtRuntime                      m_runtime;
	std::span<const uint8_t>         m_clean_flat_slots;
	SrtWalker*                      m_clean_evaluator = nullptr;
	Value                           m_active_mask;
	ResourcePlan::EvaluationContext& m_context;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
