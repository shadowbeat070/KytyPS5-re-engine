#include "graphics/host_gpu/renderer/cache/bindlessTranslation.h"
#include "graphics/host_gpu/renderer/pipeline/shaderReadCache.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <memory>
#include <vector>

namespace {

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "ResourceMaterializationTests: failed: %s\n", text);
    std::abort();
  }
}

bool RejectSpecializationRead(void *userdata, uint64_t, std::span<uint32_t>) {
  ++*static_cast<uint32_t *>(userdata);
  return false;
}

Libs::Graphics::ShaderRecompiler::IR::Block &
AddValueBlock(Libs::Graphics::ShaderRecompiler::IR::Program &program) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto block = std::make_unique<Block>();
  auto *result = block.get();
  program.blocks.push_back(result);
  program.block_info.push_back({.id = 0});
  program.block_storage.push_back(std::move(block));
  return *result;
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan SrtPlan(uint64_t address) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto low = Value(static_cast<uint32_t>(address));
  const auto high = Value(static_cast<uint32_t>(address >> 32u));
  auto &handle =
      value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
  auto &raw = value_block.AppendNewInst(
      ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(0u), Value(0u), Value(true)});
  raw.SetFlags(MemoryFlags{.index = 0, .pc = 0x40});
  program.srt_reads.push_back({Value(&raw), 0});

  auto &srt = value_block.AppendNewInst(ValueOpcode::GetSrtResource);
  auto &flat = value_block.AppendNewInst(ValueOpcode::ReadConst,
                                         {Value(&srt), Value(0u)});
  DescriptorSource source;
  source.dwords[0] = Value(&flat);
  source.dwords[1] = Value(0u);
  source.dword_count = 2;
  program.descriptor_sources.push_back(source);
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UnbasedFlatPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  AddValueBlock(program);
  program.info.uses_dma = true;
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::ResourcePlan UserDataBufferPlan() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &value_block = AddValueBlock(program);

  auto &user_data = value_block.AppendNewInst(
      ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0))});
  DescriptorSource source;
  source.dwords[0] = Value(&user_data);
  source.dwords[1] = Value(0u);
  source.dwords[2] = Value(0u);
  source.dwords[3] = Value(0u);
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0});
  return ExtractResourcePlan(program);
}

Libs::Graphics::ShaderRecompiler::IR::Program MixedSamplerProgram() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);

  const auto AddSource = [&program](uint32_t dword_count) {
    DescriptorSource source;
    source.dword_count = dword_count;
    for (uint32_t i = 0; i < dword_count; i++) {
      source.dwords[i] = Value(0u);
    }
    program.descriptor_sources.push_back(source);
    return static_cast<uint32_t>(program.descriptor_sources.size() - 1u);
  };

  const auto image0 = AddSource(8);
  const auto image1 = AddSource(8);
  const auto sampler0 = AddSource(4);
  const auto sampler1 = AddSource(4);
  program.descriptor_sources[image1].dwords[0] = Value(1u);
  program.descriptor_sources[image1].dwords[1] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::BufferFormat::k11_11_10UInt) << 20u);
  program.descriptor_sources[image1].dwords[3] = Value(static_cast<uint32_t>(
      Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
  for (uint32_t index = 0; index < 2; ++index) {
    auto &value = block.AppendNewInst(
        ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(index))});
    program.descriptor_sources[sampler0 + index].dwords[0] = Value(&value);
  }
  program.info.images.push_back(
      {.source = image0,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.images.push_back(
      {.source = image1,
       .resource_class = ImageResourceClass::Sampled,
       .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
       .dimension =
           Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  program.info.samplers.push_back({.source = sampler0});
  program.info.samplers.push_back({.source = sampler1});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 0});
  program.info.sampled_pairs.push_back({.image = 0, .sampler = 1});
  program.info.sampled_pairs.push_back({.image = 1, .sampler = 1});
  return program;
}

void TestMappedSrtUsesDirectReaderByDefault() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const uint32_t dword = 0x12345678;
  auto plan = SrtPlan(reinterpret_cast<uint64_t>(&dword));
  uint32_t specialization_reads = 0;
  const SrtRuntime runtime{.userdata = &specialization_reads,
                           .read_specialization_memory =
                               RejectSpecializationRead};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mapped SRT stage materialization failed");
  Check(specialization_reads == 0,
        "ordinary SRT read used the specialization reader");
  Check(snapshot.flattened_srt.size() == 1 &&
            snapshot.flattened_srt[0] == dword,
        "cache rematerialization did not use the direct reader by default");
}

void TestIntegerRuntimeValueFollowsSrtReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = SrtPlan(0x10000);
  const auto root = plan.descriptor_sources.front().dwords[0];
  Check(ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT read was rejected");

  Block values;
  auto &comparison = values.AppendNewInst(ValueOpcode::FPOrdLessThanEqual32,
                                          {Value::F32(1.f), Value::F32(0.f)});
  auto &selection = values.AppendNewInst(
      ValueOpcode::SelectU32, {Value(&comparison), Value(1u), Value(0u)});
  plan.srt_reads[0].value = Value(&selection);
  Check(ValidateRuntimeValue(plan, root),
        "ordinary SRT validation rejected a floating-point dependency");
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "integer SRT validation missed a hidden floating-point dependency");

  auto &first =
      values.AppendNewInst(ValueOpcode::ReadFirstLane, {root, Value(true)});
  Check(!ValidateRuntimeValue(plan, Value(&first), RuntimeValueType::Integer),
        "read-first-lane lost integer-only SRT validation");

  auto &active = values.AppendNewInst(ValueOpcode::ReadFirstLane,
                                      {Value(&selection), Value(&comparison)});
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point execution mask was accepted as integer-only");

  auto &lane = values.AppendNewInst(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)),
       Value(0u)});
  auto &mask =
      values.AppendNewInst(ValueOpcode::INotEqual32, {Value(&lane), Value(0u)});
  selection.SetArg(0, Value(&mask));
  active.SetArg(1, Value(&mask));
  Check(ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "nonuniform integer execution mask was rejected");
  auto &float_value =
      values.AppendNewInst(ValueOpcode::BitCastU32F32, {Value::F32(1.f)});
  selection.SetArg(2, Value(&float_value));
  Check(!ValidateRuntimeValue(plan, Value(&active), RuntimeValueType::Integer),
        "floating-point inactive arm was accepted as integer-only");

  plan.srt_reads[0].value = Value(&first);
  Check(!ValidateRuntimeValue(plan, root, RuntimeValueType::Integer),
        "cyclic SRT read-first-lane dependency was accepted");
}

void TestUniformVectorDescriptorRead() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  auto &handle = block.AppendNewInst(ValueOpcode::GetBufferResource,
      {Value(0x1000u), Value(4u << 16u), Value(1u), Value(0x16204u)});
  program.memory_info.push_back({.kind = ResourceKind::Buffer});
  auto &count = block.AppendNewInst(ValueOpcode::LoadBufferU32,
      {Value(&handle), Value(0u), Value(0u), Value(0u), Value(true)});
  count.SetFlags(MemoryFlags{.index = 0});
  // The captured indirect kernel shares one read across sibling scalar lane reads,
  // enclosed by a different EXEC mask. Its resource plan must share that read too.
  auto &lane = block.AppendNewInst(ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)), Value(0u)});
  auto &active = block.AppendNewInst(ValueOpcode::ULessThan32, {Value(&lane), Value(32u)});
  auto &first = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(true)});
  auto &next = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)});
  auto &second = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&next), Value(true)});
  auto &sum = block.AppendNewInst(ValueOpcode::IAdd32, {Value(&first), Value(&second)});
  auto &small = block.AppendNewInst(ValueOpcode::ULessThanEqual32, {Value(&count), Value(72u)});
  auto &selected = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&small), Value(&sum), Value(&count)});
  auto &masked = block.AppendNewInst(ValueOpcode::SelectU32, {Value(&active), Value(&selected), Value(0u)});
  auto &records = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&masked), Value(&active)});
  DescriptorSource source;
  source.dword_count = 4;
  source.dwords = {Value(0x2000u), Value(4u << 16u), Value(&records), Value(0x16204u)};
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  Check(ValidateRuntimeValue(program, Value(&count)), "uniform DWORD count was rejected");
  struct Reads { uint32_t value = 72; uint32_t strict = 0; uint32_t ordinary = 0; bool clean = true; } reads;
  const SrtRuntime runtime{
      // Both pipeline readers synchronize with the GPU before reading, so the ordinary reader
      // sees the same current count the strict one does.
      .read_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.ordinary;
        if (address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = [](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1000u || words.size() != 1) return false;
        words[0] = reads.value;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  for (const bool written : {false, true}) {
    program.info.buffers[0].written = written;
    auto plan = ExtractResourcePlan(program);
    Check(plan.control_flow.empty(),
          "vector descriptor read depended on incidental control-flow capture");
    reads = {};
    Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
              snapshot.buffers[0].dwords[2] == 145 && reads.strict + reads.ordinary >= 1,
          "vector descriptor input was not read");
  }
  count.SetArg(4, Value(false));
  auto &inactive = block.AppendNewInst(ValueOpcode::ReadFirstLane, {Value(&count), Value(false)});
  reads.strict = 0;
  uint32_t result = 99;
  Check(SrtWalker(program, runtime).Evaluate(Value(&inactive), result) &&
            result == 0 && reads.strict == 0 && reads.ordinary == 0,
        "literal false EXEC read vector memory");
  count.SetArg(4, Value(true));
  // An invalid-format V# is host-evaluated like any other; only the in-shader decode zeroes it.
  count.SetArg(1, Value(&lane));
  Check(!ValidateRuntimeValue(program, Value(&count)),
        "varying vector address was treated as a uniform descriptor read");
}

void TestExactReciprocalDescriptorArithmetic() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  auto &block = AddValueBlock(program);
  for (const float divisor : {64.f, 128.f, 256.f, 512.f,
                              std::numeric_limits<float>::min(),
                              std::bit_cast<float>(253u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(SrtWalker(program, {}).Evaluate(Value(&reciprocal), result) &&
              result == std::bit_cast<uint32_t>(1.f / divisor),
          "power-of-two reciprocal was not exact");
  }
  for (const float divisor : {0.f, 3.f, -64.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::denorm_min(),
                              std::bit_cast<float>(254u << 23u)}) {
    auto &reciprocal = block.AppendNewInst(ValueOpcode::FPRecipIFlag32, {Value::F32(divisor)});
    uint32_t result = 0;
    Check(!SrtWalker(program, {}).Evaluate(Value(&reciprocal), result),
          "unsupported reciprocal rounding or exceptional input was accepted");
  }
}

void TestUnbasedFlatCacheHitMaterializes() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UnbasedFlatPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, {}, snapshot, specialization),
        "unbased FLAT stage materialization failed");
  Check(snapshot.buffers.empty() && snapshot.images.empty(),
        "unbased FLAT plan produced unexpected descriptors");
}

void TestWrittenDescriptorUsesStrictReaderOnce() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 1;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &offset = block.AppendNewInst(ValueOpcode::GetUserData,
                                     {Value(static_cast<ScalarReg>(0))});
  auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
      {Value(&handle), Value(&offset), Value(0u), Value(false)});
  read.SetFlags(MemoryFlags{.index = 0});
  DescriptorSource source;
  source.dwords = {Value(&read), Value(0u), Value(4u), Value(0u)};
  source.dword_count = 4;
  program.descriptor_sources.push_back(source);
  program.info.buffers.push_back({.source = 0, .written = true});
  // A host-evaluable branch captures resource reads and needs the writable
  // descriptor's clean provenance for the renderer's disjointness proof.
  auto &condition = block.AppendNewInst(ValueOpcode::IEqual32,
                                        {Value(&offset), Value(4u)});
  auto &store_block = AddValueBlock(program);
  AddValueBlock(program);
  program.block_info[0].condition = Value(&condition);
  program.block_info[0].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::ConditionalBranch;
  program.block_info[0].terminator.true_block = 1;
  program.block_info[0].terminator.false_block = 2;
  program.block_info[1].id = 1;
  program.block_info[1].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.block_info[2].id = 2;
  program.block_info[2].terminator.kind =
      Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind::Return;
  program.memory_info.push_back({.kind = ResourceKind::Buffer, .resource = 0});
  auto &output = store_block.AppendNewInst(ValueOpcode::GetBufferResource,
      {source.dwords[0], source.dwords[1], source.dwords[2], source.dwords[3]});
  store_block.AppendNewInst(ValueOpcode::StoreBufferU32,
      {Value(&output), Value(0u), Value(0u), Value(0u), Value(1u), Value(true)})
      .SetFlags(MemoryFlags{.index = 1});
  auto plan = ExtractResourcePlan(program);
  Check(plan.capture_specialization_reads,
        "conditional writable descriptor lost its alias proof");
  struct Reads { uint32_t ordinary = 0; uint32_t strict = 0; bool clean = false; } reads;
  const std::array<uint32_t, 1> user_data{4u};
  const SrtRuntime runtime{
      .user_data = user_data,
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t> words) {
        ++static_cast<Reads *>(data)->ordinary;
        words[0] = 0x8000u;
        return true;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address, std::span<uint32_t> words) {
        auto &reads = *static_cast<Reads *>(data);
        ++reads.strict;
        if (!reads.clean || address != 0x1004u) return false;
        words[0] = 0x8000u;
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, runtime, snapshot, specialization) &&
            reads.ordinary == 0 && reads.strict == 1,
        "GPU-dirty dynamic writable descriptor bypassed strict provenance");
  reads.clean = true;
  reads.strict = 0;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            reads.ordinary == 0 && reads.strict == 1 && snapshot.buffers[0].dwords[0] == 0x8000u &&
            snapshot.specialization_reads ==
                std::vector<std::pair<uint64_t, uint64_t>>{{0x1004u, 4u}},
        "writable descriptor was evaluated twice or scalar EXEC suppressed its read");
}

void TestFailedMaterializationRejectsStage() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto plan = UserDataBufferPlan();
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(!MaterializeResources(plan, {}, snapshot, specialization),
        "missing runtime user data did not reject the cached stage");
}

void TestFiniteImageRefreshReusesScalarReads() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
  auto &handle = block.AppendNewInst(ValueOpcode::GetAddressResource,
                                     {Value(0x1000u), Value(0u)});
  auto &srt = block.AppendNewInst(ValueOpcode::GetSrtResource);
  for (uint32_t index = 0; index < 3; ++index) {
    auto &read = block.AppendNewInst(ValueOpcode::LoadAddressU32,
        {Value(&handle), Value(index * 4u), Value(0u), Value(true)});
    read.SetFlags(MemoryFlags{.index = 0});
    program.srt_reads.push_back({Value(&read), index});
    auto &flat = block.AppendNewInst(ValueOpcode::ReadConst,
                                     {Value(&srt), Value(index)});
    DescriptorSource source;
    source.dword_count = 8;
    source.dwords.fill(Value(0u));
    source.dwords[0] = Value(&flat);
    source.dwords[1] = Value(static_cast<uint32_t>(
        Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u);
    source.dwords[3] = Value(Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u));
    program.descriptor_sources.push_back(source);
  }
  DescriptorSource root;
  root.dword_count = 8;
  root.dwords.fill(Value(0u));
  root.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{}).sources = {0, 1, 2, 1};
  program.descriptor_sources.push_back(root);
  program.info.images.push_back({
      .source = 3,
      .resource_class = ImageResourceClass::Sampled,
      .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
      .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D});
  auto plan = ExtractResourcePlan(program);
  struct Reads {
    std::array<uint32_t, 3> words{0x100u, 0x200u, 0x200u};
    std::array<uint32_t, 3> counts{};
    uint32_t ordinary = 0;
  } reads;
  const SrtRuntime runtime{
      .read_memory = +[](void *data, uint64_t, std::span<uint32_t>) {
        ++static_cast<Reads *>(data)->ordinary;
        return false;
      },
      .userdata = &reads,
      .read_specialization_memory = +[](void *data, uint64_t address,
                                        std::span<uint32_t> words) {
        if (words.size() != 1 || address < 0x1000u || address >= 0x100cu ||
            (address & 3u) != 0) return false;
        auto &reads = *static_cast<Reads *>(data);
        const auto index = (address - 0x1000u) / 4u;
        ++reads.counts[index];
        words[0] = reads.words[index];
        return true;
      }};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  const auto capacities = [&] {
    return std::array{snapshot.images.capacity(), snapshot.flattened_srt.capacity(),
                      snapshot.specialization_reads.capacity(), specialization.images.capacity()};
  };
  const auto check_mapping = [&](std::array<uint32_t, 4> ordinals) {
    // The specialization names a directory slot; the slot holds the mapping
    // offset.
    const auto offset =
        snapshot
            .flattened_srt[specialization.images[0].indirect_mapping_offset];
    Check(snapshot.images.size() == 2 && snapshot.flattened_srt[offset] == 4,
          "finite image candidates were not deduplicated");
    for (uint32_t key = 0; key < ordinals.size(); ++key) {
      Check(snapshot.flattened_srt[offset + 1u + key * 2u] == key &&
                snapshot.flattened_srt[offset + 2u + key * 2u] == ordinals[key],
            "finite image selector mapping is stale or incorrect");
    }
  };
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image materialization failed");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{1, 1, 1} &&
            snapshot.specialization_reads.size() == 3,
        "finite image candidates repeated scalar reads or bypassed clean provenance");
  check_mapping({0, 1, 1, 1});
  const auto warm_capacities = capacities();
  reads.words = {0x300u, 0x300u, 0x400u};
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "finite image refresh failed after descriptor changes");
  Check(reads.ordinary == 0 && reads.counts == std::array<uint32_t, 3>{2, 2, 2} &&
            snapshot.specialization_reads.size() == 3 &&
            snapshot.images[0].dwords[0] == 0x300u && snapshot.images[1].dwords[0] == 0x400u,
        "finite image refresh retained old scalar values or repeated reads");
  check_mapping({0, 0, 1, 0});
  Check(capacities() == warm_capacities,
        "finite image refresh grew reusable resource storage after warmup");
}

void TestMixedSamplerVariantsShareRuntimeDescriptor() {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  auto program = MixedSamplerProgram();
  auto plan = ExtractResourcePlan(program);
  std::array<uint32_t, 2> user_data{0x11111111u, 0x22222222u};
  const SrtRuntime runtime{.user_data = user_data};
  ResourceSnapshot snapshot;
  ResourceSpecialization specialization;
  Check(MaterializeResources(plan, runtime, snapshot, specialization),
        "mixed sampler materialization failed");
  ApplyResourceSpecialization(program, specialization);
  const auto &samplers = program.info.samplers;
  Check(snapshot.samplers.size() == 2 && samplers.size() == 3 &&
            samplers[0].snapshot_index == 0 && samplers[1].snapshot_index == 1 &&
            samplers[2].snapshot_index == 1 &&
            samplers[0].source == plan.info.samplers[0].source &&
            samplers[1].source == plan.info.samplers[1].source &&
            samplers[2].source == samplers[1].source &&
            !samplers[1].force_point_filtering && samplers[2].force_point_filtering &&
            program.info.sampled_pairs[2].sampler == 2,
        "native sampler variants lost their source identity or binding order");
  const auto capacity = snapshot.samplers.capacity();
  user_data[1] = 0x33333333u;
  Check(MaterializeResources(plan, runtime, snapshot, specialization) &&
            snapshot.samplers.size() == 2 && snapshot.samplers.capacity() == capacity &&
            snapshot.samplers[samplers[0].snapshot_index].dwords[0] == user_data[0] &&
            snapshot.samplers[samplers[1].snapshot_index].dwords[0] == user_data[1] &&
            snapshot.samplers[samplers[2].snapshot_index].dwords[0] == user_data[1],
        "sampler variants retained stale or duplicated descriptors after refresh");
}

namespace Bindless = Libs::Graphics::Bindless;
using Libs::Graphics::Prospero::BufferFormat;
using Libs::Graphics::Prospero::ImageType;
using Libs::Graphics::ShaderRecompiler::IR::BindlessShape;

Bindless::TSharp TestTSharp(uint32_t address, BufferFormat format, ImageType type,
                            uint32_t depth_field = 0) {
  Bindless::TSharp words{};
  words[0] = address;
  words[1] = static_cast<uint32_t>(format) << 20u;
  words[2] = 3u | (3u << 14u);
  words[3] = Libs::Graphics::DstSel(4, 5, 6, 7) | (static_cast<uint32_t>(type) << 28u);
  words[4] = depth_field;
  return words;
}

void TestBindlessRecordClassification() {
  using Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  const auto Shape = [](const Bindless::TSharp &words) {
    return Bindless::ClassifyRecord(words);
  };
  const auto plain = Shape(TestTSharp(0x100, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2D));
  Check(plain && plain->array == BindlessShape::Image2D &&
            plain->dimension == ImageDimension::Dim2D && !plain->cube,
        "a unorm 2D T# was not served from the 2D array");
  const auto array =
      Shape(TestTSharp(0x100, BufferFormat::k32Float, ImageType::kColor2DArray, 3u));
  Check(array && array->array == BindlessShape::Image2DArray && !array->cube,
        "a 2D array T# was not served from the 2D array array");
  const auto cube = Shape(TestTSharp(0x100, BufferFormat::kBc1UNorm, ImageType::kCube, 5u));
  Check(cube && cube->array == BindlessShape::Image2DArray && cube->cube &&
            cube->Code() == Libs::Graphics::ShaderRecompiler::IR::IndirectImageShape(
                                ImageDimension::Dim2DArray, true),
        "a cube T# was not served as a 2D array with cube coordinates");
  const auto volume =
      Shape(TestTSharp(0x100, BufferFormat::k16_16_16_16Float, ImageType::kColor3D));
  Check(volume && volume->array == BindlessShape::Image3D &&
            volume->dimension == ImageDimension::Dim3D,
        "a volume T# was not served from the 3D array");
  Check(!Shape(Bindless::TSharp{}), "the null T# was served");
  Check(!Shape(TestTSharp(0x100, BufferFormat::k32UInt, ImageType::kColor2D)),
        "an integer T# was served from a float array");
  Check(!Shape(TestTSharp(0x100, BufferFormat::k8_8_8_8UNorm, ImageType::kColor1D)),
        "a 1D T# was served though no array holds 1D views");
  Check(!Shape(TestTSharp(0x100, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2DMsaa)),
        "an MSAA T# was served");
  auto reserved = TestTSharp(0x100, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2D);
  reserved[1] |= 0x20000000u;
  Check(!Shape(reserved), "a T# with reserved bits set was served");
  Check(!Shape(TestTSharp(0x100, BufferFormat::k32Float, ImageType::kColor2DArray,
                          (5u << 16u) | 3u)),
        "a 2D array T# whose base layer passes its last layer was served");
  auto mips = TestTSharp(0x100, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2D);
  mips[3] |= 2u << 16u;
  mips[5] = 2u << 4u;
  Check(Shape(mips).has_value(), "a T# with a sane mip range was refused");
  auto past_max = mips;
  past_max[5] = 1u << 4u;
  Check(!Shape(past_max), "a T# whose last level passes its mip count was served");
  auto inverted = mips;
  inverted[3] |= 3u << 12u;
  Check(!Shape(inverted), "a T# whose base level passes its last level was served");
  auto clamped = mips;
  clamped[1] |= (2u * 256u + 1u) << 8u;
  Check(!Shape(clamped), "a T# whose minimum LOD passes its last level was served");
}

void TestBindlessRecordCount() {
  Check(Bindless::HeapRecordCount(48u * 8u, 48u, 16u, 1u << 20u) == 8u,
        "eight 48-byte records were not counted");
  Check(Bindless::HeapRecordCount(48u * 7u + 16u + 32u, 48u, 16u, 1u << 20u) == 8u,
        "a last record whose descriptor ends at the heap end was dropped");
  Check(Bindless::HeapRecordCount(48u * 7u + 16u + 31u, 48u, 16u, 1u << 20u) == 7u,
        "a last record whose descriptor crosses the heap end was counted");
  Check(Bindless::HeapRecordCount(48u * 32768u, 48u, 16u, 1000u) == 1000u,
        "the record count ignored its ceiling");
  Check(Bindless::HeapRecordCount(16u, 48u, 16u, 1u << 20u) == 0u,
        "a heap too small for one descriptor had records");
  Check(Bindless::HeapRecordCount(1024u, 48u, 18u, 1u << 20u) == 0u &&
            Bindless::HeapRecordCount(1024u, 24u, 0u, 1u << 20u) == 0u &&
            Bindless::HeapRecordCount(1024u, 48u, 20u, 1u << 20u) == 0u,
        "a misaligned or overlapping record layout was accepted");
}

void TestBindlessSlotAllocation() {
  Bindless::SlotAllocator slots;
  const auto a = slots.Assign(BindlessShape::Image2D, 0xa);
  const auto b = slots.Assign(BindlessShape::Image2D, 0xb);
  const auto again = slots.Assign(BindlessShape::Image2D, 0xa);
  const auto volume = slots.Assign(BindlessShape::Image3D, 0xa);
  Check(a == 0u && b == 1u && again == 0u && volume == 0u &&
            slots.Used(BindlessShape::Image2D) == 2u &&
            slots.Used(BindlessShape::Image2DArray) == 0u &&
            slots.Used(BindlessShape::Image3D) == 1u,
        "elements were not handed out once per view and per array");
  for (uint32_t key = 2; key < Libs::Graphics::ShaderRecompiler::IR::BindlessImageSlots; key++) {
    Check(slots.Assign(BindlessShape::Image2D, 0x100 + key).has_value(),
          "an array refused an element below its length");
  }
  Check(!slots.Assign(BindlessShape::Image2D, 0xdead).has_value() &&
            slots.Assign(BindlessShape::Image2D, 0xa) == 0u,
        "a full array handed out an element past its length");
  slots.Reset();
  Check(slots.Assign(BindlessShape::Image2D, 0xdead) == 0u &&
            slots.Used(BindlessShape::Image3D) == 0u,
        "a reset left elements assigned");
}

void TestBindlessHeapTranslation() {
  constexpr uint32_t kStride = 48u;
  constexpr uint32_t kOffset = 16u;
  constexpr uint32_t kRecords = 12u;
  std::vector<uint32_t> heap(kRecords * kStride / 4u, 0xcdcdcdcdu);
  const auto Put = [&](uint32_t record, const Bindless::TSharp &tsharp) {
    std::copy(tsharp.begin(), tsharp.end(), heap.begin() + (record * kStride + kOffset) / 4u);
  };
  const auto a = TestTSharp(0x1000, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2D);
  const auto b = TestTSharp(0x2000, BufferFormat::k8_8_8_8Srgb, ImageType::kColor2D);
  const auto volume = TestTSharp(0x3000, BufferFormat::k32Float, ImageType::kColor3D);
  const auto cube = TestTSharp(0x4000, BufferFormat::k8_8_8_8UNorm, ImageType::kCube, 5u);
  const auto integer = TestTSharp(0x5000, BufferFormat::k32UInt, ImageType::kColor2D);
  const auto refused = TestTSharp(0x6000, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2D);
  const auto later = TestTSharp(0x7000, BufferFormat::k8_8_8_8UNorm, ImageType::kColor2D);
  Put(0, Bindless::TSharp{});
  Put(1, a);
  Put(2, b);
  Put(3, a);
  Put(4, volume);
  Put(5, cube);
  Put(6, integer);
  Put(7, refused);
  Put(8, b);
  Put(9, later);
  Put(10, Bindless::TSharp{});
  Put(11, volume);

  Bindless::TranslationCache cache;
  Bindless::SlotAllocator slots;
  std::map<uint32_t, uint32_t> resolved;
  bool later_ready = false;
  const Bindless::RecordResolver resolve =
      [&](const Bindless::TSharp &tsharp,
          const Bindless::RecordShape &shape) -> std::optional<uint32_t> {
    resolved[tsharp[0]]++;
    if (tsharp == refused) {
      return 0u;
    }
    if (tsharp == later && !later_ready) {
      return std::nullopt;
    }
    return Bindless::TranslationWord(shape, *slots.Assign(shape.array, tsharp[0]));
  };
  const auto missing = Bindless::MissingRecords(heap, kStride, kOffset, kRecords, cache);
  Check(missing.size() == 6u,
        "the missing records were not the distinct servable descriptors of the heap");
  std::vector<uint32_t> words(kRecords, 0xffffffffu);
  Bindless::TranslateHeap(heap, kStride, kOffset, kRecords, cache, resolve, words);
  using Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;
  using Libs::Graphics::ShaderRecompiler::IR::IndirectImageShape;
  using Libs::Graphics::ShaderRecompiler::IR::IndirectImageSlot;
  const auto Word = [](ImageDimension dimension, bool is_cube, uint32_t element) {
    return IndirectImageSlot(IndirectImageShape(dimension, is_cube), element);
  };
  Check(words[0] == 0u && words[10] == 0u, "a null record named an element");
  Check(words[1] == Word(ImageDimension::Dim2D, false, 0) && words[3] == words[1],
        "two records naming one descriptor did not share its element");
  Check(words[2] == Word(ImageDimension::Dim2D, false, 1) && words[8] == words[2],
        "a second descriptor did not take the next element");
  Check(words[4] == Word(ImageDimension::Dim3D, false, 0) && words[11] == words[4],
        "a volume record did not take its own array's element");
  Check(words[5] == Word(ImageDimension::Dim2DArray, true, 0),
        "a cube record did not take a 2D array element with the cube shape");
  Check(words[6] == 0u && words[7] == 0u && words[9] == 0u,
        "an unservable, refused or unsettled record named an element");
  Check(resolved[0x1000] == 1u && resolved[0x2000] == 1u && resolved[0x3000] == 1u &&
            !resolved.contains(0x5000),
        "a descriptor was resolved more than once, or an unservable one at all");

  later_ready = true;
  Check(Bindless::MissingRecords(heap, kStride, kOffset, kRecords, cache).size() == 1u,
        "only the unsettled record should still be missing");
  Bindless::TranslateHeap(heap, kStride, kOffset, kRecords, cache, resolve, words);
  Check(words[9] == Word(ImageDimension::Dim2D, false, 2) && resolved[0x7000] == 2u &&
            resolved[0x1000] == 1u && resolved[0x6000] == 1u,
        "the second pass re-resolved cached records or missed the unsettled one");
}

void TestShaderReadCache() {
  using Libs::Graphics::ShaderGuestReadCache;
  constexpr uint64_t base = 0x10000;
  std::vector<uint32_t> memory(1024);
  for (uint32_t i = 0; i < memory.size(); i++) {
    memory[i] = 0x1000u + i;
  }
  uint64_t dirty_begin = UINT64_MAX;
  uint64_t dirty_end = UINT64_MAX;
  bool drain = false;
  uint32_t line_reads = 0;
  uint32_t word_reads = 0;
  const auto inside = [&](uint64_t address, uint64_t size) {
    return address >= base && address + size <= base + memory.size() * 4u;
  };
  const auto clean = [&](uint64_t address, uint64_t size) {
    return inside(address, size) &&
           (address + size <= dirty_begin || address >= dirty_end);
  };
  const auto copy = [&](uint64_t address, void *data, uint64_t size) {
    std::memcpy(data,
                reinterpret_cast<const uint8_t *>(memory.data()) +
                    (address - base),
                size);
  };
  const auto clean_line = [&](uint64_t address, void *data, uint64_t size) {
    line_reads++;
    if (!clean(address, size)) {
      return false;
    }
    copy(address, data, size);
    return true;
  };
  const auto reader = [&](uint64_t address, std::span<uint32_t> words,
                          bool &drained) {
    word_reads++;
    if (words.empty() || !inside(address, words.size_bytes())) {
      return false;
    }
    drained = drain && !clean(address, words.size_bytes());
    copy(address, words.data(), words.size_bytes());
    return true;
  };
  ShaderGuestReadCache cache;
  cache.Reset();
  uint32_t word = 0;
  const auto read = [&](uint64_t address) {
    return cache.Read(address, {&word, 1}, clean_line, reader);
  };

  bool served = true;
  for (uint32_t i = 0; i < 64; i++) {
    served = served && read(base + i * 4u) && word == 0x1000u + i;
  }
  Check(served && line_reads == 1 && word_reads == 0,
        "a clean line was not read once for every word in it");
  std::array<uint32_t, 8> descriptor{};
  Check(cache.Read(base + 256 + 32, descriptor, clean_line, reader) &&
            descriptor[0] == 0x1000u + 72 && descriptor[7] == 0x1000u + 79 &&
            line_reads == 2 && word_reads == 0,
        "a descriptor inside one line was not served from it");
  std::array<uint32_t, 2> straddle{};
  Check(cache.Read(base + 252, straddle, clean_line, reader) &&
            straddle[0] == 0x1000u + 63 && straddle[1] == 0x1000u + 64 &&
            word_reads == 1 && line_reads == 2,
        "a read across two lines did not go to the reader");

  memory[0] = 0xdeadu;
  Check(read(base) && word == 0x1000u && line_reads == 2,
        "a walk did not see the line it had already read");
  cache.Reset();
  Check(read(base) && word == 0xdeadu && line_reads == 3,
        "a reset cache served a line read before it");

  dirty_begin = base + 512 + 8;
  dirty_end = dirty_begin + 4;
  const auto words_before = word_reads;
  served = true;
  for (uint32_t i = 0; i < 4; i++) {
    served = served && read(base + 512 + i * 4u) && word == 0x1000u + 128 + i;
  }
  Check(served && word_reads == words_before + 4 && line_reads == 4,
        "a line with one dirty word was not read word by word, or was "
        "re-checked per word");

  drain = true;
  Check(read(base + 512 + 8) && read(base) && line_reads == 5,
        "the cache kept its lines across a reader that may have drained");
  Check(!read(base - 4) && !cache.Read(base, {}, clean_line, reader),
        "the cache accepted a read the reader refuses");
}

} // namespace

namespace Common {

int DbgExitHandler(const char *, int, std::string_view) { std::abort(); }

int DbgExitHandler(const char *, int, fmt::text_style, std::string_view) {
  std::abort();
}

int DbgExitIfHandler(const char *, const char *, int) { return 1; }

void DbgExit(int) { std::abort(); }

} // namespace Common

int main() {
  TestMappedSrtUsesDirectReaderByDefault();
  TestIntegerRuntimeValueFollowsSrtReads();
  TestUniformVectorDescriptorRead();
  TestExactReciprocalDescriptorArithmetic();
  TestUnbasedFlatCacheHitMaterializes();
  TestWrittenDescriptorUsesStrictReaderOnce();
  TestFailedMaterializationRejectsStage();
  TestFiniteImageRefreshReusesScalarReads();
  TestMixedSamplerVariantsShareRuntimeDescriptor();
  TestBindlessRecordClassification();
  TestBindlessRecordCount();
  TestBindlessSlotAllocation();
  TestBindlessHeapTranslation();
  TestShaderReadCache();
  std::puts("ResourceMaterializationTests: all cases passed");
  return 0;
}

// Keep this focused standalone target self-contained by amalgamating its small
// typed-IR implementation set.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
