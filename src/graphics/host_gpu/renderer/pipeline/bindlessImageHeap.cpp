#include "graphics/host_gpu/renderer/pipeline/bindlessImageHeap.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <mutex>
#include <xxhash.h>

namespace Libs::Graphics {

namespace IR = ShaderRecompiler::IR;
using IR::BindlessShape;

namespace {

struct SharedLayouts {
	std::mutex              mutex;
	vk::DescriptorSetLayout set   = nullptr;
	vk::DescriptorSetLayout empty = nullptr;
};

SharedLayouts& Layouts() {
	static SharedLayouts layouts;
	return layouts;
}

constexpr uint32_t PoolSets = 8;

constexpr vk::PipelineStageFlags2 SampleStages =
    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;

[[nodiscard]] std::optional<BindlessShape> ArrayOfWord(uint32_t word) {
	const auto code = word >> 16u;
	return IR::BindlessShapeFor(static_cast<ShaderRecompiler::Decoder::ImageDimension>(code >> 1u));
}

} // namespace

size_t BindlessImageHeap::HeapKeyHash::operator()(const HeapKey& key) const noexcept {
	const std::array<uint64_t, 3> words {key.base, key.size,
	                                     (static_cast<uint64_t>(key.stride) << 32u) | key.offset};
	return static_cast<size_t>(XXH3_64bits(words.data(), sizeof(words)));
}

std::array<vk::DescriptorSetLayoutBinding, 4>
BindlessImageHeap::LayoutBindings(vk::ShaderStageFlags stages) {
	std::array<vk::DescriptorSetLayoutBinding, 4> bindings {};
	bindings[0] = {IR::BindlessArenaBinding, vk::DescriptorType::eStorageBuffer, 1, stages,
	               nullptr};
	for (uint32_t shape = 0; shape < Arrays; shape++) {
		bindings[1u + shape] = {IR::BindlessImageBinding(static_cast<BindlessShape>(shape)),
		                        vk::DescriptorType::eSampledImage, IR::BindlessImageSlots, stages,
		                        nullptr};
	}
	return bindings;
}

vk::DescriptorSetLayout BindlessImageHeap::SetLayout(GraphicContext& graphics) {
	auto&            layouts = Layouts();
	std::scoped_lock lock(layouts.mutex);
	if (layouts.set == nullptr) {
		const auto                        bindings = LayoutBindings(vk::ShaderStageFlagBits::eAll);
		vk::DescriptorSetLayoutCreateInfo create {};
		create.bindingCount = static_cast<uint32_t>(bindings.size());
		create.pBindings    = bindings.data();
		EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &layouts.set) !=
		        vk::Result::eSuccess);
	}
	return layouts.set;
}

vk::DescriptorSetLayout BindlessImageHeap::EmptySetLayout(GraphicContext& graphics) {
	auto&            layouts = Layouts();
	std::scoped_lock lock(layouts.mutex);
	if (layouts.empty == nullptr) {
		vk::DescriptorSetLayoutCreateInfo create {};
		EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &layouts.empty) !=
		        vk::Result::eSuccess);
	}
	return layouts.empty;
}

BindlessImageHeap::BindlessImageHeap(RenderContext& context, RenderExecutor& executor)
    : m_context(context), m_executor(executor) {
	for (size_t array = 0; array < Arrays; array++) {
		m_elements[array].resize(IR::BindlessImageSlots);
		m_infos[array].resize(IR::BindlessImageSlots);
	}
}

BindlessImageHeap::~BindlessImageHeap() {
	auto& device = m_context.GetGraphics().device;
	for (const auto pool: m_pools) {
		device.destroyDescriptorPool(pool, nullptr);
	}
}

BindlessImageHeap::HeapKey
BindlessImageHeap::KeyOf(const ShaderRecompiler::IR::BindlessImageTable& table) {
	ShaderBufferResource heap {};
	std::copy(table.heap.begin(), table.heap.end(), heap.fields);
	HeapKey key;
	key.base = heap.Base48();
	key.size =
	    key.base != 0 ? Libs::LibKernel::Memory::ClampRangeSize(key.base, heap.GetSize()) : 0u;
	key.stride = table.stride;
	key.offset = table.record_offset;
	return key;
}

void BindlessImageHeap::EnsureNullImages() {
	if (m_null_ready) {
		return;
	}
	auto& cache = m_context.GetTextureCache();
	for (uint32_t shape = 0; shape < Arrays; shape++) {
		const auto              kind = static_cast<BindlessShape>(shape);
		TextureCache::ImageDesc desc {};
		desc.info.guest_format    = Prospero::BufferFormat::k32Float;
		desc.info.pixel_format    = vk::Format::eR32Sfloat;
		desc.info.type            = kind == BindlessShape::Image3D ? Prospero::ImageType::kColor3D
		                                                           : Prospero::ImageType::kColor2D;
		desc.info.extent          = {1, 1, 1};
		desc.info.resources       = {1, 1};
		desc.info.bytes_per_block = 4;
		desc.info.samples         = 1;
		desc.info.mip_layout[0]   = {0, 0, 1, 1};
		desc.view_info.format     = desc.info.pixel_format;
		desc.view_info.type   = kind == BindlessShape::Image3D        ? vk::ImageViewType::e3D
		                        : kind == BindlessShape::Image2DArray ? vk::ImageViewType::e2DArray
		                                                              : vk::ImageViewType::e2D;
		desc.view_info.aspect = vk::ImageAspectFlagBits::eColor;
		desc.view_info.usage  = vk::ImageUsageFlagBits::eSampled;
		desc.type             = TextureCache::BindingType::Texture;
		const auto id         = cache.FindImage(desc);
		const auto view       = cache.FindTexture(id, desc);
		auto&      null       = m_null[shape];
		null.id               = id;
		null.view             = view;
		null.desc             = desc;
		null.range            = {0, 1, 0, 1};
		null.layout           = vk::ImageLayout::eGeneral;
		null.live             = true;
		for (uint32_t slot = 0; slot < IR::BindlessImageSlots; slot++) {
			if (!m_elements[shape][slot].live) {
				m_infos[shape][slot] = {nullptr, view, vk::ImageLayout::eGeneral};
			}
		}
	}
	m_null_ready = true;
	m_set_dirty  = true;
}

bool BindlessImageHeap::ElementAlive(const Element& element) {
	const auto* image = m_context.GetTextureCache().m_slot_images.try_get(element.id);
	return image != nullptr && image->registered && !image->dormant &&
	       !image->binding.needs_rebind && !image->depth_id && image->backing.image != nullptr;
}

void BindlessImageHeap::Kill(BindlessShape array, uint32_t slot) {
	const auto index   = static_cast<size_t>(array);
	auto&      element = m_elements[index][slot];
	if (!element.live) {
		return;
	}
	element.live         = false;
	element.view         = nullptr;
	m_infos[index][slot] = {nullptr, m_null[index].view, vk::ImageLayout::eGeneral};
	m_set_dirty          = true;
	std::erase_if(m_cache, [&](const auto& entry) {
		return entry.second != 0u && (entry.second & 0xffffu) == slot &&
		       ArrayOfWord(entry.second) == array;
	});
	for (auto& [key, heap]: m_heaps) {
		heap.stale = true;
	}
}

void BindlessImageHeap::SweepDeadElements() {
	for (uint32_t array = 0; array < Arrays; array++) {
		const auto kind = static_cast<BindlessShape>(array);
		const auto used = m_slots.Used(kind);
		for (uint32_t slot = 0; slot < used; slot++) {
			const auto& element = m_elements[array][slot];
			if (element.live && !ElementAlive(element)) {
				Kill(kind, slot);
			}
		}
	}
}

void BindlessImageHeap::TouchLiveElements() {
	auto& cache = m_context.GetTextureCache();
	if (cache.m_gc_tick == m_touched_gc_tick) {
		return;
	}
	m_touched_gc_tick = cache.m_gc_tick;
	for (uint32_t array = 0; array < Arrays; array++) {
		const auto used = m_slots.Used(static_cast<BindlessShape>(array));
		for (uint32_t slot = 0; slot < used; slot++) {
			const auto& element = m_elements[array][slot];
			if (element.live) {
				cache.TouchImage(cache.m_slot_images[element.id]);
			}
		}
	}
}

void BindlessImageHeap::EvictIdleHeaps(uint64_t frame) {
	if (frame == m_evicted_frame) {
		return;
	}
	m_evicted_frame    = frame;
	const auto evicted = std::erase_if(m_heaps, [&](const auto& entry) {
		return frame - entry.second.used_frame > HeapIdleFrames;
	});
	if (evicted != 0) {
		m_arena_dirty = true;
	}
}

void BindlessImageHeap::RefreshLiveElements() {
	auto& cache = m_context.GetTextureCache();
	for (uint32_t array = 0; array < Arrays; array++) {
		const auto kind = static_cast<BindlessShape>(array);
		const auto used = m_slots.Used(kind);
		for (uint32_t slot = 0; slot < used; slot++) {
			auto& element = m_elements[array][slot];
			if (!element.live) {
				continue;
			}
			if (!ElementAlive(element)) {
				Kill(kind, slot);
				continue;
			}
			auto& image         = cache.GetImage(element.id);
			image.usage.texture = true;
			if (!image.IsCpuDirty() && !image.IsBufferModified()) {
				continue;
			}
			const auto view = cache.FindTexture(element.id, element.desc);
			if (view != element.view && view != nullptr) {
				element.view                   = view;
				m_infos[array][slot].imageView = view;
				m_set_dirty                    = true;
			}
		}
	}
}

bool BindlessImageHeap::ReadHeap(const HeapKey& key, uint32_t records,
                                 std::vector<uint32_t>& dwords) const {
	if (records == 0) {
		dwords.clear();
		return false;
	}
	const auto bytes =
	    static_cast<uint64_t>(records - 1u) * key.stride + key.offset + sizeof(Bindless::TSharp);
	dwords.resize(static_cast<size_t>((bytes + 3u) / 4u));
	if (!Libs::LibKernel::Memory::TryReadBufferBacking(key.base, dwords.data(), bytes)) {
		(void)Libs::LibKernel::Memory::ReadBackingPartial(key.base, dwords.data(), bytes);
	}
	return true;
}

bool BindlessImageHeap::Rescan(const HeapKey& key, Heap& heap,
                               const std::vector<uint32_t>& dwords) {
	auto&      cache = m_context.GetTextureCache();
	const auto missing =
	    Bindless::MissingRecords(dwords, key.stride, key.offset, heap.records, m_cache);
	struct Pending {
		Bindless::TSharp      tsharp;
		Bindless::RecordShape shape;
		TextureBinding        binding;
	};
	const auto Resolve = [&](const Bindless::TSharp& tsharp, const Bindless::RecordShape& shape) {
		IR::ImageResource resource;
		resource.resource_class = IR::ImageResourceClass::Sampled;
		resource.numeric_class  = Prospero::TextureNumericClass::Float;
		resource.dimension      = shape.dimension;
		resource.cube           = shape.cube;
		resource.read           = true;
		IR::DescriptorValue value;
		value.dword_count = 8u;
		value.dwords      = tsharp;
		return m_executor.ResolveTexture(resource, value);
	};
	const auto Usable = [&](const TextureBinding& binding) {
		const auto* image = cache.m_slot_images.try_get(binding.image_id);
		return image != nullptr && !binding.desc.info.data.Empty() && image->registered &&
		       !image->dormant && !image->binding.needs_rebind && !image->depth_id &&
		       image->backing.image != nullptr;
	};
	std::vector<Pending> pending;
	pending.reserve(missing.size());
	for (const auto& tsharp: missing) {
		const auto shape   = Bindless::ClassifyRecord(tsharp);
		auto       binding = Resolve(tsharp, *shape);
		if (binding.desc.info.data.Empty()) {
			m_cache[tsharp] = 0u;
			continue;
		}
		pending.push_back({tsharp, *shape, std::move(binding)});
	}
	for (uint32_t pass = 0; pass < 4u; pass++) {
		bool churn = false;
		for (auto& entry: pending) {
			if (!Usable(entry.binding)) {
				entry.binding = Resolve(entry.tsharp, entry.shape);
				churn         = true;
			}
		}
		if (!churn) {
			break;
		}
	}
	for (auto& entry: pending) {
		if (!Usable(entry.binding)) {
			continue;
		}
		const auto view = cache.FindTexture(entry.binding.image_id, entry.binding.desc);
		if (view == nullptr) {
			continue;
		}
		const auto slot = m_slots.Assign(
		    entry.shape.array, reinterpret_cast<uint64_t>(static_cast<VkImageView>(view)));
		if (!slot.has_value()) {
			return false;
		}
		const auto array   = static_cast<size_t>(entry.shape.array);
		auto&      element = m_elements[array][*slot];
		if (!element.live || element.view != view) {
			auto&       image = cache.m_slot_images[entry.binding.image_id];
			const auto& info  = entry.binding.desc.view_info;
			element.id        = entry.binding.image_id;
			element.view      = view;
			element.desc      = entry.binding.desc;
			element.range  = {info.base_level, info.level_count, info.base_layer, info.layer_count};
			element.layout = image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
			                                      : vk::ImageLayout::eShaderReadOnlyOptimal;
			element.live   = true;
			image.usage.texture   = true;
			m_infos[array][*slot] = {nullptr, view, element.layout};
			m_set_dirty           = true;
		}
		m_cache[entry.tsharp] = Bindless::TranslationWord(entry.shape, *slot);
	}
	heap.words.assign(heap.records, 0u);
	Bindless::TranslateHeap(
	    dwords, key.stride, key.offset, heap.records, m_cache,
	    [](const Bindless::TSharp&, const Bindless::RecordShape&) -> std::optional<uint32_t> {
		    return std::nullopt;
	    },
	    heap.words);
	heap.scanned  = true;
	heap.stale    = false;
	m_arena_dirty = true;
	return true;
}

void BindlessImageHeap::ResetElements() {
	m_slots.Reset();
	m_cache.clear();
	for (uint32_t array = 0; array < Arrays; array++) {
		for (uint32_t slot = 0; slot < IR::BindlessImageSlots; slot++) {
			m_elements[array][slot].live = false;
			m_elements[array][slot].view = nullptr;
			m_infos[array][slot]         = {nullptr, m_null[array].view, vk::ImageLayout::eGeneral};
		}
	}
	for (auto& [key, heap]: m_heaps) {
		heap.stale = true;
	}
	m_set_dirty = true;
}

void BindlessImageHeap::Prepare(const ShaderRecompiler::IR::BindlessImageTable& table) {
	EnsureNullImages();
	SweepDeadElements();
	const auto frame = m_context.GetTextureCache().FrameIndex();
	EvictIdleHeaps(frame);
	const auto key         = KeyOf(table);
	auto [found, inserted] = m_heaps.try_emplace(key);
	auto& heap             = found->second;
	heap.used_frame        = frame;
	if (inserted) {
		heap.records = Bindless::HeapRecordCount(key.size, key.stride, key.offset, MaxHeapRecords);
	}
	if (heap.checked_frame != frame || heap.stale || !heap.scanned) {
		if (heap.rescan_frame != frame) {
			heap.rescan_frame = frame;
			heap.rescans      = 0;
		}
		static thread_local std::vector<uint32_t> dwords;
		if (ReadHeap(key, heap.records, dwords)) {
			const auto hash  = XXH3_64bits(dwords.data(), dwords.size() * sizeof(uint32_t));
			const bool moved = !heap.scanned || hash != heap.hash;
			if ((moved || heap.stale) && (heap.rescans < MaxRescansPerFrame || !heap.scanned)) {
				heap.rescans++;
				if (!Rescan(key, heap, dwords)) {
					ResetElements();
					for (auto& [other_key, other]: m_heaps) {
						static thread_local std::vector<uint32_t> other_dwords;
						if (ReadHeap(other_key, other.records, other_dwords) &&
						    !Rescan(other_key, other, other_dwords)) {
							other.words.assign(other.records, 0u);
						}
					}
				}
				heap.hash = hash;
			}
		} else if (!heap.scanned) {
			heap.words.assign(heap.records, 0u);
			heap.scanned  = true;
			heap.stale    = false;
			m_arena_dirty = true;
		}
		heap.checked_frame = frame;
	}
	if (m_refreshed_frame != frame) {
		RefreshLiveElements();
		m_refreshed_frame = frame;
	}
	if (m_arena_dirty) {
		BuildArena();
	}
}

BindlessImageHeap::Region
BindlessImageHeap::Lookup(const ShaderRecompiler::IR::BindlessImageTable& table) const {
	const auto found = m_heaps.find(KeyOf(table));
	if (found == m_heaps.end() || !found->second.scanned || m_arena_dirty) {
		return {};
	}
	return {found->second.arena_base, found->second.records};
}

void BindlessImageHeap::BuildArena() {
	// Word 0 is the null translation, so a record past every region still reads as no image.
	uint64_t words = 1;
	for (auto& [key, heap]: m_heaps) {
		heap.arena_base = static_cast<uint32_t>(words);
		words += heap.records;
	}
	EXIT_IF(words > UINT32_MAX);
	auto& scheduler = m_context.GetCommandScheduler();
	auto  arena =
	    std::make_shared<Buffer>(m_context.GetGraphics(), scheduler, MemoryUsage::Stream, 0,
	                             vk::BufferUsageFlagBits::eStorageBuffer, words * sizeof(uint32_t));
	auto mapped = arena->Mapped();
	EXIT_IF(mapped.size() < words * sizeof(uint32_t));
	auto* out = reinterpret_cast<uint32_t*>(mapped.data());
	out[0]    = 0u;
	for (const auto& [key, heap]: m_heaps) {
		auto* region = out + heap.arena_base;
		std::fill_n(region, heap.records, 0u);
		std::copy_n(heap.words.begin(), std::min<size_t>(heap.words.size(), heap.records), region);
	}
	arena->Flush(0, words * sizeof(uint32_t));
	m_arena       = std::move(arena);
	m_arena_words = static_cast<uint32_t>(words);
	m_arena_dirty = false;
	m_set_dirty   = true;
}

vk::DescriptorSet BindlessImageHeap::AllocateSet(vk::DescriptorPool& pool) {
	auto&                         device = m_context.GetGraphics().device;
	const auto                    layout = SetLayout(m_context.GetGraphics());
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &layout;
	vk::DescriptorSet set       = nullptr;
	for (const auto candidate: m_pools) {
		allocate.descriptorPool = candidate;
		if (device.allocateDescriptorSets(&allocate, &set) == vk::Result::eSuccess) {
			pool = candidate;
			return set;
		}
	}
	const std::array sizes {
	    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage,
	                            PoolSets * static_cast<uint32_t>(Arrays) * IR::BindlessImageSlots},
	    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, PoolSets},
	};
	vk::DescriptorPoolCreateInfo create {};
	create.flags             = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
	create.maxSets           = PoolSets;
	create.poolSizeCount     = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes        = sizes.data();
	vk::DescriptorPool fresh = nullptr;
	EXIT_IF(device.createDescriptorPool(&create, nullptr, &fresh) != vk::Result::eSuccess);
	m_pools.push_back(fresh);
	allocate.descriptorPool = fresh;
	EXIT_IF(device.allocateDescriptorSets(&allocate, &set) != vk::Result::eSuccess);
	pool = fresh;
	return set;
}

void BindlessImageHeap::RetireVersions() {
	auto& scheduler = m_context.GetCommandScheduler();
	auto& device    = m_context.GetGraphics().device;
	std::erase_if(m_retired, [&](Version& version) {
		if (!scheduler.IsFree(version.last_used)) {
			return false;
		}
		(void)device.freeDescriptorSets(version.pool, 1, &version.set);
		return true;
	});
}

void BindlessImageHeap::BuildVersion() {
	if (m_current.set != nullptr) {
		m_retired.push_back(std::move(m_current));
		m_current = {};
	}
	RetireVersions();
	Version version;
	version.set   = AllocateSet(version.pool);
	version.arena = m_arena;
	const vk::DescriptorBufferInfo arena_info {m_arena->Handle(), 0,
	                                           static_cast<uint64_t>(m_arena_words) * 4u};
	std::array<vk::WriteDescriptorSet, 1 + Arrays> writes {};
	writes[0].dstSet          = version.set;
	writes[0].dstBinding      = IR::BindlessArenaBinding;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType  = vk::DescriptorType::eStorageBuffer;
	writes[0].pBufferInfo     = &arena_info;
	for (uint32_t array = 0; array < Arrays; array++) {
		auto& write           = writes[1u + array];
		write.dstSet          = version.set;
		write.dstBinding      = IR::BindlessImageBinding(static_cast<BindlessShape>(array));
		write.descriptorCount = IR::BindlessImageSlots;
		write.descriptorType  = vk::DescriptorType::eSampledImage;
		write.pImageInfo      = m_infos[array].data();
	}
	m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
	                                                    writes.data(), 0, nullptr);
	m_current   = std::move(version);
	m_set_dirty = false;
}

vk::DescriptorSet BindlessImageHeap::Commit(vk::CommandBuffer command) {
	EnsureNullImages();
	if (m_arena == nullptr || m_arena_dirty) {
		BuildArena();
	}
	SweepDeadElements();
	TouchLiveElements();
	if (m_set_dirty || m_current.set == nullptr) {
		BuildVersion();
	}
	auto& cache = m_context.GetTextureCache();
	m_barriers.clear();
	const auto Bring = [&](Image& image, vk::ImageLayout layout,
	                       const ImageSubresourceRange& range) {
		const auto& state = image.backing.state;
		if (image.backing.subresource_states.empty() && state.layout == layout &&
		    state.access_mask == vk::AccessFlagBits2::eShaderRead) {
			return;
		}
		const auto barriers =
		    image.GetBarriers(layout, vk::AccessFlagBits2::eShaderRead, SampleStages, range);
		m_barriers.insert(m_barriers.end(), barriers.begin(), barriers.end());
	};
	for (uint32_t array = 0; array < Arrays; array++) {
		const auto used = m_slots.Used(static_cast<BindlessShape>(array));
		for (uint32_t slot = 0; slot < used; slot++) {
			const auto& element = m_elements[array][slot];
			if (!element.live) {
				continue;
			}
			auto& image = cache.m_slot_images[element.id];
			if (image.binding.is_target ||
			    (image.binding.is_bound &&
			     (image.binding.shader_write || image.binding.force_general))) {
				continue;
			}
			Bring(image, element.layout, element.range);
		}
		Bring(cache.m_slot_images[m_null[array].id], vk::ImageLayout::eGeneral,
		      m_null[array].range);
	}
	if (!m_barriers.empty()) {
		m_context.GetCommandScheduler().EndRendering();
		vk::DependencyInfo dependency {};
		dependency.imageMemoryBarrierCount = static_cast<uint32_t>(m_barriers.size());
		dependency.pImageMemoryBarriers    = m_barriers.data();
		command.pipelineBarrier2(dependency);
	}
	m_current.last_used = m_context.GetCommandScheduler().CurrentTick();
	return m_current.set;
}

} // namespace Libs::Graphics
