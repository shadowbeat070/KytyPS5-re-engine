#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSIMAGEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSIMAGEHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/bindlessTranslation.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"

#include <array>
#include <cstdint>
#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class RenderContext;
class RenderExecutor;
struct GraphicContext;

class BindlessImageHeap {
public:
	BindlessImageHeap(RenderContext& context, RenderExecutor& executor);
	~BindlessImageHeap();
	KYTY_CLASS_NO_COPY(BindlessImageHeap);

	struct Region {
		uint32_t base  = 0;
		uint32_t count = 0;
	};

	void                 Prepare(const ShaderRecompiler::IR::BindlessImageTable& table);
	[[nodiscard]] Region Lookup(const ShaderRecompiler::IR::BindlessImageTable& table) const;
	[[nodiscard]] vk::DescriptorSet Commit(vk::CommandBuffer command);

	[[nodiscard]] static vk::DescriptorSetLayout SetLayout(GraphicContext& graphics);
	[[nodiscard]] static vk::DescriptorSetLayout EmptySetLayout(GraphicContext& graphics);
	[[nodiscard]] static std::array<
	    vk::DescriptorSetLayoutBinding,
	    1u + static_cast<size_t>(ShaderRecompiler::IR::BindlessShape::Count)>
	LayoutBindings(vk::ShaderStageFlags stages);

private:
	static constexpr uint32_t MaxHeapRecords     = 1u << 18u;
	static constexpr uint32_t MaxRescansPerFrame = 4u;
	static constexpr uint64_t HeapIdleFrames     = 600u;
	static constexpr size_t   Arrays =
	    static_cast<size_t>(ShaderRecompiler::IR::BindlessShape::Count);

	struct HeapKey {
		uint64_t base   = 0;
		uint64_t size   = 0;
		uint32_t stride = 0;
		uint32_t offset = 0;

		bool operator==(const HeapKey&) const = default;
	};
	struct HeapKeyHash {
		size_t operator()(const HeapKey& key) const noexcept;
	};
	struct Heap {
		uint32_t              records       = 0;
		uint64_t              hash          = 0;
		uint64_t              checked_frame = UINT64_MAX;
		uint64_t              rescan_frame  = UINT64_MAX;
		uint64_t              used_frame    = 0;
		uint32_t              rescans       = 0;
		bool                  scanned       = false;
		bool                  stale         = true;
		uint32_t              arena_base    = 0;
		std::vector<uint32_t> words;
	};
	struct Element {
		ImageId                 id;
		vk::ImageView           view = nullptr;
		TextureCache::ImageDesc desc;
		ImageSubresourceRange   range;
		vk::ImageLayout         layout = vk::ImageLayout::eUndefined;
		bool                    live   = false;
	};
	struct Version {
		vk::DescriptorSet       set  = nullptr;
		vk::DescriptorPool      pool = nullptr;
		std::shared_ptr<Buffer> arena;
		uint64_t                last_used = 0;
	};

	[[nodiscard]] static HeapKey KeyOf(const ShaderRecompiler::IR::BindlessImageTable& table);
	void                         EnsureNullImages();
	[[nodiscard]] bool           ElementAlive(const Element& element);
	void                         Kill(ShaderRecompiler::IR::BindlessShape array, uint32_t slot);
	void                         SweepDeadElements();
	void                         TouchLiveElements();
	void                         EvictIdleHeaps(uint64_t frame);
	void                         RefreshLiveElements();
	[[nodiscard]] bool           ReadHeap(const HeapKey& key, uint32_t records,
	                                      std::vector<uint32_t>& dwords) const;
	[[nodiscard]] bool Rescan(const HeapKey& key, Heap& heap, const std::vector<uint32_t>& dwords);
	void               ResetElements();
	void               BuildArena();
	void               BuildVersion();
	void               RetireVersions();
	[[nodiscard]] vk::DescriptorSet AllocateSet(vk::DescriptorPool& pool);

	RenderContext&                                           m_context;
	RenderExecutor&                                          m_executor;
	std::unordered_map<HeapKey, Heap, HeapKeyHash>           m_heaps;
	Bindless::TranslationCache                               m_cache;
	Bindless::SlotAllocator                                  m_slots;
	std::array<std::vector<Element>, Arrays>                 m_elements;
	std::array<std::vector<vk::DescriptorImageInfo>, Arrays> m_infos;
	std::array<Element, Arrays>                              m_null;
	bool                                                     m_null_ready = false;
	std::shared_ptr<Buffer>                                  m_arena;
	uint32_t                                                 m_arena_words     = 0;
	bool                                                     m_arena_dirty     = true;
	bool                                                     m_set_dirty       = true;
	uint64_t                                                 m_refreshed_frame = UINT64_MAX;
	uint64_t                                                 m_evicted_frame   = 0;
	uint64_t                                                 m_touched_gc_tick = UINT64_MAX;
	Version                                                  m_current;
	std::deque<Version>                                      m_retired;
	std::vector<vk::DescriptorPool>                          m_pools;
	std::vector<vk::ImageMemoryBarrier2>                     m_barriers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDLESSIMAGEHEAP_H_
