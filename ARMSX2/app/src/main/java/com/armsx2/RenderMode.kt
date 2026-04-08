package com.armsx2

enum class RenderMode(val id: Int) {
    // VULKAN_HW(14), // TODO: re-enable once VK HW blending bugs are fixed (see RenderModeButton)
    VULKAN_SW(13),
    OPENGL(12)
}