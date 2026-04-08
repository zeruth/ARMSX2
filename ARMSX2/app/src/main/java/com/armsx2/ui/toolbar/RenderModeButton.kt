package com.armsx2.ui.toolbar

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.vector.ImageVector
import com.armsx2.Main
import com.armsx2.RenderMode
import compose.icons.LineAwesomeIcons
import compose.icons.lineawesomeicons.Dashcube

class RenderModeButton : ToolbarButton() {
    // VK HW is currently disabled because of unresolved blending visual bugs (BIOS pillars/SCEA
    // text show black boxes when AccBlendLevel is Full). The VULKAN_HW enum entry and the
    // VK HW branches below are intentionally left commented out so this can be re-enabled
    // once the HW blend issue is fixed.
    var renderMode = RenderMode.VULKAN_SW

    override var icon = mutableStateOf<ImageVector?>(LineAwesomeIcons.Dashcube)
    override fun isVisible(): Boolean {
        return true
    }

    override fun action() {
        when (renderMode) {
            // RenderMode.VULKAN_HW -> {
            //     renderMode = RenderMode.VULKAN_SW
            //     Main.renderSoftware()
            // }
            RenderMode.VULKAN_SW -> {
                renderMode = RenderMode.OPENGL
                Main.renderOpenGL()
            }
            RenderMode.OPENGL -> {
                // TODO: cycle back to RenderMode.VULKAN_HW once VK HW blending is fixed.
                renderMode = RenderMode.VULKAN_SW
                Main.renderSoftware()
            }
        }
    }
}