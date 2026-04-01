package com.pasx2.ui.toolbar

import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.vector.ImageVector
import com.pasx2.Main
import com.pasx2.RenderMode
import compose.icons.LineAwesomeIcons
import compose.icons.lineawesomeicons.Dashcube
import compose.icons.lineawesomeicons.FolderOpen

class RenderModeButton : ToolbarButton() {
    var renderMode = RenderMode.VULKAN_HW

    override var icon = mutableStateOf<ImageVector?>(LineAwesomeIcons.Dashcube)
    override fun isVisible(): Boolean {
        return true
    }

    override fun action() {
        when (renderMode) {
            RenderMode.VULKAN_HW -> {
                renderMode = RenderMode.VULKAN_SW
                Main.renderSoftware()
            }
            //TODO: Cycle ALL renderers
            RenderMode.VULKAN_SW -> {
                renderMode = RenderMode.OPENGL
                Main.renderOpenGL()
            }
            RenderMode.OPENGL -> {
                renderMode = RenderMode.VULKAN_SW
                Main.renderSoftware()
            }
        }
    }
}